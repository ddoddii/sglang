from __future__ import annotations

"""Idle KV parking (PD disaggregation).

During tool-call idle windows in agentic multi-turn workloads, a decode (D) node's
conversation prefix KV is parked into an idle prefill (P) node's GPU radix cache over
NVLink (CUDA IPC + P2P), so the next turn prefix-hits on P instead of recomputing.
See docs/developer_guide/idle_kv_parking_design.md.

Incremental build:
  slice 1: scaffolding + flag wiring (done).
  slice 2a (this): P<->D CUDA IPC handle exchange + verification (prove that P can
                   read D's KV pool over NVLink in the real, separate sglang
                   processes). Rendezvous is a shared file under /dev/shm because
                   NVLink parking is inherently intra-node (NVLink pairs live in one
                   box), so no network bootstrap is needed.
  slice 2b: page-gather P2P copy of a prefix D->P.
  slice 2c/3/4: park trigger on idle + radix insert.

IMPORTANT (2a): CUDA IPC requires each process to *see* the peer GPU. Launch P and D
without CUDA_VISIBLE_DEVICES isolation and place them with --base-gpu-id instead,
otherwise _new_shared_cuda cannot open a handle for an invisible device.
"""

import logging
import os
import pickle
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Same-node rendezvous directory (NVLink parking is intra-node). Overridable for tests.
PARK_DIR = os.environ.get("SGLANG_KV_PARK_DIR", "/dev/shm/sglang_kv_parking")
# Decode publishes its KV-pool IPC handles here; prefill consumes them.
DECODE_IPC_FILE = os.path.join(PARK_DIR, "decode_kvpool_ipc.pkl")
RENDEZVOUS_TIMEOUT_S = 180


def _export_ipc(tensor: torch.Tensor) -> dict:
    """Export a CUDA tensor as a picklable IPC descriptor (mirrors mm_utils.py)."""
    storage = tensor.untyped_storage()
    return {
        "handle": storage._share_cuda_(),
        "shape": tuple(tensor.shape),
        "dtype": tensor.dtype,
        "stride": tuple(tensor.stride()),
        "device_index": tensor.device.index,
        "storage_offset": tensor.storage_offset(),
    }


def _open_ipc(desc: dict) -> torch.Tensor:
    """Reconstruct a tensor from an IPC descriptor produced by _export_ipc.

    The tensor lives on the *producer's* physical device (desc['device_index']),
    which must be visible to this process. copy_ from it to a local tensor rides P2P.
    """
    dev = torch.device(f"cuda:{desc['device_index']}")
    with torch.cuda.device(dev):
        storage = torch.UntypedStorage._new_shared_cuda(*desc["handle"])
        return torch.empty(0, dtype=desc["dtype"], device=dev).set_(
            storage,
            storage_offset=desc["storage_offset"],
            size=desc["shape"],
            stride=desc["stride"],
        )


class IdleKVParkManager:
    """Manage idle-time KV parking between decode and prefill nodes.

    One instance lives on each PD node; ``role`` ("prefill" or "decode") selects the
    side-specific behavior.
    """

    def __init__(
        self,
        *,
        role: str,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        tp_group: torch.distributed.ProcessGroup,
        tree_cache: BasePrefixCache,
        server_args: ServerArgs,
    ) -> None:
        assert role in ("prefill", "decode"), role
        self.role = role
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.tp_group = tp_group
        self.tree_cache = tree_cache
        self.server_args = server_args
        self.page_size = server_args.page_size
        self.gpu_id = server_args.base_gpu_id

        self.kv_cache = token_to_kv_pool_allocator.get_kvcache()
        self.k_buffer: Optional[List[torch.Tensor]] = getattr(self.kv_cache, "k_buffer", None)
        self.v_buffer: Optional[List[torch.Tensor]] = getattr(self.kv_cache, "v_buffer", None)

        # Peer KV pool mapped via IPC (filled on prefill side in slice 2a).
        self.peer_k_buffer: Optional[List[torch.Tensor]] = None
        self.peer_v_buffer: Optional[List[torch.Tensor]] = None
        self.peer_ready = threading.Event()

        os.makedirs(PARK_DIR, exist_ok=True)
        logger.info(
            "Idle KV parking enabled (role=%s, gpu_id=%s, layers=%s, page_size=%s); "
            "slice 2a: CUDA IPC handle exchange via %s",
            role,
            self.gpu_id,
            len(self.k_buffer) if self.k_buffer else "n/a",
            self.page_size,
            PARK_DIR,
        )

        if self.k_buffer is None or self.v_buffer is None:
            logger.warning(
                "Idle KV parking: unsupported KV cache type %s (no k_buffer/v_buffer); "
                "disabling.",
                type(self.kv_cache).__name__,
            )
            return

        # Run IPC setup off the hot path so server startup is not blocked.
        threading.Thread(
            target=self._setup_2a, name="idle-kv-park-2a", daemon=True
        ).start()

    # --- slice 2a: IPC handle exchange + verification --------------------------
    def _setup_2a(self) -> None:
        try:
            if self.role == "decode":
                self._decode_publish_ipc()
            else:
                self._prefill_consume_ipc()
        except Exception as e:  # noqa: BLE001
            logger.error("Idle KV parking 2a setup failed (role=%s): %r", self.role, e)

    def _decode_publish_ipc(self) -> None:
        """D: export KV-pool buffer handles + a verification tensor to the rendezvous."""
        torch.cuda.set_device(self.gpu_id)
        # A dedicated verification tensor with a known pattern (avoids touching live KV).
        verify = torch.arange(4096, dtype=torch.float16, device=f"cuda:{self.gpu_id}")
        verify_checksum = float(verify.double().sum().item())
        torch.cuda.synchronize(self.gpu_id)

        payload = {
            "role": "decode",
            "gpu_id": self.gpu_id,
            "num_layers": len(self.k_buffer),
            "page_size": self.page_size,
            "kv_shape": tuple(self.k_buffer[0].shape),
            "kv_dtype": str(self.k_buffer[0].dtype),
            "verify": _export_ipc(verify),
            "verify_checksum": verify_checksum,
            "verify_numel": verify.numel(),
            "k_handles": [_export_ipc(t) for t in self.k_buffer],
            "v_handles": [_export_ipc(t) for t in self.v_buffer],
            "ts": time.time(),
        }
        # Keep verify alive for the peer's lifetime.
        self._verify_keepalive = verify

        tmp = DECODE_IPC_FILE + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, DECODE_IPC_FILE)  # atomic publish
        logger.info(
            "Idle KV parking [decode]: published %d k + %d v IPC handles + verify "
            "(checksum=%.1f) to %s",
            len(self.k_buffer),
            len(self.v_buffer),
            verify_checksum,
            DECODE_IPC_FILE,
        )

    def _prefill_consume_ipc(self) -> None:
        """P: open D's handles, verify NVLink P2P read, keep KV-pool mapping for 2b."""
        torch.cuda.set_device(self.gpu_id)
        deadline = time.time() + RENDEZVOUS_TIMEOUT_S
        while not os.path.exists(DECODE_IPC_FILE):
            if time.time() > deadline:
                logger.warning(
                    "Idle KV parking [prefill]: no decode IPC file at %s after %ds; "
                    "peer may not be up. Parking inactive.",
                    DECODE_IPC_FILE,
                    RENDEZVOUS_TIMEOUT_S,
                )
                return
            time.sleep(1.0)

        with open(DECODE_IPC_FILE, "rb") as f:
            payload = pickle.load(f)

        peer_gpu = payload["gpu_id"]
        can_p2p = torch.cuda.can_device_access_peer(self.gpu_id, peer_gpu)

        # 1) Verify correctness over IPC + P2P using the dedicated verify tensor.
        verify_peer = _open_ipc(payload["verify"])
        local = torch.empty_like(verify_peer, device=f"cuda:{self.gpu_id}")
        local.copy_(verify_peer)  # D_gpu -> P_gpu, P2P/NVLink
        torch.cuda.synchronize(self.gpu_id)
        got = float(local.double().sum().item())
        want = payload["verify_checksum"]
        ok = abs(got - want) < 1.0

        # 2) Time a real KV-sized P2P read from D's pool (bandwidth in situ).
        self.peer_k_buffer = [_open_ipc(h) for h in payload["k_handles"]]
        self.peer_v_buffer = [_open_ipc(h) for h in payload["v_handles"]]
        bw_gbps = self._bench_kv_read()

        if ok:
            self.peer_ready.set()
        logger.info(
            "Idle KV parking [prefill]: opened peer KV pool (gpu%s<-gpu%s, p2p=%s, "
            "%d layers). Verify checksum %s (got=%.1f want=%.1f). KV P2P read ~%.1f GB/s. "
            "%s",
            self.gpu_id,
            peer_gpu,
            can_p2p,
            len(self.peer_k_buffer),
            "MATCH" if ok else "MISMATCH",
            got,
            want,
            bw_gbps,
            "NVLink cross-process IPC verified -> ready for 2b."
            if (ok and can_p2p)
            else "WARNING: verification/p2p not clean; investigate.",
        )

    def _bench_kv_read(self, iters: int = 20) -> float:
        """Copy one layer's k_buffer from the peer pool to a local buffer, timed."""
        try:
            src = self.peer_k_buffer[0]
            dst = torch.empty(src.shape, dtype=src.dtype, device=f"cuda:{self.gpu_id}")
            for _ in range(3):
                dst.copy_(src)
            torch.cuda.synchronize(self.gpu_id)
            t0 = time.perf_counter()
            for _ in range(iters):
                dst.copy_(src)
            torch.cuda.synchronize(self.gpu_id)
            ms = (time.perf_counter() - t0) / iters * 1000.0
            nbytes = src.numel() * src.element_size()
            return nbytes / (ms / 1000.0) / 1e9
        except Exception as e:  # noqa: BLE001
            logger.warning("Idle KV parking [prefill]: KV read bench failed: %r", e)
            return -1.0

    # --- decode side (slice 3): park a finished/idle request -------------------
    def park(self, req: "Req") -> bool:
        return False

    # --- prefill side (slice 4): receive parked KV + radix insert --------------
    def poll_incoming(self) -> None:
        return None

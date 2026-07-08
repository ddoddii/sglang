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
import queue
import threading
import time
from typing import TYPE_CHECKING, List, Optional

import torch
import zmq

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_zmq_socket

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)

# Same-node rendezvous directory (NVLink parking is intra-node). Overridable for tests.
PARK_DIR = os.environ.get("SGLANG_KV_PARK_DIR", "/dev/shm/sglang_kv_parking")
# Dedicated park pool GPU (slice 4). If set, prefill parks into a separate buffer on
# this idle GPU instead of its own radix, so parked entries survive P-GPU pressure.
# (Should move to environ.py for upstream; os.environ for research iteration.)
_PARK_GPU_ENV = os.environ.get("SGLANG_KV_PARK_GPU")
PARK_GPU = int(_PARK_GPU_ENV) if _PARK_GPU_ENV not in (None, "") else None
PARK_POOL_TOKENS = int(os.environ.get("SGLANG_KV_PARK_POOL_TOKENS", "100000"))
# Decode publishes its KV-pool IPC handles here; prefill consumes them.
DECODE_IPC_FILE = os.path.join(PARK_DIR, "decode_kvpool_ipc.pkl")
# Prefill publishes its ZMQ park-control PULL address here; decode connects a PUSH.
PREFILL_ZMQ_FILE = os.path.join(PARK_DIR, "prefill_park_zmq.pkl")
RENDEZVOUS_TIMEOUT_S = 180

# slice 2b self-test: number of KV slots to round-trip D->P to validate the
# multi-layer indexed gather-copy over NVLink. Uses free slots [1..N] at startup
# (slot 0 is the padded dummy); the allocator overwrites them on real use.
SELFTEST_N_SLOTS = 64


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

        # ZMQ park control channel state (slice 3a).
        self._zmq_ctx = None
        self._pull = None  # prefill: receives park messages
        self._push = None  # decode: sends park messages
        # prefill: park messages queued by the recv thread, drained on the main
        # scheduler thread by poll_incoming() (the allocator is not thread-safe).
        self._incoming: "queue.Queue[dict]" = queue.Queue()
        self._parked_count = 0
        self._parked_tokens = 0
        # diagnostics (why parking helps or not): skip vs copy, how much P already had,
        # and whether parked prefixes survive in the radix until they could be hit.
        self._skipped_count = 0
        self._copied_count = 0
        self._existing_sum = 0
        self._n_sum = 0
        self._received_msgs = 0  # park messages enqueued by the recv thread

        # slice 4: dedicated park pool on an idle GPU (survives P-GPU eviction).
        self.park_gpu = PARK_GPU
        self._park_k = None
        self._park_v = None
        self._park_index = None  # OrderedDict: hash(token_ids) -> (start_slot, n)
        self._park_next = 0
        if self.role == "prefill" and self.park_gpu is not None and self.k_buffer is not None:
            self._init_park_gpu_pool()
        from collections import deque as _deque

        self._recent_parked = _deque(maxlen=16)  # (token_ids, n) samples for survival probe

        # Run setup off the hot path so server startup is not blocked.
        threading.Thread(
            target=self._setup, name="idle-kv-park-setup", daemon=True
        ).start()

    def _setup(self) -> None:
        try:
            if self.role == "decode":
                self._decode_publish_ipc()   # 2a: publish KV-pool IPC handles
                self._decode_connect_zmq()   # 3a: connect to prefill's park channel
            else:
                self._prefill_setup_zmq()    # 3a: bind park channel + start receiver
                self._prefill_consume_ipc()  # 2a/2b: open D's IPC, verify
        except Exception as e:  # noqa: BLE001
            logger.error("Idle KV parking setup failed (role=%s): %r", self.role, e)

    def _decode_publish_ipc(self) -> None:
        """D: export KV-pool buffer handles + a verification tensor to the rendezvous."""
        torch.cuda.set_device(self.gpu_id)
        # A dedicated verification tensor with a known pattern (avoids touching live KV).
        verify = torch.arange(4096, dtype=torch.float16, device=f"cuda:{self.gpu_id}")
        verify_checksum = float(verify.double().sum().item())
        torch.cuda.synchronize(self.gpu_id)

        # slice 2b: dedicated race-free test buffers (KV-shaped) with a known pattern,
        # so the prefill side validates the multi-layer indexed gather-copy without
        # racing the live pool (warmup/serving can overwrite live slots).
        st_indices = list(range(1, 1 + SELFTEST_N_SLOTS))
        st_k, st_v, st_checksum = self._build_selftest_buffers(st_indices)
        self._st_keepalive = (st_k, st_v)  # keep alive for the peer's lifetime

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
            "selftest_indices": st_indices,
            "selftest_checksum": st_checksum,
            "st_k_handles": [_export_ipc(t) for t in st_k],
            "st_v_handles": [_export_ipc(t) for t in st_v],
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
            "(checksum=%.1f) + selftest(%d slots, checksum=%.1f) to %s",
            len(self.k_buffer),
            len(self.v_buffer),
            verify_checksum,
            len(st_indices),
            st_checksum,
            DECODE_IPC_FILE,
        )

    def _build_selftest_buffers(self, indices):
        """Allocate dedicated KV-shaped test buffers, write a known per-slot pattern,
        and return (k_bufs, v_bufs, checksum). Race-free: nothing else touches them."""
        head_num, head_dim = self.k_buffer[0].shape[1], self.k_buffer[0].shape[2]
        dtype = self.k_buffer[0].dtype
        dev = f"cuda:{self.gpu_id}"
        n = max(indices) + 1
        k_bufs = [torch.zeros(n, head_num, head_dim, dtype=dtype, device=dev) for _ in self.k_buffer]
        v_bufs = [torch.zeros(n, head_num, head_dim, dtype=dtype, device=dev) for _ in self.v_buffer]
        idx = torch.tensor(indices, dtype=torch.long, device=dev)
        vals = ((idx % 50) + 1).to(dtype)  # small ints, exact in fp16/bf16
        total = 0.0
        for layer in range(len(k_bufs)):
            for buf in (k_bufs[layer], v_bufs[layer]):
                buf[idx] = vals.view(-1, *([1] * (buf.dim() - 1)))
                total += float(buf[idx].float().sum().item())
        torch.cuda.synchronize(self.gpu_id)
        return k_bufs, v_bufs, total

    def _p2p_gather(self, src_k, src_v, dst_k, dst_v, src_indices, dst_indices) -> None:
        """Copy slots src_indices (src_k/src_v, peer device) -> dst_indices
        (dst_k/dst_v, local device) across all layers over NVLink P2P.

        Cross-device indexed assignment is unsupported, so gather on the peer device,
        .to() the local device (the P2P copy), then scatter locally."""
        local_dev = f"cuda:{self.gpu_id}"
        peer_dev = f"cuda:{src_k[0].device.index}"
        s = torch.tensor(src_indices, dtype=torch.long, device=peer_dev)
        d = torch.tensor(dst_indices, dtype=torch.long, device=local_dev)
        for layer in range(len(src_k)):
            dst_k[layer][d] = src_k[layer][s].to(local_dev)
            dst_v[layer][d] = src_v[layer][s].to(local_dev)
        torch.cuda.synchronize(self.gpu_id)

    def _gather_copy_from_peer(self, src_indices, dst_indices) -> None:
        """Real park copy (slice 3): peer KV pool slots -> local KV pool slots."""
        self._p2p_gather(
            self.peer_k_buffer, self.peer_v_buffer,
            self.k_buffer, self.v_buffer,
            src_indices, dst_indices,
        )

    def _run_2b_selftest(self, payload) -> None:
        """P: gather-copy the decode's dedicated test buffers into local test buffers
        and verify the checksum (race-free validation of the copy primitive)."""
        st_indices = payload.get("selftest_indices")
        want = payload.get("selftest_checksum")
        if not st_indices or want is None or "st_k_handles" not in payload:
            logger.warning("Idle KV parking [prefill]: no selftest payload; skip 2b check.")
            return
        peer_st_k = [_open_ipc(h) for h in payload["st_k_handles"]]
        peer_st_v = [_open_ipc(h) for h in payload["st_v_handles"]]
        dev = f"cuda:{self.gpu_id}"
        local_st_k = [torch.zeros(t.shape, dtype=t.dtype, device=dev) for t in peer_st_k]
        local_st_v = [torch.zeros(t.shape, dtype=t.dtype, device=dev) for t in peer_st_v]

        t0 = time.perf_counter()
        self._p2p_gather(peer_st_k, peer_st_v, local_st_k, local_st_v, st_indices, st_indices)
        ms = (time.perf_counter() - t0) * 1000.0

        idx = torch.tensor(st_indices, dtype=torch.long, device=dev)
        got = 0.0
        for layer in range(len(local_st_k)):
            got += float(local_st_k[layer][idx].float().sum().item())
            got += float(local_st_v[layer][idx].float().sum().item())
        ok = abs(got - want) <= max(1.0, abs(want) * 1e-3)  # relative tol for fp32 sums
        nbytes = (
            len(st_indices)
            * peer_st_k[0][0].numel()
            * peer_st_k[0].element_size()
            * 2
            * len(peer_st_k)
        )
        logger.info(
            "Idle KV parking [prefill] 2b selftest: gather-copy %d slots x %d layers "
            "(k+v) %s (got=%.1f want=%.1f) in %.2fms (%.1f MB, ~%.1f GB/s). %s",
            len(st_indices),
            len(local_st_k),
            "MATCH" if ok else "MISMATCH",
            got,
            want,
            ms,
            nbytes / 1e6,
            (nbytes / (ms / 1000.0) / 1e9) if ms > 0 else -1.0,
            "indexed gather-copy over NVLink verified -> ready for slice 3."
            if ok
            else "WARNING: gather-copy mismatch; investigate.",
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

        # slice 2b: validate the real multi-layer indexed gather-copy from D's pool.
        if ok and can_p2p:
            try:
                self._run_2b_selftest(payload)
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [prefill] 2b selftest failed: %r", e)

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

    # --- slice 3a: ZMQ park control channel (D pushes park messages to P) -------
    def _prefill_setup_zmq(self) -> None:
        """P: bind a PULL socket, publish its address, and start the receiver loop."""
        self._zmq_ctx = zmq.Context(1)
        port, self._pull = get_zmq_socket(self._zmq_ctx, zmq.PULL, endpoint=None)
        addr = f"tcp://127.0.0.1:{port}"
        tmp = PREFILL_ZMQ_FILE + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump({"addr": addr, "gpu_id": self.gpu_id, "ts": time.time()}, f)
        os.replace(tmp, PREFILL_ZMQ_FILE)
        threading.Thread(
            target=self._prefill_recv_loop, name="idle-kv-park-recv", daemon=True
        ).start()
        logger.info(
            "Idle KV parking [prefill]: park control channel PULL bound at %s "
            "(published to %s)",
            addr,
            PREFILL_ZMQ_FILE,
        )

    def _prefill_recv_loop(self) -> None:
        while True:
            try:
                msg = pickle.loads(self._pull.recv())
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [prefill]: recv failed: %r", e)
                return
            try:
                self._handle_park_message(msg)
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [prefill]: handle msg failed: %r", e)

    def _handle_park_message(self, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "ping":
            logger.info(
                "Idle KV parking [prefill]: park control channel OK — received %s "
                "from %s (gpu%s). Ready for slice 3b park messages.",
                mtype,
                msg.get("from"),
                msg.get("gpu_id"),
            )
        elif mtype == "park":
            # Enqueue; the copy/insert runs on the scheduler main thread (poll_incoming).
            self._received_msgs += 1
            self._incoming.put(msg)
        else:
            logger.warning("Idle KV parking [prefill]: unknown msg type %r", mtype)

    def _decode_connect_zmq(self) -> None:
        """D: read prefill's PULL address, connect a PUSH, and send a test ping."""
        deadline = time.time() + RENDEZVOUS_TIMEOUT_S
        while not os.path.exists(PREFILL_ZMQ_FILE):
            if time.time() > deadline:
                logger.warning(
                    "Idle KV parking [decode]: no prefill ZMQ file at %s after %ds; "
                    "park control channel inactive.",
                    PREFILL_ZMQ_FILE,
                    RENDEZVOUS_TIMEOUT_S,
                )
                return
            time.sleep(1.0)
        with open(PREFILL_ZMQ_FILE, "rb") as f:
            info = pickle.load(f)
        addr = info["addr"]
        self._zmq_ctx = zmq.Context(1)
        self._push = get_zmq_socket(self._zmq_ctx, zmq.PUSH, endpoint=addr, bind=False)
        self._push.send(
            pickle.dumps(
                {"type": "ping", "from": "decode", "gpu_id": self.gpu_id, "ts": time.time()}
            )
        )
        logger.info(
            "Idle KV parking [decode]: park control channel PUSH connected to %s, "
            "sent ping.",
            addr,
        )

    # --- decode side (slice 3b): park a finished request -----------------------
    def park(self, req: "Req") -> bool:
        """D: send the finished request's prefix (token ids + KV slot indices) to P.

        Called at request completion, before release_kv_cache frees the slots. The
        KV is still valid at send time; P copies it when it drains the message.
        NOTE (slice 3c): correctness under reuse needs an ack so D holds the slots
        until P has copied; for now the tool-call idle gap keeps them valid at low load.
        """
        if self.role != "decode" or self._push is None:
            return False
        if getattr(req, "req_pool_idx", -1) == -1:
            return False
        try:
            token_indices = self.req_to_token_pool.req_to_token[req.req_pool_idx]
            token_ids = list(req.origin_input_ids) + list(req.output_ids)
            n = (len(token_ids) // self.page_size) * self.page_size
            if n == 0 or token_indices.numel() < n:
                return False
            kv_indices = token_indices[:n].detach().to("cpu", torch.int64).tolist()
            self._push.send(
                pickle.dumps(
                    {
                        "type": "park",
                        "rid": req.rid,
                        "token_ids": token_ids[:n],
                        "kv_indices": kv_indices,
                    }
                )
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.error("Idle KV parking [decode]: park failed rid=%s: %r",
                         getattr(req, "rid", "?"), e)
            return False

    # --- prefill side (slice 3b/3c): drain parked messages on the main thread --
    def poll_incoming(self, max_msgs: int = 4) -> None:
        """P: drain up to max_msgs parked prefixes, copy their KV from D over NVLink.

        Runs on the scheduler main thread (allocator is not thread-safe). Slice 3b
        copies + frees (validation); slice 3c will radix-insert instead of free.
        """
        # Gate on peer_ready so both peer_k_buffer and peer_v_buffer are fully mapped
        # and verified before we touch them.
        if self.role != "prefill" or not self.peer_ready.is_set():
            return
        for _ in range(max_msgs):
            try:
                msg = self._incoming.get_nowait()
            except queue.Empty:
                return
            try:
                self._receive_park(msg)
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [prefill]: receive park failed: %r", e)

    # --- slice 4: dedicated idle-GPU park pool -------------------------------
    def _init_park_gpu_pool(self) -> None:
        from collections import OrderedDict

        dev = f"cuda:{self.park_gpu}"
        head_num, head_dim = self.k_buffer[0].shape[1], self.k_buffer[0].shape[2]
        dtype = self.k_buffer[0].dtype
        L = len(self.k_buffer)
        N = PARK_POOL_TOKENS
        self._park_k = [torch.zeros(N, head_num, head_dim, dtype=dtype, device=dev) for _ in range(L)]
        self._park_v = [torch.zeros(N, head_num, head_dim, dtype=dtype, device=dev) for _ in range(L)]
        self._park_index = OrderedDict()  # hash -> (start, n)
        self._park_next = 0
        gb = 2 * N * head_num * head_dim * self.k_buffer[0].element_size() * L / 1e9
        logger.info(
            "Idle KV parking [prefill]: dedicated park pool on GPU%d = %d tokens x %d "
            "layers (~%.1f GB). Parked entries survive P-GPU eviction.",
            self.park_gpu, N, L, gb,
        )

    def _gather_copy_peer_to_park(self, src_indices, start: int, n: int) -> None:
        """Copy peer(D) KV slots -> the park pool on the idle GPU, across all layers."""
        park_dev = f"cuda:{self.park_gpu}"
        peer_dev = f"cuda:{self.peer_k_buffer[0].device.index}"
        s = torch.tensor(src_indices, dtype=torch.long, device=peer_dev)
        for layer in range(len(self.peer_k_buffer)):
            self._park_k[layer][start : start + n] = self.peer_k_buffer[layer][s].to(park_dev)
            self._park_v[layer][start : start + n] = self.peer_v_buffer[layer][s].to(park_dev)
        torch.cuda.synchronize(self.park_gpu)

    def _park_to_gpu(self, token_ids, src_indices, n: int) -> None:
        """Store the prefix KV in the dedicated idle-GPU pool (ring buffer + LRU index)."""
        if n > PARK_POOL_TOKENS:
            return
        h = hash(tuple(token_ids))
        if h in self._park_index:
            self._park_index.move_to_end(h)
            self._skipped_count += 1  # already parked
            return
        start = self._park_next
        if start + n > PARK_POOL_TOKENS:
            start = 0  # wrap
        end = start + n
        # Evict index entries whose slots the new write overlaps.
        for k in list(self._park_index.keys()):
            s0, ln = self._park_index[k]
            if not (s0 + ln <= start or s0 >= end):
                del self._park_index[k]
        t0 = time.perf_counter()
        self._gather_copy_peer_to_park(src_indices, start, n)
        ms = (time.perf_counter() - t0) * 1000.0
        self._park_index[h] = (start, n)
        self._park_next = end % PARK_POOL_TOKENS
        self._copied_count += 1
        self._n_sum += n
        if self._copied_count % 8 == 0:
            self._recent_parked.append((h, n))
        if self._parked_count <= 5 or self._copied_count % 50 == 0:
            self._parked_count += 1
            logger.info(
                "Idle KV parking [prefill] GPU%d-park: rid=%s, %d tok x %d layers in "
                "%.1fms. index=%d entries, next=%d.",
                self.park_gpu, "?", n, len(self.peer_k_buffer), ms,
                len(self._park_index), self._park_next,
            )

    def _receive_park(self, msg: dict) -> None:
        src_indices = msg["kv_indices"]
        token_ids = msg["token_ids"]
        n = len(src_indices)
        if n == 0 or n != len(token_ids):
            return

        # slice 4: park into a dedicated idle-GPU pool that survives P-GPU pressure.
        if self.park_gpu is not None:
            self._park_to_gpu(token_ids, src_indices, n)
            self._maybe_diag()
            return

        # slice 3d: only park what P is missing. P already caches the prompt prefix it
        # prefilled itself; parking's real contribution is the tokens P lacks (the
        # decode-generated tail, or the whole prefix once P has evicted it under
        # pressure). Copy only [existing:n] -> huge saving vs copying the full prefix.
        key = RadixKey(list(token_ids), extra_key=None)
        existing = len(self.tree_cache.match_prefix(key).device_indices)
        if existing >= n:
            self._skipped_count += 1  # P already had the whole prefix (no eviction)
            self._maybe_diag()
            return  # nothing to park.

        dst = self.token_to_kv_pool_allocator.alloc(n)
        if dst is None:
            logger.debug("Idle KV parking [prefill]: no space to park rid=%s (%d tok)",
                         msg.get("rid"), n)
            return
        dst_list = dst.detach().to("cpu", torch.int64).tolist()
        # From here dst is allocated: any failure must free it, or the KV pool leaks.
        inserted_into_tree = False
        try:
            t0 = time.perf_counter()
            # Copy only the missing tail; dst[:existing] stays uninitialized but is
            # freed below (insert matches that prefix and never reads those values).
            self._gather_copy_from_peer(src_indices[existing:n], dst_list[existing:n])
            ms = (time.perf_counter() - t0) * 1000.0

            new_prefix_len = self.tree_cache.insert(key, dst.to(torch.int64))
            inserted_into_tree = True  # tree now owns dst[new_prefix_len:]
            if new_prefix_len > 0:
                self.token_to_kv_pool_allocator.free(dst[:new_prefix_len])
        except Exception:
            if not inserted_into_tree:
                self.token_to_kv_pool_allocator.free(dst)  # rollback whole alloc
            raise
        inserted = n - new_prefix_len

        self._parked_count += 1
        self._parked_tokens += inserted
        self._copied_count += 1
        self._existing_sum += existing
        self._n_sum += n
        if self._copied_count % 8 == 0:
            self._recent_parked.append((list(token_ids), n))
        if self._parked_count <= 5 or self._parked_count % 50 == 0:
            logger.info(
                "Idle KV parking [prefill]: parked+inserted rid=%s, %d tok "
                "(P had %d, copied+inserted %d new) x %d layers in %.1fms. "
                "total parked=%d, tokens inserted=%d.",
                msg.get("rid"),
                n,
                existing,
                inserted,
                len(self.k_buffer),
                ms,
                self._parked_count,
                self._parked_tokens,
            )
        self._maybe_diag()

    def _maybe_diag(self, every: int = 10) -> None:
        """Periodic diagnostic to explain whether parking can help.

        Distinguishes: H1 P retains prefix (high skip / high existing-fraction);
        H2 parked entries evicted before hit (low survival); H3 parked but never
        matched (high survival yet no reuse gain); plus backlog (recv >> processed
        means the copy can't keep up with the park rate under load).
        """
        total = self._skipped_count + self._copied_count
        if total == 0 or total % every != 0:
            return
        skip_rate = self._skipped_count / total
        existing_frac = (self._existing_sum / self._n_sum) if self._n_sum else 0.0
        # survival: are recently parked prefixes still present?
        # park-GPU mode -> still in the park index; GPU-radix mode -> still match in radix.
        survived = 0
        checked = 0
        for item, n0 in list(self._recent_parked):
            try:
                if self.park_gpu is not None:
                    present = item in self._park_index  # item is the hash
                else:
                    present = (
                        len(self.tree_cache.match_prefix(RadixKey(item, extra_key=None)).device_indices)
                        >= n0
                    )
            except Exception:  # noqa: BLE001
                continue
            checked += 1
            if present:
                survived += 1
        surv_rate = (survived / checked) if checked else -1.0
        logger.info(
            "Idle KV parking [prefill] DIAG: recv=%d processed=%d backlog=%d | "
            "skip=%d(%.0f%%) copy=%d avg-P-had=%.2f | survival=%.0f%% (%d/%d) | "
            "H1(retain):skip↑had↑  H2(evict):survival↓  H3(no-match):survival↑reuse-flat  "
            "BACKLOG:recv≫processed=copy-too-slow",
            self._received_msgs,
            total,
            self._incoming.qsize(),
            self._skipped_count,
            skip_rate * 100,
            self._copied_count,
            existing_frac,
            surv_rate * 100 if surv_rate >= 0 else -1,
            survived,
            checked,
        )

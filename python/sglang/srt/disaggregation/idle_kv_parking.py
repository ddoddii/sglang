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
# Candidate idle-GPU park pools. SGLANG_KV_PARK_GPUS="2,3" gives several idle GPUs the
# prefill node can park into; parking picks, per request, the pool with the most headroom
# (opportunistic placement onto whichever idle resource has room -- Phase 2 slice 1).
# SGLANG_KV_PARK_GPU (single) is kept for back-compat.
_PARK_GPUS_ENV = os.environ.get("SGLANG_KV_PARK_GPUS")
_PARK_GPU_ENV = os.environ.get("SGLANG_KV_PARK_GPU")


def _parse_park_gpus():
    if _PARK_GPUS_ENV not in (None, ""):
        return [int(x) for x in _PARK_GPUS_ENV.split(",") if x.strip() != ""]
    if _PARK_GPU_ENV not in (None, ""):
        return [int(_PARK_GPU_ENV)]
    return []


PARK_GPUS = _parse_park_gpus()
PARK_GPU = PARK_GPUS[0] if PARK_GPUS else None  # back-compat alias (primary pool)
PARK_POOL_TOKENS = int(os.environ.get("SGLANG_KV_PARK_POOL_TOKENS", "200000"))
# Decode publishes its KV-pool IPC handles here; prefill consumes them.
DECODE_IPC_FILE = os.path.join(PARK_DIR, "decode_kvpool_ipc.pkl")
# Prefill publishes its ZMQ park-control PULL address here; decode connects a PUSH.
PREFILL_ZMQ_FILE = os.path.join(PARK_DIR, "prefill_park_zmq.pkl")
RENDEZVOUS_TIMEOUT_S = 180


def _gpu_usage_file(gpu: int) -> str:
    """Per-GPU live serving KV-usage telemetry file (Phase 2 slice-2). Each node writes
    its own GPU's KV pool usage here; the parking node reads candidate GPUs' usage to
    place a park onto whichever GPU is momentarily idle (pressure-aware placement)."""
    return os.path.join(PARK_DIR, f"usage_gpu{gpu}.txt")

# slice 2b self-test: number of KV slots to round-trip D->P to validate the
# multi-layer indexed gather-copy over NVLink. Uses free slots [1..N] at startup
# (slot 0 is the padded dummy); the allocator overwrites them on real use.
SELFTEST_N_SLOTS = 64

# Rolling polynomial prefix hash (fetch-path optimization). Park stores an entry
# keyed by the hash of its token_ids at two page-aligned boundaries; fetch must probe
# the request's prefix at each parked length. Computing hash(tuple(token_ids[:L])) per
# length allocates an L-int tuple and rehashes L ints *every* length -- O(#lengths x L)
# on the scheduler main thread, which becomes the bottleneck at high concurrency
# (saturation: bottleneck moves from GPU-prefill-FLOPs to scheduler-CPU). A polynomial
# hash h_L = (((tok_0+1)*B + tok_1+1)*B + ...) mod 2^63 lets fetch compute the hash at
# every boundary in ONE O(n) pass and look each length up in O(1), with no per-length
# tuple allocation. Both park and fetch use _prefix_hash so the keys agree.
_PH_B = 1_000_003
_PH_MASK = (1 << 63) - 1


def _prefix_hash(token_ids, upto: int) -> int:
    """Polynomial hash of token_ids[:upto] (see _PH_B note). O(upto), no allocation."""
    h = 0
    for i in range(upto):
        h = (h * _PH_B + token_ids[i] + 1) & _PH_MASK
    return h


class _ParkPool:
    """One idle GPU's KV park buffer: a ring of N token slots + an LRU index.

    Phase 2 slice 1 holds several of these (one per candidate idle GPU) so the prefill
    node can place a park onto whichever pool currently has the most headroom -- the
    "store KV on whichever GPU is idle right now" mechanism. Copy in/out stays in the
    manager (it needs the peer/local KV buffers); this class owns the per-pool
    bookkeeping (index, ring pointer, occupancy)."""

    def __init__(self, gpu: int, k_buffer, N: int):
        from collections import Counter, OrderedDict

        self.gpu = gpu
        self.dev = f"cuda:{gpu}"
        head_num, head_dim = k_buffer[0].shape[1], k_buffer[0].shape[2]
        dtype = k_buffer[0].dtype
        L = len(k_buffer)
        self.N = N
        self.k = [torch.zeros(N, head_num, head_dim, dtype=dtype, device=self.dev) for _ in range(L)]
        self.v = [torch.zeros(N, head_num, head_dim, dtype=dtype, device=self.dev) for _ in range(L)]
        self.index = OrderedDict()  # prefix-hash -> (start, n)
        self.lens = Counter()       # distinct parked lengths present (for fetch probe)
        self.next = 0               # ring write pointer
        self.written = 0            # cumulative tokens ever written (monotonic)
        self.gb = 2 * N * head_num * head_dim * k_buffer[0].element_size() * L / 1e9

    def occupancy(self) -> int:
        """Occupied slots. Once cumulative writes reach N the ring has cycled and every
        slot holds (LRU) data, so occupancy saturates at N. Monotonic proxy, never
        drifts (unlike a live counter with dual-index + wrap)."""
        return min(self.written, self.N)

    def headroom(self) -> int:
        """Free token slots. Selection prefers the pool with the most room -- the most
        idle GPU. A cycled pool reports 0 (about to LRU-evict on the next write)."""
        return self.N - self.occupancy()


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
        # Rendezvous freshness: only accept a peer IPC file published AFTER this
        # manager started. A stale decode_kvpool_ipc.pkl left in /dev/shm by a prior
        # run points at a dead process's GPU memory -> opening its handles throws
        # CUDA "invalid resource handle" and disables parking. Prefill launches before
        # decode (start script), so a fresh decode publish always has ts >= this.
        self._setup_start_ts = time.time()

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
        # slice 4b: fetch-on-hit (prefill pulls a parked prefix back before prefill).
        self._fetch_hits = 0        # requests whose parked prefix was fetched into radix
        self._fetched_tokens = 0    # tokens copied park-GPU -> local + inserted
        self._fetch_miss = 0        # request had no parked prefix
        self._fetch_already = 0     # P already had the prefix (natural radix hit)
        self._fetch_nospace = 0     # KV pool full even after evict-to-room (gave up)
        self._fetch_evicted = 0     # had to LRU-evict cold entries to make room (like hicache)
        self._fetch_ms_sum = 0.0
        # async fetch: enqueue the GPU2->GPU0 copy on the default stream and DON'T
        # host-synchronize. SGLang does forward_stream.wait_stream(default_stream)
        # before every forward, so the copy is guaranteed complete before the model
        # reads the KV -- correct without blocking the scheduler for the copy (~235ms).
        # Set SGLANG_KV_PARK_SYNC_FETCH=1 to fall back to the blocking copy.
        self.sync_fetch = os.environ.get("SGLANG_KV_PARK_SYNC_FETCH", "0") == "1"
        # Also park the decode-generated KV (not just the prompt prefix) so a next turn
        # can reuse it when the assistant tokens recur. Set SGLANG_KV_PARK_GEN=0 to park
        # the prefix only (the earlier behavior) for a clean A/B of the generated gain.
        self.park_gen = os.environ.get("SGLANG_KV_PARK_GEN", "1") == "1"
        # Phase 2 slice-2: place a park onto the candidate GPU with the lowest LIVE
        # serving KV usage (the momentarily-idle GPU), read from per-GPU telemetry, not
        # just the park pool's own fill. Falls back to headroom when telemetry is absent
        # (e.g. dedicated spare GPUs). Set 0 to force slice-1 headroom-only selection.
        self.pressure_aware = os.environ.get("SGLANG_KV_PARK_PRESSURE_AWARE", "1") == "1"

        # slice 4 / Phase 2 slice 1: park pools on one or more idle GPUs. Parking picks,
        # per request, the pool with the most headroom (opportunistic idle-GPU placement).
        self.park_gpus = list(PARK_GPUS)
        self.park_gpu = PARK_GPU  # primary (first) pool; kept for logs/back-compat
        self._pools: List["_ParkPool"] = []
        if self.role == "prefill" and self.park_gpus and self.k_buffer is not None:
            self._init_park_gpu_pool()
        from collections import deque as _deque

        self._recent_parked = _deque(maxlen=32)  # recent parks for survival probe

        # Run setup off the hot path so server startup is not blocked.
        threading.Thread(
            target=self._setup, name="idle-kv-park-setup", daemon=True
        ).start()

        # Phase 2 slice-2: every node publishes its live serving KV usage so the parking
        # node can pick the momentarily-idle GPU. Runs for all roles (allocator present).
        if self.k_buffer is not None:
            threading.Thread(
                target=self._publish_usage_loop, name="idle-kv-park-usage", daemon=True
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
        # Wait for a FRESH decode IPC file (ts >= our start). Skip/stale files left by
        # a prior run would open dead GPU handles (CUDA invalid resource handle).
        payload = None
        while True:
            if os.path.exists(DECODE_IPC_FILE):
                try:
                    with open(DECODE_IPC_FILE, "rb") as f:
                        cand = pickle.load(f)
                except Exception:  # noqa: BLE001 (partial write / race) -> retry
                    cand = None
                if cand is not None and cand.get("ts", 0) >= self._setup_start_ts:
                    payload = cand
                    break
            if time.time() > deadline:
                logger.warning(
                    "Idle KV parking [prefill]: no FRESH decode IPC file at %s after "
                    "%ds (stale-only or peer down). Parking inactive.",
                    DECODE_IPC_FILE,
                    RENDEZVOUS_TIMEOUT_S,
                )
                return
            time.sleep(1.0)

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
            # Park the full finished sequence: prompt prefix (origin_input_ids) + the
            # decode-generated tokens (output_ids). The prefill side indexes it at TWO
            # matchable boundaries into one stored block:
            #   - prefix (origin): a token-exact prefix of the next turn (chat templates
            #     concatenate messages) -> always hits, recovers the prompt KV.
            #   - full (origin+generated): hits ONLY when the next turn's re-rendered
            #     assistant tokens equal the raw generation (plain text: yes; tool calls:
            #     often no, because the template re-serializes the tool call). When it
            #     hits, the generated KV is reused too, cutting the per-turn recompute of
            #     the assistant response. output_ids[:-1] = the committed KV range.
            prompt_ids = list(req.origin_input_ids)
            gen_ids = list(req.output_ids[:-1]) if self.park_gen else []
            full_ids = prompt_ids + gen_ids
            n_full = (len(full_ids) // self.page_size) * self.page_size
            n_prefix = (len(prompt_ids) // self.page_size) * self.page_size
            if n_full == 0 or token_indices.numel() < n_full:
                return False
            kv_indices = token_indices[:n_full].detach().to("cpu", torch.int64).tolist()
            self._push.send(
                pickle.dumps(
                    {
                        "type": "park",
                        "rid": req.rid,
                        "token_ids": full_ids[:n_full],
                        "prefix_len": n_prefix,
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

    # --- slice 4 / Phase 2 slice 1: idle-GPU park pools -----------------------
    def _init_park_gpu_pool(self) -> None:
        """Build one park pool per candidate idle GPU (SGLANG_KV_PARK_GPUS)."""
        N = PARK_POOL_TOKENS
        for gpu in self.park_gpus:
            self._pools.append(_ParkPool(gpu, self.k_buffer, N))
        total_gb = sum(p.gb for p in self._pools)
        logger.info(
            "Idle KV parking [prefill]: %d idle-GPU park pool(s) on GPU%s = %d tokens x "
            "%d layers each (~%.1f GB total). Parking picks the pool with most headroom.",
            len(self._pools), [p.gpu for p in self._pools], N, len(self.k_buffer), total_gb,
        )

    def _live_kv_usage(self):
        """This node's live serving KV pool usage fraction [0,1], or None."""
        try:
            alloc = self.token_to_kv_pool_allocator
            total = getattr(alloc, "size", None)
            if not total:
                return None
            return max(0.0, min(1.0, 1.0 - alloc.available_size() / total))
        except Exception:  # noqa: BLE001
            return None

    def _publish_usage_loop(self) -> None:
        """Write this GPU's live serving KV usage to its telemetry file every 0.5s."""
        path = _gpu_usage_file(self.gpu_id)
        while True:
            u = self._live_kv_usage()
            if u is not None:
                try:
                    tmp = path + f".tmp.{os.getpid()}"
                    with open(tmp, "w") as fh:
                        fh.write(f"{u:.4f} {time.time():.1f}")
                    os.replace(tmp, path)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.5)

    def _read_gpu_usage(self, gpu: int, stale_s: float = 5.0):
        """Read GPU `gpu`'s last-published serving KV usage; None if missing/stale.
        A dedicated spare GPU (no serving process) has no file -> None -> treated as
        idle (0.0) by the selector, so slice-1 (dedicated spare) behavior is preserved."""
        try:
            with open(_gpu_usage_file(gpu)) as fh:
                parts = fh.read().split()
            u = float(parts[0])
            ts = float(parts[1]) if len(parts) > 1 else time.time()
            if time.time() - ts > stale_s:
                return None
            return u
        except Exception:  # noqa: BLE001
            return None

    def _select_pool(self) -> "_ParkPool":
        """Pick the target pool for a new park.

        Phase 2 slice-2 (pressure-aware): choose the pool on the GPU with the LOWEST live
        serving KV usage -- i.e. store onto whichever candidate GPU is momentarily idle
        right now -- tie-broken by park-pool headroom. A GPU with no telemetry (dedicated
        spare, no serving process) counts as idle (0.0), so this degrades to slice-1
        headroom-only selection. Set SGLANG_KV_PARK_PRESSURE_AWARE=0 to force slice-1."""
        if not self.pressure_aware:
            return max(self._pools, key=lambda p: p.headroom())

        def key(p: "_ParkPool"):
            u = self._read_gpu_usage(p.gpu)
            serving = 0.0 if u is None else u  # no telemetry => not serving => idle
            # low serving usage first; among equally-idle GPUs prefer more headroom.
            return (round(serving, 2), -p.headroom())

        return min(self._pools, key=key)

    def _find_parked(self, h: int):
        """Return (pool, entry) if hash h is parked in any pool, else (None, None)."""
        for p in self._pools:
            ent = p.index.get(h)
            if ent is not None:
                return p, ent
        return None, None

    def _gather_copy_peer_to_park(self, pool: "_ParkPool", src_indices, start: int, n: int) -> None:
        """Copy peer(D) KV slots -> the given idle-GPU pool, across all layers."""
        peer_dev = f"cuda:{self.peer_k_buffer[0].device.index}"
        s = torch.tensor(src_indices, dtype=torch.long, device=peer_dev)
        for layer in range(len(self.peer_k_buffer)):
            pool.k[layer][start : start + n] = self.peer_k_buffer[layer][s].to(pool.dev)
            pool.v[layer][start : start + n] = self.peer_v_buffer[layer][s].to(pool.dev)
        torch.cuda.synchronize(pool.gpu)

    def _park_to_gpu(self, token_ids, src_indices, n: int, prefix_len: int = 0) -> None:
        """Store the full sequence KV in an idle-GPU pool (ring buffer + LRU index),
        picking the pool with the most headroom. Index it at two boundaries into the
        SAME block: the prompt prefix (prefix_len, always matchable next turn) and the
        full length (matchable only when the generated tokens recur). See park()."""
        if n > PARK_POOL_TOKENS or not self._pools:
            return
        h = _prefix_hash(token_ids, n)
        found, _ = self._find_parked(h)
        if found is not None:
            found.index.move_to_end(h)
            self._skipped_count += 1  # already parked (in some pool)
            return
        pool = self._select_pool()  # opportunistic: most-idle GPU
        start = pool.next
        if start + n > pool.N:
            start = 0  # wrap
        end = start + n
        # Evict index entries whose slots the new write overlaps.
        for k in list(pool.index.keys()):
            s0, ln = pool.index[k]
            if not (s0 + ln <= start or s0 >= end):
                del pool.index[k]
                pool.lens[ln] -= 1
                if pool.lens[ln] <= 0:
                    del pool.lens[ln]
        t0 = time.perf_counter()
        self._gather_copy_peer_to_park(pool, src_indices, start, n)
        ms = (time.perf_counter() - t0) * 1000.0
        pool.index[h] = (start, n)
        pool.lens[n] += 1
        pool.written += n  # monotonic occupancy proxy (headroom-based selection)
        # Also index the prompt-prefix boundary into the SAME block, so a next turn whose
        # generated tokens diverge (tool calls) still hits the prefix and recovers the
        # prompt KV (no regression vs prefix-only parking).
        if 0 < prefix_len < n:
            hp = _prefix_hash(token_ids, prefix_len)
            if hp not in pool.index:
                pool.index[hp] = (start, prefix_len)
                pool.lens[prefix_len] += 1
        pool.next = end % pool.N
        self._copied_count += 1
        self._n_sum += n
        self._recent_parked.append((h, n))  # dense: reflects the next-turn window
        if self._parked_count <= 5 or self._copied_count % 50 == 0:
            self._parked_count += 1
            logger.info(
                "Idle KV parking [prefill] GPU%d-park: %d tok x %d layers in %.1fms. "
                "pool headroom=%d, index=%d. (pools: %s)",
                pool.gpu, n, len(self.peer_k_buffer), ms, pool.headroom(),
                len(pool.index),
                ", ".join(f"g{p.gpu}:hr={p.headroom()}" for p in self._pools),
            )

    # --- slice 4b: fetch-on-hit (pull a parked prefix back before prefill) -------
    def _match_park_prefix(self, token_ids):
        """Longest parked entry (across all idle-GPU pools) whose token_ids is a
        (page-aligned) prefix of token_ids, returned as (pool, start, n).

        Compute the rolling polynomial prefix hash in ONE O(max_L) pass over the union
        of parked lengths, recording it at each boundary, then look each length up in
        O(1) per pool. ent[1]==L guards the (astronomically rare) collision."""
        if not self._pools:
            return None
        union_lens = set()
        n_req = len(token_ids)
        for p in self._pools:
            union_lens.update(L for L in p.lens if L <= n_req)
        if not union_lens:
            return None
        max_L = max(union_lens)
        boundary = {}  # length L -> polynomial hash of token_ids[:L]
        h = 0
        for i in range(max_L):
            h = (h * _PH_B + token_ids[i] + 1) & _PH_MASK
            L = i + 1
            if L in union_lens:
                boundary[L] = h
        for L in sorted(union_lens, reverse=True):
            hL = boundary[L]
            for p in self._pools:
                ent = p.index.get(hL)
                if ent is not None and ent[1] == L:
                    p.index.move_to_end(hL)  # LRU touch on read
                    return p, ent[0], ent[1]
        return None

    def _gather_copy_park_to_local(self, pool, park_start, existing, n, dst_idx) -> None:
        """Copy park-pool slots [park_start+existing : park_start+n] (idle GPU) ->
        local KV-pool slots dst_idx (P GPU), across all layers. This is the fetch
        direction: park_gpu -> P (PCIe on this topology, NVLink if paired).

        dst_idx is the GPU LongTensor of allocated local slots (a view into the alloc
        result) -- used directly to index the KV buffers. Earlier this took a Python
        list and rebuilt a GPU tensor via torch.tensor(...), which forced a synchronous
        GPU->CPU->GPU round-trip on the scheduler main thread; passing the device tensor
        removes that host sync (the diagnosed saturation cost).

        Async by default: the copy is enqueued on the P-GPU default stream and NOT
        host-synchronized, so the scheduler is not blocked for the copy. Correctness
        holds because SGLang runs forward_stream.wait_stream(default_stream) before
        each forward, ordering this copy ahead of any model read of the KV. The park
        source (GPU2) was synchronized at park time, so it is stable to read."""
        local_dev = f"cuda:{self.gpu_id}"
        lo, hi = park_start + existing, park_start + n
        for layer in range(len(self.k_buffer)):
            self.k_buffer[layer][dst_idx] = pool.k[layer][lo:hi].to(local_dev)
            self.v_buffer[layer][dst_idx] = pool.v[layer][lo:hi].to(local_dev)
        if self.sync_fetch:
            torch.cuda.synchronize(self.gpu_id)

    def maybe_fetch(self, req: "Req") -> int:
        """P: before a request enters prefill, pull its parked prefix from an idle-GPU
        pool back into the local KV pool + radix, so the scheduler prefix-hits instead
        of recomputing. Returns #tokens fetched (0 if none). Runs on the scheduler main
        thread (allocator-safe). Park-GPU mode only (slice 4b)."""
        if self.role != "prefill" or not self._pools:
            return 0
        try:
            token_ids = list(getattr(req, "origin_input_ids", None) or [])
        except Exception:  # noqa: BLE001
            return 0
        if not token_ids:
            return 0
        hit = self._match_park_prefix(token_ids)
        if hit is None:
            self._fetch_miss += 1
            return 0
        pool, start, n = hit
        key = RadixKey(token_ids[:n], extra_key=None)
        existing = len(self.tree_cache.match_prefix(key).device_indices)
        if existing >= n:
            self._fetch_already += 1  # P still has it (natural hit); nothing to stage
            return 0
        dst = self.token_to_kv_pool_allocator.alloc(n)
        if dst is None:
            # evict-to-room (like hicache): the P GPU pool is full, but the cold
            # entries we evict are safe -- their KV is still in the park pool (or
            # cheaply recomputable). LRU-evict enough to stage this (hotter) prefix,
            # then retry. This is exactly what the scheduler does under pressure.
            try:
                self.tree_cache.evict(n)
            except Exception as e:  # noqa: BLE001
                logger.debug("Idle KV parking [prefill]: evict-to-room failed: %r", e)
            dst = self.token_to_kv_pool_allocator.alloc(n)
            if dst is None:
                self._fetch_nospace += 1  # still no room even after eviction
                return 0
            self._fetch_evicted += 1
            # evict(n) may have dropped part of the prefix `existing` measured above.
            # Re-measure so we copy exactly the tail P now lacks (the park pool holds
            # the full prefix, so any tail is safe to copy). Otherwise dst slots in
            # (new_existing, old_existing] would be inserted uninitialized.
            existing = len(self.tree_cache.match_prefix(key).device_indices)
            if existing >= n:
                self.token_to_kv_pool_allocator.free(dst)
                self._fetch_already += 1
                return 0
        dst64 = dst.to(torch.int64)  # index/insert dtype; no host sync (stays on GPU)
        inserted_into_tree = False
        try:
            t0 = time.perf_counter()
            # Copy only the tail P lacks; dst[:existing] is freed after insert matches it.
            # Pass the GPU slot view directly -- no GPU->CPU->GPU round-trip (see helper).
            self._gather_copy_park_to_local(pool, start, existing, n, dst64[existing:n])
            ms = (time.perf_counter() - t0) * 1000.0
            new_prefix_len = self.tree_cache.insert(key, dst64)
            inserted_into_tree = True  # tree now owns dst64[new_prefix_len:]
            if new_prefix_len > 0:
                self.token_to_kv_pool_allocator.free(dst64[:new_prefix_len])
        except Exception:
            if not inserted_into_tree:
                self.token_to_kv_pool_allocator.free(dst)  # rollback whole alloc
            raise
        fetched = n - new_prefix_len
        self._fetch_hits += 1
        self._fetched_tokens += fetched
        self._fetch_ms_sum += ms
        if self._fetch_hits <= 5 or self._fetch_hits % 50 == 0:
            logger.info(
                "Idle KV parking [prefill] GPU%d-fetch: rid=%s pulled %d tok "
                "(P had %d of %d) x %d layers in %.1fms. total fetch hits=%d, "
                "tokens=%d, avg=%.1fms.",
                pool.gpu, getattr(req, "rid", "?"), fetched, existing, n,
                len(self.k_buffer), ms, self._fetch_hits, self._fetched_tokens,
                self._fetch_ms_sum / max(1, self._fetch_hits),
            )
        return fetched

    def _receive_park(self, msg: dict) -> None:
        src_indices = msg["kv_indices"]
        token_ids = msg["token_ids"]
        n = len(src_indices)
        if n == 0 or n != len(token_ids):
            return

        # slice 4: park into an idle-GPU pool that survives P-GPU pressure.
        if self._pools:
            self._park_to_gpu(token_ids, src_indices, n, prefix_len=msg.get("prefix_len", n))
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
                if self._pools:
                    present = any(item in p.index for p in self._pools)  # item is the hash
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
        fetch_attempts = (
            self._fetch_hits + self._fetch_miss + self._fetch_already + self._fetch_nospace
        )
        # per-pool occupancy + live serving usage of that GPU (slice 1 fill / slice 2
        # pressure). "use=?" means no telemetry (dedicated spare) -> treated as idle.
        def _u(p):
            u = self._read_gpu_usage(p.gpu)
            return "?" if u is None else f"{u:.2f}"
        pool_occ = " ".join(
            f"g{p.gpu}:{p.occupancy()}/{p.N}({len(p.index)},use={_u(p)})" for p in self._pools
        )
        logger.info(
            "Idle KV parking [prefill] DIAG: recv=%d processed=%d backlog=%d | "
            "skip=%d(%.0f%%) copy=%d avg-P-had=%.2f | survival=%.0f%% (%d/%d) | "
            "FETCH: hits=%d(evict-to-room=%d) tok=%d avg=%.1fms | "
            "miss=%d already=%d nospace=%d (of %d) | pools[live/N(idx)]: %s",
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
            self._fetch_hits,
            self._fetch_evicted,
            self._fetched_tokens,
            self._fetch_ms_sum / max(1, self._fetch_hits),
            self._fetch_miss,
            self._fetch_already,
            self._fetch_nospace,
            fetch_attempts,
            pool_occ,
        )

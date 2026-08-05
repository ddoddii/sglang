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

import json
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
# Session-keyed parking (Phase 2 task 7): each conversation gets one fixed-size SLAB it
# overwrites/grows in place (found via prefix-supersession), so a small pool holds the
# live working set with NO fragmentation (the variable free-list fragmented). A park is
# skipped if it exceeds one slab. pool = (N // slab) slabs.
PARK_SESSION_KEYED = os.environ.get("SGLANG_KV_PARK_SESSION_KEYED", "0") == "1"
PARK_SLAB_TOKENS = int(os.environ.get("SGLANG_KV_PARK_SLAB_TOKENS", "6000"))
# Park pools come out of UNALLOCATED HBM, not out of the serving KV pool's free slots.
# A pool size that does not fit therefore has to shrink, not abort: the size is a tuning
# knob, and asking for more than the card has left used to raise OutOfMemoryError inside
# the scheduler constructor and take the whole node down at startup. Leave a margin below
# the reported free memory for the allocator's own bookkeeping and for the other prefill
# process racing for the same decode GPU.
# 4 GB, not 1. A park pool is allocated during Scheduler init, BEFORE CUDA graph capture,
# before any forward pass, and before any peer has IPC-mapped this GPU -- so mem_get_info
# at that moment cannot see three separate costs still to come:
#   1. CUDA graph capture and activation workspace
#   2. ~262 MiB of CUDA context per PEER process that maps this GPU's park pool. In a
#      2P2D cluster a prefill GPU carried three of them (786 MiB total, measured).
#   3. transient allocator headroom in the forward pass
# At a 1 GB reserve a prefill granted 34,485 tokens (correctly clamped down from 40,000)
# came up, served for ~25 s, and died inside F.linear failing to allocate 20 MiB, with
# 11.75 MiB free on a 47.40 GiB card. Runs that survived left ~2.6 GB free after parking.
# Scale this up for clusters with more peers: term 2 grows with the node count.
PARK_GPU_RESERVE_GB = float(os.environ.get("SGLANG_KV_PARK_GPU_RESERVE_GB", "4.0"))
PARK_POOL_MIN_TOKENS = int(os.environ.get("SGLANG_KV_PARK_POOL_MIN_TOKENS", "2048"))
# torch.OutOfMemoryError only exists from 2.5; older builds expose it under torch.cuda.
# Both subclass RuntimeError, which is the last-resort fallback.
_OOM_ERROR = (getattr(torch, "OutOfMemoryError", None)
              or getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
# N-node rendezvous (Phase 2 slice-2 piece 2): each node publishes its own file so a
# 2P2D cluster's 4 nodes don't clobber a single rendezvous file. Decode publishes its
# KV-pool IPC handles (so any prefill can read its KV to copy a park); prefill publishes
# its ZMQ park-control address + its park-pool IPC handles (so peer prefills can read for
# cross-P fetch). Peers are discovered by scanning PARK_DIR.
def _decode_ipc_file(gpu: int) -> str:
    return os.path.join(PARK_DIR, f"decode_{gpu}_ipc.pkl")


def _prefill_file(gpu: int) -> str:
    return os.path.join(PARK_DIR, f"prefill_{gpu}.pkl")


# All nodes of ONE run share this epoch (set by the start script). Peer files with a
# different (or missing, when we set one) epoch are from another run and ignored. The
# start script also wipes PARK_DIR, so normally only this run's files are present.
PARK_EPOCH = os.environ.get("SGLANG_KV_PARK_EPOCH", "")
RENDEZVOUS_TIMEOUT_S = 180


def _gpu_usage_file(gpu: int) -> str:
    """Per-GPU live serving KV-usage telemetry file (Phase 2 slice-2). Each node writes
    its own GPU's KV pool usage here; the parking node reads candidate GPUs' usage to
    place a park onto whichever GPU is momentarily idle (pressure-aware placement)."""
    return os.path.join(PARK_DIR, f"usage_gpu{gpu}.txt")


def _parked_bytes_file(gpu: int) -> str:
    """Per-location parked-bytes telemetry (Phase 3). Answers, over time, 'how much of
    the reusable KV is sitting in idle GPU HBM versus in CPU DRAM?' -- the evidence that
    GPU-first placement actually happens rather than being asserted. Sampled by
    benchmark/park_location_sampler.py; one file per publishing process."""
    return os.path.join(PARK_DIR, f"parked_gpu{gpu}.json")

# slice 2b self-test: number of KV slots to round-trip D->P to validate the
# multi-layer indexed gather-copy over NVLink. Uses free slots [1..N] at startup
# (slot 0 is the padded dummy); the allocator overwrites them on real use.
SELFTEST_N_SLOTS = 64

# --- Phase 3: CPU DRAM overflow tier (placement priority 3) --------------------------
# Reached only when no GPU has room, so that "GPU-first" has something to fall back to
# and can be compared against. Deliberately NOT a pre-allocated pool: HiCache reserves
# a host KV pool up front (measured 61 GB, held whether the cache is 5% or 95% full),
# and re-creating that here would defeat the whole claim. Each block's pinned host
# memory is allocated at park time and released on eviction, so committed host DRAM
# tracks cached KV.
HOST_OVERFLOW = os.environ.get("SGLANG_KV_PARK_HOST_OVERFLOW", "0") == "1"
# Allocation sizes are QUANTIZED to a multiple of this many tokens. Measured on
# server17: a cold pin costs ~480-640 ms/GiB, but a repeat request for the SAME size is
# ~0 ms because PyTorch caches host allocations. With arbitrary per-session sizes every
# park would be a cold pin (~360 ms for a 750 MiB session vs a 28.6 ms transfer), so
# bucketing is what makes this tier viable, not an optimization.
HOST_BUCKET_TOKENS = int(os.environ.get("SGLANG_KV_PARK_HOST_BUCKET_TOKENS", "2048"))
# Upper LIMIT on host bytes held by this tier -- a cap, not a reservation: nothing is
# allocated until a park actually overflows. Keeps the tier from eating the DRAM the
# agent stack needs. 0 = unlimited.
HOST_MAX_BYTES = int(float(os.environ.get("SGLANG_KV_PARK_HOST_MAX_GB", "8")) * 2**30)
# Releasing a block returns it to PyTorch's caching host allocator, NOT to the OS, so
# RSS would stay at the high-water mark and "committed == cached" would be false. We
# therefore flush the host cache when the parked set genuinely SHRINKS: live bytes fell
# to below SHRINK_FRAC of the peak since the last flush, and the drop is at least
# FLUSH_MIN. Hysteresis matters -- flushing on every eviction under steady churn would
# pay the ~480 ms/GiB re-pin cost over and over.
HOST_SHRINK_FRAC = float(os.environ.get("SGLANG_KV_PARK_HOST_SHRINK_FRAC", "0.6"))
HOST_FLUSH_MIN_BYTES = int(
    float(os.environ.get("SGLANG_KV_PARK_HOST_FLUSH_MIN_GB", "1")) * 2**30
)

# --- ablation: send every park to CPU DRAM instead of an idle GPU ---------------------
# THE ONLY REASON THIS EXISTS is to answer "is the gain the LINK, or the POLICY?".
#
# Comparing our GPU parking against hicache moves three things at once -- placement
# policy, transfer software, and storage medium -- so a win over hicache cannot say which
# one paid. FORCE_HOST holds the first two fixed and moves only the third: same index,
# same eviction, same reuse-value ranking, same fetch code, same GPU park pool still
# ALLOCATED (so HBM footprint is unchanged and the two arms are memory-matched) -- but
# every park lands in pinned host DRAM over PCIe instead of on a peer GPU.
#
#   park_gpu  vs  park_host    -> medium only              (link)
#   park_host vs  hicache      -> same medium, both DRAM   (transfer software)
#   park_gpu  vs  radix        -> total gain
#
# The middle row is the one that matters: both arms put the KV in CPU DRAM over the same
# PCIe bus, so any difference there CANNOT be bandwidth.
# Park asynchronously: enqueue the copy, record a CUDA event, and publish the index
# entries on a later scheduler pass once the event fires.
#
# The blocking version cost 26.9 s of scheduler-main-thread time on a 512-turn longctx run
# -- 89.5% of the whole park path and 63% of all park+fetch scheduler time -- while the
# transfer it was waiting on runs at 19.9 GB/s, i.e. 76% of the PCIe link. The copy was
# never the inefficiency; waiting for it on the thread that admits requests was.
#
# WHAT THE WAIT DID AND DID NOT PROTECT. It did not protect the SOURCE: decode frees its
# slots right after send() with no ack (see park()), so that race predates this change and
# is unaffected by it. What it did protect is the DESTINATION -- no fetch, here or in a
# peer prefill via the shared index, may read a slab still being written. That invariant
# is kept by deferring the index publish, not by blocking the thread.
#
# SGLANG_KV_PARK_ASYNC_PARK=0 restores the blocking behaviour for an A/B.
PARK_ASYNC = os.environ.get("SGLANG_KV_PARK_ASYNC_PARK", "1") == "1"
FORCE_HOST = os.environ.get("SGLANG_KV_PARK_FORCE_HOST", "0") == "1"
if FORCE_HOST:
    HOST_OVERFLOW = True   # the tier under test; refusing to enable it would measure 0

# Per-fetch trace: one CSV row per satisfied fetch (tier, tokens, bytes, ms), so the
# cost of a tier can be REGRESSED rather than assumed -- ms = intercept + bytes/BW.
# The intercept is per-call software overhead, the slope is the link. Those are exactly
# the two hypotheses, and only a trace can separate them.
#
# READ THIS BEFORE FITTING: with SGLANG_KV_PARK_SYNC_FETCH=0 (the default) the copy is
# enqueued and not awaited, so the recorded ms is ENQUEUE time, not transfer time, and a
# fit of it is meaningless. Traces record the sync flag per row and the analysis refuses
# to fit async rows. Run the cost-model arm with SGLANG_KV_PARK_SYNC_FETCH=1.
FETCH_TRACE_DIR = os.environ.get("SGLANG_KV_PARK_FETCH_TRACE", "")
# Column order for the per-fetch phase breakdown. copy_to and copy_scatter split the two
# halves of the per-layer copy: copy_to allocates a temporary on the local device for
# every one of the 64 layer-slices, so an allocator stall under KV-pool pressure lands
# there rather than in the kernel launch it is easily mistaken for.
_PHASES = ("find", "match", "alloc", "evict", "copy_to", "copy_scatter", "copy_sync",
           "insert")

# --- Phase 3: reuse value (what to give up first) ------------------------------------
# Plain LRU throws away whichever block was touched longest ago, which ignores the one
# thing that actually differs between parked prefixes: how expensive they are to get
# back. Re-prefilling 32k tokens costs 6681 ms but 1k costs 161 ms (measured,
# results/intro/fig_ttft_ctx_sweep.json), so under LRU a slightly-staler 32k session
# loses to a fresh 1k one and the system pays 40x more to rebuild it.
#
# This is also the semantics the CUDA Unified Memory argument turns on: UM cannot know
# how long a re-prefill a page costs. If our own policy were pure LRU, that criticism
# would apply to us too, so reuse value is load-bearing for the paper, not a tweak.
#
#   reuse_value = reprefill_ms(n) * 0.5 ** (age / halflife)
#
# reprefill_ms is a least-squares fit of a*n + b*n^2 to the measured sweep above
# (residuals: -17% at 1k, -7% at 2k, -4% at 4k, <1% at 8k/16k/32k -- accurate exactly
# where the stakes are highest). The quadratic term is attention's O(n^2); a purely
# linear model would under-value long prefixes. RE-FIT THESE PER MODEL AND GPU: they
# encode Llama-3.1-8B on an A6000, not a universal constant.
REUSE_AWARE = os.environ.get("SGLANG_KV_PARK_REUSE_AWARE", "1") == "1"
PREFILL_MS_PER_TOK = float(os.environ.get("SGLANG_KV_PARK_PREFILL_MS_PER_TOK", "0.131656"))
PREFILL_MS_PER_TOK2 = float(
    os.environ.get("SGLANG_KV_PARK_PREFILL_MS_PER_TOK2", "2.40638e-06")
)
# Staleness halflife. Tuned to agentic multi-turn: the next turn of a live session
# arrives one tool call later (seconds), so a block untouched for several halflives is
# most likely a finished conversation. Too short degenerates to LRU; too long keeps dead
# sessions resident ahead of live ones.
REUSE_HALFLIFE_S = float(os.environ.get("SGLANG_KV_PARK_REUSE_HALFLIFE_S", "30"))


def _reprefill_ms(n_tokens: int) -> float:
    """Estimated cost of rebuilding an n-token prefix from scratch (see REUSE_* notes)."""
    n = float(max(0, n_tokens))
    return PREFILL_MS_PER_TOK * n + PREFILL_MS_PER_TOK2 * n * n


def _reuse_value(n_tokens: int, last_touch: float, now: float = None) -> float:
    """Worth of keeping this parked prefix: what a miss would cost, discounted by how
    stale it is. Lower value = evict/demote first."""
    if not REUSE_AWARE:
        # LRU-equivalent: older wins the "evict me" contest regardless of size.
        return last_touch
    age = max(0.0, (now if now is not None else time.time()) - last_touch)
    return _reprefill_ms(n_tokens) * (0.5 ** (age / REUSE_HALFLIFE_S))

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


class _FreeList:
    """First-fit free-list allocator over [0, N) token slots (session-keyed park mode).

    A pure ring can't reclaim a superseded conversation's space (each turn appends a
    bigger block), so the ring churns and evicts live entries. The free-list lets a park
    FREE the old version's region and reuse it, so live data ~= sum of latest-per-session
    lengths -> a small pool can hold the working set (higher survival, less variance)."""

    def __init__(self, N: int):
        self.N = N
        self.gaps = [[0, N]]  # sorted, coalesced [start, len] free regions

    def alloc(self, n: int):
        for i, g in enumerate(self.gaps):
            if g[1] >= n:
                s = g[0]
                if g[1] == n:
                    self.gaps.pop(i)
                else:
                    g[0] += n
                    g[1] -= n
                return s
        return None  # no single gap big enough (fragmented/full)

    def free(self, s: int, n: int) -> None:
        self.gaps.append([s, n])
        self.gaps.sort()
        merged = []
        for g in self.gaps:
            if merged and merged[-1][0] + merged[-1][1] == g[0]:
                merged[-1][1] += g[1]
            else:
                merged.append(list(g))
        self.gaps = merged

    def free_tokens(self) -> int:
        return sum(l for _, l in self.gaps)


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
        self.next = 0               # ring write pointer (ring mode)
        self.written = 0            # cumulative tokens ever written (monotonic)
        self.gb = 2 * N * head_num * head_dim * k_buffer[0].element_size() * L / 1e9
        # session-keyed mode: fixed-size SLAB allocator (no fragmentation). Each session
        # owns one slab, overwritten/grown in place (found via prefix-supersession).
        self.session_keyed = PARK_SESSION_KEYED
        self.slab = PARK_SLAB_TOKENS
        self.n_slabs = max(1, N // self.slab) if self.session_keyed else 0
        self.free_slabs = list(range(self.n_slabs))  # free slab indices
        self.blocks = OrderedDict()  # slab_base -> (n, [hashes]); LRU order for eviction

    def slab_alloc(self):
        """Return a free slab's base offset, or None if all slabs are in use."""
        if not self.free_slabs:
            return None
        return self.free_slabs.pop(0) * self.slab

    def slab_release(self, base: int) -> None:
        self.free_slabs.append(base // self.slab)

    def occupancy(self) -> int:
        """Occupied slots. Session-keyed: used slabs x slab size. Ring: min(cumulative
        writes, N) -- saturates once the ring cycles."""
        if self.session_keyed:
            return (self.n_slabs - len(self.free_slabs)) * self.slab
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


def _host_empty_cache() -> bool:
    """Ask PyTorch to return cached-but-unused pinned host memory to the OS.

    `del`eting a pinned tensor only hands it back to PyTorch's caching host allocator,
    so RSS stays at the high-water mark. Only this flush actually releases it, which is
    what makes committed host DRAM track cached KV instead of the peak. Measured on
    torch 2.9.1: RSS +1024 MB after a free becomes +0 MB after the flush. The API is
    private and has moved between versions, hence the probing."""
    for fn in (
        getattr(torch._C, "_host_emptyCache", None),
        getattr(torch.cuda, "host_empty_cache", None),
        getattr(torch._C, "_cuda_hostEmptyCache", None),
    ):
        if fn is None:
            continue
        try:
            fn()
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


class _HostBlock:
    """One parked prefix living in pinned host memory.

    All layers share ONE allocation of shape (2L, bucket_tokens, head_num, head_dim):
    layer l's K is t[l] and its V is t[L + l]. One tensor rather than 2L separate ones
    because the caching host allocator keys on size -- 2L allocations per park would
    multiply the number of live size classes and turn warm hits into cold pins."""

    __slots__ = ("t", "n", "bucket", "hashes", "nbytes", "ts")

    def __init__(self, layers, bucket, head_num, head_dim, dtype):
        self.t = torch.empty(
            (2 * layers, bucket, head_num, head_dim), dtype=dtype, pin_memory=True
        )
        self.bucket = bucket
        self.n = 0
        self.hashes = []
        self.nbytes = self.t.numel() * self.t.element_size()
        self.ts = time.time()      # last touch; input to _reuse_value

    def value(self, now=None) -> float:
        return _reuse_value(self.n, self.ts, now)

    def k(self, layer: int):
        return self.t[layer]

    def v(self, layer: int, layers: int):
        return self.t[layers + layer]


class _HostParkStore:
    """CPU DRAM overflow tier (placement priority 3).

    Not a pool: blocks are allocated on demand at quantized sizes and freed on
    eviction, with an explicit host-cache flush once the live set shrinks (see
    HOST_* knobs). Process-local by necessity -- pinned host memory has no IPC handle,
    so a block is only readable in the process that allocated it; the shared index still
    records these parks as LOC_HOST with the owning pid so other nodes know the prefix
    exists elsewhere and can choose to recompute instead of duplicating it."""

    def __init__(self, layers, head_num, head_dim, dtype, max_bytes=HOST_MAX_BYTES):
        self.layers = layers
        self.head_num = head_num
        self.head_dim = head_dim
        self.dtype = dtype
        self.max_bytes = max_bytes
        from collections import Counter, OrderedDict

        self.blocks = OrderedDict()      # hash -> _HostBlock (LRU: first = coldest)
        self.lens = Counter()            # parked lengths present (fetch-probe support)
        self.live_bytes = 0
        self.peak_bytes = 0              # since the last flush (hysteresis input)
        self.n_parked = 0
        self.n_evicted = 0
        self.evicted_bytes = 0           # cumulative KV bytes this tier destroyed
        self.n_flushes = 0
        self.n_alloc_fail = 0
        itemsize = torch.empty((), dtype=dtype).element_size()
        self.bytes_per_token = 2 * layers * head_num * head_dim * itemsize

    # ------------------------------------------------------------------ accounting

    def bucket_for(self, n: int) -> int:
        b = HOST_BUCKET_TOKENS
        return max(b, ((n + b - 1) // b) * b)

    def would_exceed(self, n: int) -> bool:
        if self.max_bytes <= 0:
            return False
        return self.live_bytes + self.bucket_for(n) * self.bytes_per_token > self.max_bytes

    # ------------------------------------------------------------------- mutations

    def alloc(self, n: int):
        """Allocate a block for n tokens, evicting LRU blocks first if the cap is hit.
        Returns the block, or None if even that is not enough."""
        while self.would_exceed(n) and self.blocks:
            self.evict_victim()
        if self.would_exceed(n):
            return None
        try:
            blk = _HostBlock(self.layers, self.bucket_for(n), self.head_num,
                             self.head_dim, self.dtype)
        except Exception as e:  # noqa: BLE001  (host OOM / pin failure)
            self.n_alloc_fail += 1
            logger.warning("Idle KV parking [host]: pinned alloc of %d tokens failed (%r)",
                           self.bucket_for(n), e)
            return None
        blk.n = n
        self.live_bytes += blk.nbytes
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self.n_parked += 1
        return blk

    def put(self, h: int, blk: "_HostBlock", n: int) -> None:
        blk.hashes.append(h)
        self.blocks[h] = blk
        self.blocks.move_to_end(h)
        self.lens[n] += 1

    def get(self, h: int):
        blk = self.blocks.get(h)
        if blk is not None:
            self.blocks.move_to_end(h)     # LRU order (still used as the tiebreak)
            blk.ts = time.time()           # recency input to reuse value
        return blk

    def evict_victim(self) -> None:
        """Give up the LEAST valuable block: cheapest to rebuild, discounted by
        staleness (_reuse_value). Under plain LRU a slightly-staler 32k session would
        lose to a fresh 1k one, and rebuilding it costs 40x more."""
        if not self.blocks:
            return
        now = time.time()
        h, blk = min(self.blocks.items(), key=lambda kv: kv[1].value(now))
        self.evicted_bytes += blk.n * self.bytes_per_token
        self._drop(h, blk)
        self.n_evicted += 1

    def drop(self, h: int) -> bool:
        blk = self.blocks.get(h)
        if blk is None:
            return False
        self._drop(h, blk)
        return True

    def _drop(self, h: int, blk: "_HostBlock") -> None:
        # A block is indexed under several hashes (full length and prompt prefix), so
        # every alias must leave the index or a later lookup would hand back freed memory.
        for hh in list(blk.hashes):
            if self.blocks.get(hh) is blk:
                del self.blocks[hh]
        self.blocks.pop(h, None)
        self.lens[blk.n] -= 1
        if self.lens[blk.n] <= 0:
            del self.lens[blk.n]
        self.live_bytes -= blk.nbytes
        del blk
        self.maybe_flush()

    def maybe_flush(self) -> bool:
        """Return the freed pages to the OS once the live set has genuinely shrunk.
        Without this, RSS sticks at the peak and "committed host DRAM == cached KV"
        would be false; with it on every eviction, steady churn would re-pay ~480 ms/GiB
        to re-pin. Hence the two-sided condition."""
        drop = self.peak_bytes - self.live_bytes
        if (self.live_bytes < self.peak_bytes * HOST_SHRINK_FRAC
                and drop >= HOST_FLUSH_MIN_BYTES):
            if _host_empty_cache():
                self.n_flushes += 1
                logger.info(
                    "Idle KV parking [host]: flushed host cache (live %.2f GB, peak "
                    "%.2f GB, released ~%.2f GB)",
                    self.live_bytes / 1e9, self.peak_bytes / 1e9, drop / 1e9)
            self.peak_bytes = self.live_bytes
            return True
        return False

    def stats(self) -> str:
        return (f"host: blocks={len(self.blocks)} live={self.live_bytes/1e9:.2f}GB "
                f"peak={self.peak_bytes/1e9:.2f}GB parked={self.n_parked} "
                f"evicted={self.n_evicted}({self.evicted_bytes/1e9:.2f}GB) "
                f"flushes={self.n_flushes} "
                f"allocfail={self.n_alloc_fail}")


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
        # N-node rendezvous (piece 2). prefill side:
        #   peer_decode_pools[d_gpu] = (k_bufs, v_bufs)  -> read D's KV to copy a park
        #   peer_park_pools[p_gpu]   = (k_bufs, v_bufs)  -> read a peer P's park pool (cross-P fetch)
        # decode side:
        #   _push_by_gpu[p_gpu] = zmq PUSH socket        -> send a park msg to any prefill
        self.peer_decode_pools = {}
        self.peer_park_pools = {}
        self._push_by_gpu = {}
        self._rdv_lock = threading.Lock()  # guards the discovery dicts (recv/main threads)

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
        self._fetch_cross_hits = 0  # of those, fetched from a PEER prefill's park pool
        self._fetch_host_hits = 0   # of those, fetched from the CPU DRAM overflow tier
        self._fetched_tokens = 0    # tokens copied park-GPU -> local + inserted
        self._fetch_miss = 0        # request had no parked prefix
        self._fetch_already = 0     # P already had the prefix (natural radix hit)
        self._fetch_nospace = 0     # KV pool full even after evict-to-room (gave up)
        self._fetch_evicted = 0     # had to LRU-evict cold entries to make room (like hicache)
        self._fetch_ms_sum = 0.0
        # Per-tier fetch cost. The aggregate above cannot answer "is the GPU link why
        # this is faster" -- a run that fetches mostly from the local pool and a run that
        # fetches mostly over PCIe both collapse into one number. Split by where the
        # bytes came from, and carry the token count so ms/token is derivable per tier.
        self._fetch_ms_tier = {"local": 0.0, "peer": 0.0, "host": 0.0}
        self._fetch_tok_tier = {"local": 0, "peer": 0, "host": 0}
        self._fetch_n_tier = {"local": 0, "peer": 0, "host": 0}
        # Where the scheduler-thread time in a fetch actually goes. The copy was the only
        # timed phase, and it read ~80 ms on the async path -- far too high for a pure
        # enqueue -- so the alternatives (index scan, radix match, allocation, and
        # evict-to-room on a full pool) have to be separable from it and from each other.
        self._fetch_phase_ms = {"find": 0.0, "match": 0.0, "alloc": 0.0, "evict": 0.0,
                                "copy_to": 0.0, "copy_scatter": 0.0, "copy_sync": 0.0,
                                "insert": 0.0}
        self._miss_find_ms_sum = 0.0   # index scans that found nothing: pure overhead
        # Per SOURCE GPU, not just per tier. "peer" lumps together an NVLink-bridged
        # neighbour and a PCIe-only one, and on this box the park candidates span both
        # (NVLink pairs are (0,1) and (2,3)). Achieved bandwidth per source is the only
        # way to tell a slow link from a slow copy implementation.
        # Park-side cost. Unmeasured until the fetch breakdown showed the fetch could
        # only account for 1% of the TTFT gap while parking moved 5.7x the bytes.
        self._park_phase_ms = {"index": 0.0, "gather": 0.0, "xfer": 0.0, "write": 0.0,
                               "sync": 0.0, "select": 0.0}
        self._park_bytes = 0
        self._park_n = 0
        # Parks whose copy is enqueued but not yet known to have landed. Drained on each
        # scheduler pass; their slabs are reserved but deliberately not yet findable.
        self._idx_free = []            # pinned index-staging buffers, see _upload_indices
        self._pending_parks = []
        self._pending_starts = {}      # id(pool) -> {start, ...}, so eviction skips them
        self._park_pending_peak = 0
        self._pending_idx_buf = None
        self._park_publish_lag_ms = 0.0
        self._fetch_src_ms = {}
        self._fetch_src_tok = {}
        self._fetch_src_n = {}
        self._last_copy_to_ms = 0.0
        self._last_copy_scatter_ms = 0.0
        self._last_copy_sync_ms = 0.0
        self._trace_fh = None       # opened lazily; see _trace_fetch
        # Residency accounting for the placement figure: tokens this process DESTROYED,
        # i.e. KV that no longer exists in any tier and would have to be re-prefilled.
        # Counted separately from host-tier evictions (self._host.evicted_bytes) because
        # a GPU eviction and a host eviction are different policy decisions.
        self._dropped_tokens = 0
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

        # Phase 3 (bandwidth-aware placement): rank candidate park GPUs by the MEASURED
        # link bandwidth to this GPU, so a PCIe-only peer (3.3 GB/s measured) is not
        # preferred over CPU DRAM (26.3 GB/s). Matrix is measured once per node and
        # published to /dev/shm; see link_bandwidth.py. Set 0 to disable.
        self.bw_aware = os.environ.get("SGLANG_KV_PARK_BW_AWARE", "1") == "1"
        self._linkbw = None      # lazily attached on first park (GPUs must be up)

        # Placement decision log (Exp 2 / M3). _select_pool returns only the winner, so
        # the occupancy of the candidates it REJECTED is discarded -- and "the KV went
        # where there was room" is a claim about the rejected ones. Without them a run
        # can show placement correlating with headroom and there is no way to separate
        # that from the workload having put the headroom where the parks were going
        # anyway. One JSONL line per park: chosen GPU plus every candidate's live usage,
        # headroom and link class at the instant of the decision, which also permits
        # replaying random / round-robin / always-local null models offline from the
        # same log rather than spending a GPU-hour per null model.
        #
        # Off unless a path is set: this is the hot path, one park per finished request.
        self._decision_log = os.environ.get("SGLANG_KV_PARK_DECISION_LOG", "")
        self._decision_fh = None
        self._decision_t0 = time.time()

        # Phase 3 placement priority 3: CPU DRAM overflow. Built lazily on the first
        # overflow so a run that never overflows allocates no host memory at all --
        # that is the point of the tier (see the HOST_* knobs).
        self.host_overflow = HOST_OVERFLOW
        self._host: Optional["_HostParkStore"] = None

        # slice 4 / Phase 2 slice 1: park pools on one or more idle GPUs. Parking picks,
        # per request, the pool with the most headroom (opportunistic idle-GPU placement).
        self.park_gpus = list(PARK_GPUS)
        self.park_gpu = PARK_GPU  # primary (first) pool; kept for logs/back-compat
        self._pools: List["_ParkPool"] = []
        # Shared cross-node park index (piece 1/3): mirror of park entries so a different
        # prefill can discover a park written by this node. Lives in /dev/shm, shared by
        # all prefills of this run.
        self._shared_index = None
        if self.role == "prefill" and self.park_gpus and self.k_buffer is not None:
            self._init_park_gpu_pool()
            try:
                from sglang.srt.disaggregation.shared_park_index import SharedParkIndex

                self._shared_index = SharedParkIndex(os.path.join(PARK_DIR, "park_index.bin"))
                logger.info("Idle KV parking [prefill gpu%s]: shared park index attached "
                            "(%s).", self.gpu_id, os.path.join(PARK_DIR, "park_index.bin"))
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [prefill gpu%s]: shared index init failed: %r; "
                             "cross-P discovery disabled.", self.gpu_id, e)
                self._shared_index = None
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
        # piece 4: each prefill publishes its distinct parked lengths so peer prefills
        # know which lengths to probe in the shared index for cross-P fetch.
        if self.role == "prefill" and self._pools:
            threading.Thread(
                target=self._publish_lengths_loop, name="idle-kv-park-lengths", daemon=True
            ).start()

    def _setup(self) -> None:
        try:
            if self.role == "decode":
                self._decode_publish_ipc()   # 2a: publish KV-pool IPC handles
                self._decode_connect_zmq()   # 3a: connect to prefill's park channel
            else:
                self._prefill_setup_zmq()    # 3a: bind park channel + publish handles
                self._prefill_open_peers()   # piece 2: open all D pools + peer P park pools
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
        payload["epoch"] = PARK_EPOCH
        # Keep verify alive for the peer's lifetime.
        self._verify_keepalive = verify

        path = _decode_ipc_file(self.gpu_id)
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, path)  # atomic publish
        logger.info(
            "Idle KV parking [decode gpu%s]: published %d k + %d v IPC handles + verify "
            "(checksum=%.1f) to %s (epoch=%s)",
            self.gpu_id, len(self.k_buffer), len(self.v_buffer),
            verify_checksum, path, PARK_EPOCH or "-",
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

    def _load_fresh(self, path: str):
        """Load a rendezvous pickle; return its payload iff it belongs to this run
        (matching epoch when one is set), else None."""
        try:
            with open(path, "rb") as f:
                cand = pickle.load(f)
        except Exception:  # noqa: BLE001 (missing / partial write)
            return None
        if PARK_EPOCH and cand.get("epoch", "") != PARK_EPOCH:
            return None
        if not PARK_EPOCH and cand.get("ts", 0) < self._setup_start_ts - 300:
            return None  # heuristic staleness guard when no epoch is configured
        return cand

    def _discover(self, kind: str):
        """Yield (gpu, payload) for every fresh peer file of `kind` in PARK_DIR.
        kind='decode' -> decode_<gpu>_ipc.pkl; kind='prefill' -> prefill_<gpu>.pkl."""
        import glob

        pat = "decode_*_ipc.pkl" if kind == "decode" else "prefill_*.pkl"
        for path in glob.glob(os.path.join(PARK_DIR, pat)):
            payload = self._load_fresh(path)
            if payload is not None and "gpu_id" in payload:
                yield payload["gpu_id"], payload

    def _prefill_open_peers(self) -> None:
        """P: discover and open (a) every decode node's KV pool (to copy a park from
        whichever D held the sequence) and (b) every PEER prefill's park pool (to fetch a
        park written by another P). Initial blocking wait, then background rediscovery."""
        torch.cuda.set_device(self.gpu_id)
        deadline = time.time() + RENDEZVOUS_TIMEOUT_S
        while not self.peer_decode_pools:
            self._rediscover_peers(verify_first=True)
            if self.peer_decode_pools:
                break
            if time.time() > deadline:
                logger.warning(
                    "Idle KV parking [prefill gpu%s]: no decode peers discovered after "
                    "%ds. Parking inactive.", self.gpu_id, RENDEZVOUS_TIMEOUT_S,
                )
                return
            time.sleep(1.0)
        self.peer_ready.set()
        threading.Thread(
            target=self._rediscover_loop, name="idle-kv-park-rdv", daemon=True
        ).start()

    def _rediscover_peers(self, verify_first: bool = False) -> None:
        """Open any not-yet-opened decode KV pools + peer prefill park pools."""
        for d_gpu, payload in self._discover("decode"):
            if d_gpu in self.peer_decode_pools:
                continue
            try:
                k = [_open_ipc(h) for h in payload["k_handles"]]
                v = [_open_ipc(h) for h in payload["v_handles"]]
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [prefill gpu%s]: open decode gpu%s failed: %r",
                             self.gpu_id, d_gpu, e)
                continue
            with self._rdv_lock:
                self.peer_decode_pools[d_gpu] = (k, v)
                if self.peer_k_buffer is None:  # back-compat: 1P1D _receive_park path
                    self.peer_k_buffer, self.peer_v_buffer = k, v
            can_p2p = torch.cuda.can_device_access_peer(self.gpu_id, d_gpu)
            logger.info("Idle KV parking [prefill gpu%s]: opened decode gpu%s KV pool "
                        "(%d layers, p2p=%s).", self.gpu_id, d_gpu, len(k), can_p2p)
            if verify_first and "verify" in payload:
                try:
                    vp = _open_ipc(payload["verify"])
                    loc = torch.empty_like(vp, device=f"cuda:{self.gpu_id}")
                    loc.copy_(vp); torch.cuda.synchronize(self.gpu_id)
                    ok = abs(float(loc.double().sum().item()) - payload["verify_checksum"]) < 1.0
                    logger.info("Idle KV parking [prefill gpu%s]: decode gpu%s IPC verify %s.",
                                self.gpu_id, d_gpu, "MATCH" if ok else "MISMATCH")
                except Exception as e:  # noqa: BLE001
                    logger.error("Idle KV parking [prefill gpu%s]: verify failed: %r", self.gpu_id, e)
        for p_gpu, payload in self._discover("prefill"):
            if p_gpu == self.gpu_id or p_gpu in self.peer_park_pools:
                continue
            for ph in payload.get("park_pools", []):
                try:
                    k = [_open_ipc(h) for h in ph["k_handles"]]
                    v = [_open_ipc(h) for h in ph["v_handles"]]
                except Exception as e:  # noqa: BLE001
                    logger.error("Idle KV parking [prefill gpu%s]: open peer park gpu%s failed: %r",
                                 self.gpu_id, ph.get("gpu"), e)
                    continue
                with self._rdv_lock:
                    self.peer_park_pools[ph["gpu"]] = (k, v)
                logger.info("Idle KV parking [prefill gpu%s]: opened peer P park pool on "
                            "gpu%s (%d layers) for cross-P fetch.", self.gpu_id, ph["gpu"], len(k))

    def _rediscover_loop(self) -> None:
        for _ in range(60):  # ~5 min of late-joiner polling, then stop
            time.sleep(5.0)
            try:
                self._rediscover_peers()
            except Exception as e:  # noqa: BLE001
                logger.debug("Idle KV parking [prefill gpu%s]: rediscover: %r", self.gpu_id, e)

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

    # --- slice 3a / piece 2: ZMQ park control channel + park-pool handle publish ----
    def _prefill_setup_zmq(self) -> None:
        """P: bind a PULL socket and publish {addr, gpu, park-pool IPC handles} so decode
        nodes can send park msgs here and peer prefills can read this P's park pool."""
        self._zmq_ctx = zmq.Context(1)
        port, self._pull = get_zmq_socket(self._zmq_ctx, zmq.PULL, endpoint=None)
        addr = f"tcp://127.0.0.1:{port}"
        # Export this prefill's park pools so peer prefills can open them for cross-P fetch.
        park_handles = []
        for p in self._pools:
            park_handles.append({
                "gpu": p.gpu, "N": p.N,
                "k_handles": [_export_ipc(t) for t in p.k],
                "v_handles": [_export_ipc(t) for t in p.v],
            })
        payload = {
            "role": "prefill", "gpu_id": self.gpu_id, "addr": addr,
            "park_pools": park_handles, "epoch": PARK_EPOCH, "ts": time.time(),
        }
        path = _prefill_file(self.gpu_id)
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp, path)
        threading.Thread(
            target=self._prefill_recv_loop, name="idle-kv-park-recv", daemon=True
        ).start()
        logger.info(
            "Idle KV parking [prefill gpu%s]: PULL bound at %s, published %d park-pool "
            "handle set(s) to %s (epoch=%s)",
            self.gpu_id, addr, len(park_handles), path, PARK_EPOCH or "-",
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
        """D: discover every prefill node, connect a PUSH to each, send a test ping.
        Keeps rediscovering so a late-starting prefill is picked up too."""
        self._zmq_ctx = zmq.Context(1)
        deadline = time.time() + RENDEZVOUS_TIMEOUT_S
        while not self._push_by_gpu:
            self._connect_prefills()
            if self._push_by_gpu:
                break
            if time.time() > deadline:
                logger.warning(
                    "Idle KV parking [decode gpu%s]: no prefill nodes discovered after "
                    "%ds; parking inactive.", self.gpu_id, RENDEZVOUS_TIMEOUT_S,
                )
                return
            time.sleep(1.0)
        threading.Thread(
            target=self._decode_rediscover_loop, name="idle-kv-park-rdv", daemon=True
        ).start()

    def _connect_prefills(self) -> None:
        for p_gpu, payload in self._discover("prefill"):
            if p_gpu in self._push_by_gpu:
                continue
            addr = payload.get("addr")
            if not addr:
                continue
            try:
                sock = get_zmq_socket(self._zmq_ctx, zmq.PUSH, endpoint=addr, bind=False)
                sock.send(pickle.dumps(
                    {"type": "ping", "from": "decode", "gpu_id": self.gpu_id, "ts": time.time()}))
            except Exception as e:  # noqa: BLE001
                logger.error("Idle KV parking [decode gpu%s]: connect prefill gpu%s (%s) "
                             "failed: %r", self.gpu_id, p_gpu, addr, e)
                continue
            with self._rdv_lock:
                self._push_by_gpu[p_gpu] = sock
            # back-compat: 1P1D park() uses self._push if set.
            if self._push is None:
                self._push = sock
            logger.info("Idle KV parking [decode gpu%s]: PUSH connected to prefill gpu%s "
                        "at %s.", self.gpu_id, p_gpu, addr)

    def _decode_rediscover_loop(self) -> None:
        for _ in range(60):
            time.sleep(5.0)
            try:
                self._connect_prefills()
            except Exception as e:  # noqa: BLE001
                logger.debug("Idle KV parking [decode gpu%s]: rediscover: %r", self.gpu_id, e)

    # --- decode side (slice 3b): park a finished request -----------------------
    def park(self, req: "Req") -> bool:
        """D: send the finished request's prefix (token ids + KV slot indices) to P.

        Called at request completion, before release_kv_cache frees the slots. The
        KV is still valid at send time; P copies it when it drains the message.
        NOTE (slice 3c): correctness under reuse needs an ack so D holds the slots
        until P has copied; for now the tool-call idle gap keeps them valid at low load.
        """
        if self.role != "decode" or not self._push_by_gpu:
            return False
        if getattr(req, "req_pool_idx", -1) == -1:
            return False
        try:
            target_gpu, sock = self._select_target_prefill()
            if sock is None:
                return False
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
            sock.send(
                pickle.dumps(
                    {
                        "type": "park",
                        "rid": req.rid,
                        "src_gpu": self.gpu_id,   # which decode pool the KV lives in
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

    def _select_target_prefill(self):
        """D: pick the prefill node to park onto -- the one with the lowest LIVE serving
        KV usage (the momentarily-idle P), from per-GPU telemetry. This is where the
        pressure-aware placement decision happens in NpNd. Falls back to any P when no
        telemetry. Returns (gpu, socket) or (None, None)."""
        items = list(self._push_by_gpu.items())
        if not items:
            return None, None
        if self.pressure_aware:
            def usage(g):
                u = self._read_gpu_usage(g)
                return 1.0 if u is None else u  # unknown P -> treat as busy (avoid)
            g = min((g for g, _ in items), key=usage)
            return g, self._push_by_gpu[g]
        g, s = items[0]
        return g, s

    # --- prefill side (slice 3b/3c): drain parked messages on the main thread --
    def _drain_pending_parks(self) -> None:
        """Publish parks whose copy has landed. Non-blocking: Event.query() asks, it does
        not wait, so a park that is still in flight simply stays pending one more pass.

        Order is preserved only within a pool, which is all that matters -- two parks to
        different pools are independent, and two to the same pool were enqueued on that
        device's stream in order, so an earlier one cannot still be running when a later
        one has finished."""
        if not self._pending_parks:
            return
        still = []
        for e in self._pending_parks:
            try:
                done = e["ev"].query()
            except Exception:  # noqa: BLE001
                done = True    # cannot query -> publish rather than leak the slab
            if not done:
                still.append(e)
                continue
            self._park_publish_lag_ms += (time.time() - e["t"]) * 1000.0
            self._release_idx_buf(e.get("idx_buf"))
            pend = self._pending_starts.get(id(e["pool"]))
            if pend is not None:
                pend.discard(e["start"])
            try:
                self._publish_park(e["pool"], e["h"], e["start"], e["n"],
                                   e["prefix_len"], e["token_ids"], e["ms"])
            except Exception as exc:  # noqa: BLE001
                logger.error("Idle KV parking [prefill]: publish park failed: %r", exc)
        self._pending_parks = still
        self._park_pending_peak = max(self._park_pending_peak, len(still))

    def poll_incoming(self, max_msgs: int = 4) -> None:
        """P: drain up to max_msgs parked prefixes, copy their KV from D over NVLink.

        Runs on the scheduler main thread (allocator is not thread-safe). Slice 3b
        copies + frees (validation); slice 3c will radix-insert instead of free.
        """
        # Gate on peer_ready so both peer_k_buffer and peer_v_buffer are fully mapped
        # and verified before we touch them.
        if self.role != "prefill" or not self.peer_ready.is_set():
            return
        # Before admitting new parks, retire finished ones -- otherwise a slab stays
        # invisible (and its pool short of headroom) for as long as parks keep arriving.
        self._drain_pending_parks()
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
    def _park_bytes_per_token(self) -> int:
        k0 = self.k_buffer[0]
        return 2 * k0.shape[1] * k0.shape[2] * k0.element_size() * len(self.k_buffer)

    def _fit_pool_tokens(self, gpu: int, want: int) -> int:
        """Largest pool that currently fits on `gpu`, capped at `want`.

        Sized from the live free memory rather than from a configured budget, because the
        two prefill processes allocate onto the same decode GPUs independently and neither
        knows what the other has taken. Whoever runs second simply gets the smaller pool
        instead of dying.
        """
        try:
            free, _total = torch.cuda.mem_get_info(gpu)
        except Exception:  # noqa: BLE001  -- device unreadable: let the alloc decide
            return want
        budget = free - PARK_GPU_RESERVE_GB * 1e9
        if budget <= 0:
            return 0
        return max(0, min(want, int(budget // self._park_bytes_per_token())))

    def _init_park_gpu_pool(self) -> None:
        """Build one park pool per candidate idle GPU (SGLANG_KV_PARK_GPUS).

        Each pool is clamped to what its GPU can actually hold and, if even that loses a
        race with the peer prefill, halved until it fits. A GPU with no room is skipped
        with a warning rather than aborting the node -- losing one park target degrades
        placement, while raising here would have taken down a 4-node cluster over a knob.
        """
        want = PARK_POOL_TOKENS
        for gpu in self.park_gpus:
            N = self._fit_pool_tokens(gpu, want)
            while N >= PARK_POOL_MIN_TOKENS:
                try:
                    self._pools.append(_ParkPool(gpu, self.k_buffer, N))
                    break
                except _OOM_ERROR:
                    torch.cuda.empty_cache()
                    N //= 2
            else:
                free_gb = 0.0
                try:
                    free_gb = torch.cuda.mem_get_info(gpu)[0] / 1e9
                except Exception:  # noqa: BLE001
                    pass
                logger.warning(
                    "Idle KV parking [prefill]: SKIPPING park pool on GPU%d -- asked for "
                    "%d tokens (%.1f GB) but only %.1f GB is free there. Park pools are "
                    "carved from UNALLOCATED HBM, not from free slots inside the serving "
                    "KV pool, so lower --mem-fraction-static on that GPU (or lower "
                    "SGLANG_KV_PARK_POOL_TOKENS) to make room.",
                    gpu, want, want * self._park_bytes_per_token() / 1e9, free_gb,
                )
        if not self._pools:
            raise RuntimeError(
                "Idle KV parking [prefill]: no park pool could be allocated on any of "
                f"GPU{self.park_gpus}. Every candidate is out of unallocated HBM; "
                "reduce --mem-fraction-static or SGLANG_KV_PARK_POOL_TOKENS."
            )
        # Is peer access actually ON for each park target? This decides what "GPU parking"
        # even means. Without P2P, a cross-device copy in PyTorch is STAGED THROUGH HOST --
        # the fetch becomes a GPU->host->GPU round trip while still being reported as a
        # peer-GPU hit, and the premise of the design is quietly false. The measured fetch
        # path runs at 10.2 GB/s against a 52.7 GB/s peer ceiling, which is exactly what a
        # staged copy would look like, so this must be visible at startup rather than
        # inferred from a bandwidth number after the fact.
        for p in self._pools:
            try:
                ok = (p.gpu == self.gpu_id
                      or torch.cuda.can_device_access_peer(self.gpu_id, p.gpu))
            except Exception:  # noqa: BLE001
                ok = None
            if ok is False:
                logger.warning(
                    "Idle KV parking [prefill]: NO P2P from GPU%d to park GPU%d. Fetches "
                    "from that pool are staged through host memory by PyTorch, so they "
                    "are a GPU->host->GPU round trip, not a peer copy.",
                    self.gpu_id, p.gpu)
            else:
                logger.info("Idle KV parking [prefill]: GPU%d -> park GPU%d peer access "
                            "%s", self.gpu_id, p.gpu,
                            "local" if p.gpu == self.gpu_id else
                            ("ENABLED" if ok else "UNKNOWN"))
        got = [p.N for p in self._pools]
        total_gb = sum(p.gb for p in self._pools)
        logger.info(
            "Idle KV parking [prefill]: %d idle-GPU park pool(s) on GPU%s = %s tokens x "
            "%d layers each (~%.1f GB total)%s. Parking picks the pool with most headroom.",
            len(self._pools), [p.gpu for p in self._pools], got, len(self.k_buffer),
            total_gb,
            "" if all(n == want for n in got) else f" [CLAMPED from {want} to fit]",
        )
        # What is left on this node's OWN GPU, stated at startup. Everything that goes
        # wrong here goes wrong later: graph capture, activations, and the ~262 MiB CUDA
        # context that EACH peer creates on this GPU when it IPC-maps the park pool -- a
        # cost that does not exist yet at this moment, so mem_get_info cannot see it. A
        # node that took too much came up clean, served for 25 s, then died inside
        # F.linear failing to allocate 20 MiB. Say it now instead.
        try:
            own_free = torch.cuda.mem_get_info(self.gpu_id)[0] / 1e9
            # Lower bound on the peer count: every process that IPC-maps this GPU adds a
            # context, and a node cannot see from here how many will. In the 2P2D run
            # three foreign contexts landed on a prefill GPU while this estimate said two.
            n_peers = max(0, len(self.park_gpus) - 1)
            peer_ctx = 0.28 * n_peers          # measured ~262 MiB per foreign context
            logger.info(
                "Idle KV parking [prefill]: GPU%d has %.2f GB unallocated after parking; "
                "still to come are CUDA graph capture, activations, and at least %.2f GB "
                "of peer IPC contexts (>=%d peers x ~0.28 GB).",
                self.gpu_id, own_free, peer_ctx, n_peers,
            )
            if own_free - peer_ctx < 1.5:
                logger.warning(
                    "Idle KV parking [prefill]: ONLY %.2f GB will remain on GPU%d after "
                    "peer IPC contexts. Runs that survived left ~2.6 GB. This node is "
                    "likely to die mid-run with a CUDA OOM in the forward pass. Lower "
                    "SGLANG_KV_PARK_POOL_TOKENS, or raise "
                    "SGLANG_KV_PARK_GPU_RESERVE_GB (currently %.1f).",
                    own_free - peer_ctx, self.gpu_id, PARK_GPU_RESERVE_GB,
                )
        except Exception:  # noqa: BLE001
            pass

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

    def _serving_used_tokens(self):
        """Tokens currently held by THIS GPU's live serving KV pool, or None.

        The absolute count, not the fraction _live_kv_usage returns: the residency
        breakdown needs local-GPU bytes in the same unit as the parked and host bytes,
        otherwise the four categories cannot be stacked."""
        try:
            alloc = self.token_to_kv_pool_allocator
            total = getattr(alloc, "size", None)
            if not total:
                return None
            return max(0, total - alloc.available_size())
        except Exception:  # noqa: BLE001
            return None

    def _bytes_per_token(self) -> int:
        """KV bytes one token occupies across all layers (K and V)."""
        k0 = self.k_buffer[0]
        return 2 * len(self.k_buffer) * k0.shape[1] * k0.shape[2] * k0.element_size()

    def _publish_parked_bytes(self) -> None:
        """Publish where this process's parked KV physically lives, in bytes.

        Written as its own file rather than folded into the usage file because the two
        have different consumers: usage drives placement (read by peers in the hot path),
        this is offline telemetry for the placement figure."""
        try:
            bpt = self._bytes_per_token()
            payload = {
                "ts": round(time.time(), 2),
                "writer_gpu": self.gpu_id,
                "pid": os.getpid(),
                # per-target-GPU so a stacked plot can separate local from peer parking
                "gpu_bytes": {str(p.gpu): int(p.occupancy()) * bpt for p in self._pools},
                "host_bytes": int(self._host.live_bytes) if self._host else 0,
                "host_peak_bytes": int(self._host.peak_bytes) if self._host else 0,
                "host_blocks": len(self._host.blocks) if self._host else 0,
                "host_flushes": self._host.n_flushes if self._host else 0,
                "host_evicted": self._host.n_evicted if self._host else 0,
                "bytes_per_token": bpt,
                # --- residency breakdown (local GPU / peer GPU / CPU DRAM / evicted).
                # "serving" is the live radix KV on THIS GPU, so the four categories
                # together account for every reusable KV byte the process ever held.
                # gpu_bytes above is keyed by target GPU, so peer vs local-park is
                # derivable from writer_gpu without a second source of truth.
                "serving_bytes": int(self._serving_used_tokens() or 0) * bpt,
                "dropped_bytes": int(self._dropped_tokens) * bpt,
                "host_evicted_bytes": int(self._host.evicted_bytes) if self._host else 0,
                # --- where fetches were satisfied from (the counterpart of the above:
                # residency only matters if the KV is actually read back)
                "fetch_hits": self._fetch_hits,
                "fetch_peer_hits": self._fetch_cross_hits,
                "fetch_host_hits": self._fetch_host_hits,
                "fetch_local_hits": max(
                    0, self._fetch_hits - self._fetch_cross_hits - self._fetch_host_hits
                ),
                "fetch_miss": self._fetch_miss,
                # Cumulative fetch latency. Without it a run can show the hit rate
                # rising while TTFT gets worse and there is no way to tell whether the
                # fetch cost more than the re-prefill it avoided -- which is exactly
                # what Llama-2-13B did (58.3% hit rate, TTFT 0.98x). The cost scales
                # with KV BYTES (800 KiB/token on an MHA model vs 128 on a GQA one)
                # while the saving scales with TOKENS, so the two can invert.
                "fetch_ms_sum": round(self._fetch_ms_sum, 1),
                "fetch_phase_ms": {k: round(v, 1)
                                   for k, v in self._fetch_phase_ms.items()},
                "miss_find_ms_sum": round(self._miss_find_ms_sum, 1),
                "park_phase_ms": {k: round(v, 1)
                                  for k, v in self._park_phase_ms.items()},
                "park_bytes_moved": self._park_bytes,
                "park_async": 1 if PARK_ASYNC else 0,
                "park_pending_peak": self._park_pending_peak,
                "park_publish_lag_ms": round(self._park_publish_lag_ms, 1),
                "park_n": self._park_n,
                "fetch_src_ms": {k: round(v, 1) for k, v in self._fetch_src_ms.items()},
                "fetch_src_tok": dict(self._fetch_src_tok),
                "fetch_src_n": dict(self._fetch_src_n),
                # Per-tier cost, so "why is it faster" is answerable from a run rather
                # than from a microbenchmark: ms/token per tier is
                # fetch_ms_tier[t] / fetch_tok_tier[t].
                "fetch_ms_tier": {k: round(v, 1)
                                  for k, v in self._fetch_ms_tier.items()},
                "fetch_tok_tier": dict(self._fetch_tok_tier),
                "fetch_n_tier": dict(self._fetch_n_tier),
                "sync_fetch": 1 if self.sync_fetch else 0,
                "force_host": 1 if FORCE_HOST else 0,
                "fetch_already": self._fetch_already,
                "fetch_nospace": self._fetch_nospace,
                "fetched_tokens": self._fetched_tokens,
                "parked_tokens": self._parked_tokens,
            }
            path = _parked_bytes_file(self.gpu_id)
            tmp = path + f".tmp.{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(payload, fh)
            os.replace(tmp, path)      # atomic: a sampler never sees a partial file
        except Exception:  # noqa: BLE001
            pass

    def _publish_usage_loop(self) -> None:
        """Write this GPU's live serving KV usage to its telemetry file every 0.5s, plus
        the per-location parked-bytes telemetry (Phase 3)."""
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
            if self._pools:
                self._publish_parked_bytes()
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

    def _link_bw(self):
        """Lazily attach to the node's measured link-bandwidth matrix (Phase 3).
        Returns None if unavailable, in which case selection degrades to the
        pressure+headroom behaviour."""
        if self._linkbw is not None or not self.bw_aware:
            return self._linkbw
        try:
            from sglang.srt.disaggregation.link_bandwidth import LinkBandwidth

            devs = sorted({p.gpu for p in self._pools} | {self.gpu_id})
            self._linkbw = LinkBandwidth.get(devs)
            logger.info("Idle KV parking: link bandwidth -> %s", self._linkbw.describe())
        except Exception as e:  # noqa: BLE001
            logger.warning("Idle KV parking: link-bandwidth probe failed (%r); "
                           "placement falls back to pressure+headroom only", e)
            self.bw_aware = False
        return self._linkbw

    def _host_store(self):
        """Lazily build the CPU DRAM overflow tier. Returns None if disabled."""
        if not self.host_overflow:
            return None
        if self._host is None:
            k0 = self.k_buffer[0]
            self._host = _HostParkStore(
                layers=len(self.k_buffer), head_num=k0.shape[1], head_dim=k0.shape[2],
                dtype=k0.dtype)
            logger.info(
                "Idle KV parking: CPU DRAM overflow enabled -- bucket=%d tok "
                "(%.0f MB/bucket), cap=%.1f GB (a limit, not a reservation)",
                HOST_BUCKET_TOKENS,
                HOST_BUCKET_TOKENS * self._host.bytes_per_token / 1e6,
                HOST_MAX_BYTES / 2**30 if HOST_MAX_BYTES else float("inf"))
        return self._host

    def _park_to_host(self, token_ids, src_k, src_v, src_indices, n: int,
                      prefix_len: int, src_gpu: int) -> bool:
        """Placement priority 3: no GPU had room, so keep the KV in CPU DRAM instead of
        dropping it (which would cost a full re-prefill next turn: measured 1203 ms at
        8k against a 40 ms host restore). Returns True if parked."""
        store = self._host_store()
        if store is None:
            return False
        blk = store.alloc(n)
        if blk is None:
            return False
        h = _prefix_hash(token_ids, n)
        L = store.layers
        t0 = time.perf_counter()
        # D2H gather, one contiguous pinned destination per layer. non_blocking is safe
        # because the destination is pinned; we synchronize before publishing the entry.
        for layer in range(L):
            blk.t[layer][:n].copy_(src_k[layer][src_indices], non_blocking=True)
            blk.t[L + layer][:n].copy_(src_v[layer][src_indices], non_blocking=True)
        torch.cuda.synchronize(src_gpu if src_gpu >= 0 else self.gpu_id)
        ms = (time.perf_counter() - t0) * 1000.0

        store.put(h, blk, n)
        if 0 < prefix_len < n:
            hp = _prefix_hash(token_ids, prefix_len)
            if hp not in store.blocks:
                # same block, second boundary: record the alias so eviction removes both
                blk.hashes.append(hp)
                store.blocks[hp] = blk
                store.lens[prefix_len] += 1
        # Mirror to the shared index as host-resident owned by THIS pid, so another node
        # learns the prefix exists but knows it cannot read it (no IPC for pinned host)
        # and can recompute rather than park a duplicate.
        if self._shared_index is not None:
            from sglang.srt.disaggregation.shared_park_index import LOC_HOST

            self._shared_index.insert(h, os.getpid(), 0, n, loc=LOC_HOST)
            if 0 < prefix_len < n:
                self._shared_index.insert(_prefix_hash(token_ids, prefix_len),
                                          os.getpid(), 0, prefix_len, loc=LOC_HOST)
        self._copied_count += 1
        self._n_sum += n
        self._parked_tokens += n     # host tier counts as parked too (see _park_to_gpu)
        self._recent_parked.append((h, n))
        if store.n_parked <= 5 or store.n_parked % 50 == 0:
            logger.info("Idle KV parking [host]: %d tok x %d layers in %.1fms | %s",
                        n, L, ms, store.stats())
        return True

    def _select_pool(self, need: int = 0) -> "_ParkPool":
        """Pick the target pool for a new park.

        Phase 2 slice-2 (pressure-aware): choose the pool on the GPU with the LOWEST live
        serving KV usage -- i.e. store onto whichever candidate GPU is momentarily idle
        right now -- tie-broken by park-pool headroom. A GPU with no telemetry (dedicated
        spare, no serving process) counts as idle (0.0), so this degrades to slice-1
        headroom-only selection. Set SGLANG_KV_PARK_PRESSURE_AWARE=0 to force slice-1.

        Phase 3 (bandwidth-aware): candidates whose link to THIS GPU is slower than CPU
        DRAM are demoted below the ones that are faster. Measured on 4x A6000: an
        NVLink-bridged peer reads at 27-53 GB/s but a PCIe-only peer at just 3.3 GB/s,
        against 26.3 GB/s for pinned host memory -- so "any GPU beats the host" is false,
        and picking the idlest GPU without checking the link can pick a target that is
        7-8x worse than CPU DRAM. Set SGLANG_KV_PARK_BW_AWARE=0 to disable.

        Ordering is (full, slow_link, serving_usage, -headroom): a pool that can take
        this park without evicting always wins, then a fast link beats a slow one, then
        the idlest and roomiest.

        `full` sits ABOVE `slow_link` on purpose. With the two swapped, a FULL
        NVLink pool outranked an EMPTY PCIe one, so the policy evicted a live prefix
        rather than write across a slower link -- measured on Exp 2's re-run, where P0
        found (own: full, NVLink peer: full, PCIe peer: has room) on 255 of 271 parks and
        chose the full NVLink peer every time. That trade is backwards by this module's
        own numbers: a PCIe fetch restores an 8k prefix in ~297 ms against ~1203 ms to
        re-prefill it, so keeping the data on a slow link beats losing it. The link only
        decides between pools that can both actually hold the park. Slow-link
        pools are kept as candidates rather than dropped because they still beat a
        recompute (3.3 GB/s restores an 8k prefix in ~297 ms vs ~1203 ms to re-prefill);
        the host tier, once implemented, takes priority over them.

        `full` was NOT in this key originally, and its absence cost real capacity. Usage
        was compared before headroom, so a peer pool with ZERO free slots (usage 0.06)
        kept outranking a local pool that was completely empty (usage 0.86): the peer
        won on the second key before headroom was ever consulted, and every park then
        evicted something from the same small pool. Measured on Exp 2's park_pd arm --
        both decode pools pinned full at 1.31 GB, the local pool at 0.00, 3.67 GB used of
        a 7.86 GB budget, and 491 fetch hits against park_local's 594 while half the
        park capacity sat idle. "Place where there is room" has to actually check whether
        there is room.

        `need` is the size of THIS park, so the test is "can take it without evicting",
        not "is not literally full". A pool with 100 free slots cannot absorb a 5000-token
        park and would evict just the same. need=0 reproduces the plain non-empty test."""
        lb = self._link_bw()

        def slow_link(p: "_ParkPool") -> int:
            if lb is None:
                return 0
            return 0 if lb.better_than_host(src=p.gpu, dst=self.gpu_id) else 1

        # Read each candidate's usage ONCE and reuse it for both the decision and the
        # log. Reading again for the log would record a different instant than the one
        # the decision was made on, which is the single thing the log exists to capture.
        usage = {p.gpu: self._read_gpu_usage(p.gpu) for p in self._pools}

        def full(p: "_ParkPool") -> int:
            return 0 if p.headroom() >= need else 1

        if not self.pressure_aware:
            chosen = min(self._pools, key=lambda p: (slow_link(p), -p.headroom()))
        else:
            def key(p: "_ParkPool"):
                u = usage.get(p.gpu)
                serving = 0.0 if u is None else u  # no telemetry => not serving => idle
                # fast link, then room for this park, then idlest, then roomiest. When
                # EVERY pool is full the first two terms tie and the order degrades to
                # the previous behaviour, which is what should happen: something has to
                # be evicted and the idlest GPU is still the best place to do it.
                return (full(p), slow_link(p), round(serving, 2), -p.headroom())

            chosen = min(self._pools, key=key)

        self._log_decision(chosen, usage, slow_link, full, need)
        return chosen

    def _log_decision(self, chosen, usage, slow_link, full=None, need=0) -> None:
        """Append one JSONL record describing this placement decision (Exp 2 / M3).

        Records the candidates NOT chosen, with their live usage, because that is what
        makes "placement follows headroom" falsifiable rather than a restatement of the
        selector's source code."""
        if not self._decision_log:
            return
        try:
            if self._decision_fh is None:
                path = f"{self._decision_log}.gpu{self.gpu_id}.jsonl"
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                self._decision_fh = open(path, "a", buffering=1)  # survives a SIGKILL
            self._decision_fh.write(json.dumps({
                "t": round(time.time() - self._decision_t0, 3),
                "src_gpu": self.gpu_id,
                "chosen": chosen.gpu,
                "pressure_aware": bool(self.pressure_aware),
                # usage is None when a candidate publishes no telemetry; keep it null
                # rather than substituting the selector's 0.0, so the analysis can tell
                # "measured idle" from "assumed idle".
                "need": need,
                "cand": {str(p.gpu): {"use": usage.get(p.gpu),
                                      "head": p.headroom(),
                                      "slow": slow_link(p),
                                      # 1 = could not take this park without evicting
                                      "full": (full(p) if full else None)}
                         for p in self._pools},
            }) + "\n")
        except Exception:  # noqa: BLE001 -- telemetry must never fail a park
            self._decision_log = ""   # stop retrying on a broken path

    def _find_parked(self, h: int):
        """Return (pool, entry) if hash h is parked in any pool, else (None, None).
        Host-resident blocks count as parked too -- otherwise an overflowed prefix would
        be parked a second time on the next turn, once in CPU DRAM and once on a GPU."""
        for p in self._pools:
            ent = p.index.get(h)
            if ent is not None:
                return p, ent
        if self._host is not None and self._host.get(h) is not None:
            return self._host, None
        return None, None

    def _touch_slab(self, pool, base) -> None:
        """Refresh a slab's last-touch time (recency input to _reuse_value).

        Must be called on every READ hit, not just on write: pool.index's LRU order and
        pool.blocks' timestamp are separate, so without this a session fetched every turn
        would keep looking stale and get evicted despite being the hottest thing in the
        pool -- exactly backwards."""
        blk = pool.blocks.get(base)
        if blk is not None:
            pool.blocks[base] = (blk[0], blk[1], time.time())
            pool.blocks.move_to_end(base)

    def _clear_slab_index(self, pool, base) -> None:
        """Drop the index entries of the block currently in slab `base` (local + shared),
        without releasing the slab. Used before overwriting a session's slab in place."""
        blk = pool.blocks.pop(base, None)
        if blk is None:
            return
        _n_b, hashes, _ts_b = blk
        for hh in hashes:
            ent = pool.index.pop(hh, None)
            if ent is not None:
                ln = ent[1]
                pool.lens[ln] -= 1
                if pool.lens[ln] <= 0:
                    del pool.lens[ln]
                if self._shared_index is not None:
                    self._shared_index.remove(hh)

    def _evict_slab(self, pool) -> None:
        """Give up the LEAST valuable slab, not simply the oldest: cheapest to rebuild,
        discounted by staleness (_reuse_value). With REUSE_AWARE=0 the value function
        degenerates to the timestamp, which reproduces the previous LRU behaviour."""
        if not pool.blocks:
            return
        now = time.time()
        # Never give up a slab whose copy is still in flight: it is not in pool.blocks yet
        # under the session-keyed layout, but be explicit rather than relying on that, so
        # a later change to when blocks are recorded cannot turn into silent corruption.
        pend = self._pending_starts.get(id(pool)) or set()
        cand = [b for b in pool.blocks if b not in pend]
        if not cand:
            return
        base = min(cand, key=lambda b: _reuse_value(
            pool.blocks[b][0], pool.blocks[b][2], now))
        self._dropped_tokens += pool.blocks[base][0]
        self._clear_slab_index(pool, base)
        pool.slab_release(base)

    def _supersede_slab(self, pool, token_ids, n: int):
        """Return the slab base of this conversation's current version (a parked block
        whose tokens are a page-aligned prefix of the new sequence), to overwrite in
        place; None if this is a new conversation. Links turns by the prefix relation
        (no client session id needed) -- so a session reuses ONE slab across turns."""
        cand = sorted((L for L in pool.lens if L < n), reverse=True)
        if not cand:
            return None
        needed = set(cand)
        max_L = cand[0]
        boundary = {}
        h = 0
        for i in range(max_L):
            h = (h * _PH_B + token_ids[i] + 1) & _PH_MASK
            if (i + 1) in needed:
                boundary[i + 1] = h
        for L in cand:  # longest prefix first = this conversation's latest parked version
            ent = pool.index.get(boundary[L])
            if ent is not None and ent[1] == L:
                return ent[0]  # slab base to overwrite
        return None

    def _upload_indices(self, indices, device):
        """Upload slot indices to `device` without stalling on the copy stream.

        `torch.tensor(list, device=cuda)` builds a PAGEABLE host tensor and copies it with
        a blocking cudaMemcpy, which waits for everything already queued on that stream.
        That is 80 KB of indices waiting behind hundreds of megabytes of KV: once parking
        went asynchronous and the queue stopped draining between parks, this line went from
        0.2 s to 4.4 s over a run and became the largest cost in the park path.

        Pinned staging plus non_blocking makes the copy async. The buffer must then not be
        rewritten until the copy has been consumed, so it is borrowed from a pool and
        returned only when the park's event fires (or immediately, on the blocking path).

        Returns (device_tensor, buffer_to_return_later).
        """
        n = len(indices)
        cap = 1 << max(13, (n - 1).bit_length())    # power-of-two size classes, >= 8192
        buf = None
        for i, b in enumerate(self._idx_free):
            if b.numel() >= cap:
                buf = self._idx_free.pop(i)
                break
        if buf is None:
            try:
                buf = torch.empty(cap, dtype=torch.int64, pin_memory=True)
            except Exception:  # noqa: BLE001  (pinning can fail under host pressure)
                return torch.tensor(indices, dtype=torch.long, device=device), None
        buf[:n] = torch.as_tensor(indices, dtype=torch.int64)
        return buf[:n].to(device, non_blocking=True), buf

    def _release_idx_buf(self, buf) -> None:
        if buf is None:
            return
        if len(self._idx_free) < 16:      # a cap, so a burst cannot pin unbounded host RAM
            self._idx_free.append(buf)

    def _gather_copy_peer_to_park(self, pool, src_k, src_v, src_indices, start: int, n: int) -> None:
        """Copy source decode-pool KV slots -> the given park pool, across all layers.
        src_k/src_v are the IPC-mapped buffers of the decode node that held the sequence."""
        # THIS PATH MOVES 5.7x WHAT THE FETCH PATH DOES and was entirely unaccounted.
        # Chasing the fetch's 80 ms was chasing 1% of the TTFT gap; parking wrote 535 GB
        # against 94 GB fetched, and at the copy efficiency actually measured that is ~52 s
        # against a 70 s wall-clock gap. It also runs on the scheduler MAIN THREAD (via
        # poll_incoming, because the allocator is not thread-safe), and unlike the fetch it
        # ends in a full device synchronize -- so every byte of it is blocking.
        #
        # Split the same way as the fetch so the cost is attributable rather than totalled:
        #   gather   src_k[layer][s] on the PEER device -- a gather kernel plus a temporary
        #            allocated on a GPU this process does not own the allocator for.
        #   xfer     the cross-device copy, i.e. the only part that is actually the link.
        #   write    the contiguous store into the park pool.
        #   sync     the blocking wait. If this dominates, the cost is not the copy at all
        #            but the decision to make parking synchronous.
        peer_dev = f"cuda:{src_k[0].device.index}"
        t = time.perf_counter()
        s, idx_buf = self._upload_indices(src_indices, peer_dev)
        t_idx = time.perf_counter() - t
        t_g = t_x = t_w = 0.0
        for layer in range(len(src_k)):
            a = time.perf_counter()
            gk = src_k[layer][s]
            gv = src_v[layer][s]
            b = time.perf_counter()
            ck = gk.to(pool.dev)
            cv = gv.to(pool.dev)
            c = time.perf_counter()
            pool.k[layer][start : start + n] = ck
            pool.v[layer][start : start + n] = cv
            d = time.perf_counter()
            t_g += b - a
            t_x += c - b
            t_w += d - c
        a = time.perf_counter()
        ev = None
        if PARK_ASYNC:
            # Record instead of wait. The event is checked on a later scheduler pass, so
            # the copy overlaps the forward pass rather than stalling request handling.
            # Recorded on the PARK device's current stream, after that device's writes.
            with torch.cuda.device(pool.gpu):
                ev = torch.cuda.Event()
                ev.record()
        else:
            torch.cuda.synchronize(pool.gpu)
            self._release_idx_buf(idx_buf)   # copy has landed; safe to rewrite
            idx_buf = None
        t_s = time.perf_counter() - a
        for k, v in (("index", t_idx), ("gather", t_g), ("xfer", t_x), ("write", t_w),
                     ("sync", t_s)):
            self._park_phase_ms[k] += v * 1000.0
        self._park_bytes += n * self._bytes_per_token()
        # On the async path the pinned buffer is still referenced by an in-flight copy;
        # it goes back to the pool when the event fires, not here.
        self._pending_idx_buf = idx_buf
        return ev

    def _park_to_gpu(self, token_ids, src_indices, n: int, prefix_len: int = 0,
                     src_gpu: int = -1) -> None:
        """Store the full sequence KV (copied from the src_gpu decode pool) in a local
        park pool (ring buffer + LRU index), and mirror the index entry to the shared
        cross-node index so a different prefill can discover it. Index at two boundaries:
        prompt prefix (always matchable next turn) + full length (generated-tokens recur)."""
        if n > PARK_POOL_TOKENS or not self._pools:
            return
        _host_fallback = None      # set below when no GPU pool can take this park
        # Source decode pool this KV lives in (N-node). Fall back to the single-peer
        # buffer for 1P1D back-compat.
        src = self.peer_decode_pools.get(src_gpu)
        if src is None:
            if self.peer_k_buffer is None:
                return
            src_k, src_v = self.peer_k_buffer, self.peer_v_buffer
        else:
            src_k, src_v = src
        h = _prefix_hash(token_ids, n)
        found, fent = self._find_parked(h)
        if found is not None:
            # fent is None for a host-resident hit, whose LRU was already touched inside
            # _HostParkStore.get(); only GPU pools need the explicit touch here.
            if fent is not None:
                found.index.move_to_end(h)
                self._touch_slab(found, fent[0])
            self._skipped_count += 1  # already parked (on some GPU or in CPU DRAM)
            return
        if FORCE_HOST:
            # Ablation arm: identical index, policy, ranking and fetch code -- only the
            # medium moves. Reached AFTER the already-parked check above so the dedup
            # behaviour is the same as the GPU arm too. The GPU pool stays allocated and
            # simply unused, which is deliberate: it keeps HBM footprint matched, so a
            # TTFT difference between the arms cannot be blamed on memory pressure.
            self._park_to_host(token_ids, src_k, src_v, src_indices, n, prefix_len,
                               src_gpu)
            return
        # `n` matters: the choice is "which pool can take THIS park without evicting",
        # not "which pool is non-empty".
        _t = time.perf_counter()
        pool = self._select_pool(need=n)  # idlest fast-link GPU that has room
        self._park_phase_ms["select"] += (time.perf_counter() - _t) * 1000.0
        if pool.session_keyed:
            # This conversation reuses its own slab (found by prefix-supersession),
            # overwritten in place; a new conversation takes a free slab (evict LRU if
            # none). Fixed-size slabs => no fragmentation.
            if n > pool.slab:
                # Too big for any GPU slab -> priority 3 (CPU DRAM) rather than dropping.
                if self._park_to_host(token_ids, src_k, src_v, src_indices, n,
                                      prefix_len, src_gpu):
                    return
                return
            start = self._supersede_slab(pool, token_ids, n)  # own slab, or None (new)
            if start is not None:
                self._clear_slab_index(pool, start)  # drop old version's index; keep slab
            else:
                start = pool.slab_alloc()
                while start is None and pool.blocks:
                    self._evict_slab(pool)
                    self._fetch_evicted += 1
                    start = pool.slab_alloc()
                if start is None:
                    # Every GPU slab is in use by a live session -> priority 3.
                    if self._park_to_host(token_ids, src_k, src_v, src_indices, n,
                                          prefix_len, src_gpu):
                        return
                    self._fetch_nospace += 1
                    return
        else:
            start = pool.next
            if start + n > pool.N:
                start = 0  # wrap
            end = start + n
            # Reserve the range NOW, not at publish time. Under async parking the publish
            # happens a scheduler pass or more later, so leaving pool.next unadvanced
            # would hand the identical offset to the next park and have two in-flight
            # copies write the same slots. The ring is the DEFAULT layout
            # (SGLANG_KV_PARK_SESSION_KEYED=0), so this is the common path, and the
            # session-keyed path is already safe because slab_alloc pops the slab here.
            pool.next = end % pool.N
            for k in list(pool.index.keys()):  # ring: evict index entries the write overlaps
                s0, ln = pool.index[k]
                if not (s0 + ln <= start or s0 >= end):
                    del pool.index[k]
                    pool.lens[ln] -= 1
                    if pool.lens[ln] <= 0:
                        del pool.lens[ln]
                    if self._shared_index is not None:
                        self._shared_index.remove(k)
        t0 = time.perf_counter()
        ev = self._gather_copy_peer_to_park(pool, src_k, src_v, src_indices, start, n)
        ms = (time.perf_counter() - t0) * 1000.0
        if ev is not None:
            # ASYNC PARK. The copy is enqueued and NOT waited for; the index entries that
            # would make this slab findable are held back until the event says the bytes
            # landed. Publishing them now would let a fetch -- in this process or, via the
            # shared index, in a peer prefill -- read a slab that is still being written.
            #
            # The slab itself is already reserved (slab_alloc / pool.next advanced above),
            # so nothing else can claim it; only its VISIBILITY is deferred. Eviction is
            # told to skip it via _pending_starts.
            self._pending_parks.append({
                "ev": ev, "pool": pool, "h": h, "start": start, "n": n,
                "idx_buf": self._pending_idx_buf,
                "prefix_len": prefix_len, "token_ids": token_ids, "ms": ms,
                "t": time.time(),
            })
            self._pending_starts.setdefault(id(pool), set()).add(start)
            return
        self._publish_park(pool, h, start, n, prefix_len, token_ids, ms)

    def _publish_park(self, pool, h, start, n, prefix_len, token_ids, ms) -> None:
        """Make a parked slab findable. Split out of _park_to_gpu so the blocking and the
        event-deferred paths cannot drift apart -- this is the step that must not run
        before the copy has landed."""
        hashes = [h]
        pool.index[h] = (start, n)
        pool.lens[n] += 1
        pool.written += n
        if self._shared_index is not None:
            self._shared_index.insert(h, pool.gpu, start, n)
        # Also index the prompt-prefix boundary into the SAME block (no regression vs
        # prefix-only parking when generated tokens diverge).
        if 0 < prefix_len < n:
            hp = _prefix_hash(token_ids, prefix_len)
            if hp not in pool.index:
                pool.index[hp] = (start, prefix_len)
                pool.lens[prefix_len] += 1
                hashes.append(hp)
                if self._shared_index is not None:
                    self._shared_index.insert(hp, pool.gpu, start, prefix_len)
        if pool.session_keyed:
            pool.blocks[start] = (n, hashes, time.time())
            pool.blocks.move_to_end(start)  # MRU
        # The ring's pool.next was advanced when the range was reserved in _park_to_gpu,
        # not here -- advancing it at publish would let two in-flight parks share slots.
        self._copied_count += 1
        self._n_sum += n
        # Count tokens parked onto a GPU pool too. This was only incremented on the
        # P-radix path, so park-pool runs published "parked_tokens: 0" beside
        # "fetched_tokens: 985319" -- a telemetry line that contradicts itself and makes
        # the whole record look untrustworthy.
        self._parked_tokens += n
        self._park_n += 1
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

    # --- slice 4b / piece 4: fetch-on-hit across local + peer prefill park pools ----
    def _peer_park_lengths(self):
        """Union of parked lengths published by PEER prefills (lengths_<gpu>.txt), so the
        cross-node probe knows which prefix lengths to test in the shared index."""
        import glob

        out = set()
        for path in glob.glob(os.path.join(PARK_DIR, "lengths_*.txt")):
            try:
                base = os.path.basename(path)
                gpu = int(base[len("lengths_"):-len(".txt")])
            except ValueError:
                continue
            if gpu == self.gpu_id:
                continue
            try:
                with open(path) as fh:
                    out.update(int(x) for x in fh.read().split() if x)
            except Exception:  # noqa: BLE001
                continue
        return out

    def _publish_lengths_loop(self) -> None:
        """Publish this prefill's distinct parked lengths every 0.5s for peer fetchers."""
        path = os.path.join(PARK_DIR, f"lengths_{self.gpu_id}.txt")
        while True:
            lens = set()
            for p in self._pools:
                lens.update(p.lens.keys())
            try:
                tmp = path + f".tmp.{os.getpid()}"
                with open(tmp, "w") as fh:
                    fh.write(" ".join(str(L) for L in sorted(lens)))
                os.replace(tmp, path)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)

    def _find_fetch_source(self, token_ids):
        """Longest parked prefix of token_ids across LOCAL pools and (via the shared
        index) PEER prefill park pools. Returns (src_k, src_v, start, n, gpu) or None.
        One rolling-hash pass over the union of candidate parked lengths (local + peer)."""
        if not self._pools:
            return None
        n_req = len(token_ids)
        union_lens = set()
        for p in self._pools:
            union_lens.update(L for L in p.lens if L <= n_req)
        if self._host is not None:
            union_lens.update(L for L in self._host.lens if L <= n_req)
        if self._shared_index is not None:
            union_lens.update(L for L in self._peer_park_lengths() if L <= n_req)
        if not union_lens:
            return None
        max_L = max(union_lens)
        boundary = {}
        h = 0
        for i in range(max_L):
            h = (h * _PH_B + token_ids[i] + 1) & _PH_MASK
            L = i + 1
            if L in union_lens:
                boundary[L] = h
        for L in sorted(union_lens, reverse=True):
            hL = boundary[L]
            for p in self._pools:  # local pools first (fast path)
                ent = p.index.get(hL)
                if ent is not None and ent[1] == L:
                    p.index.move_to_end(hL)  # LRU touch on read
                    self._touch_slab(p, ent[0])   # recency for _reuse_value
                    return p.k, p.v, ent[0], ent[1], p.gpu
            if self._host is not None:  # then CPU DRAM (process-local, priority 3)
                blk = self._host.get(hL)
                if blk is not None and blk.n >= L:
                    # gpu=-1 marks a host-resident hit; the fetch copy switches to H2D.
                    nl = self._host.layers
                    return ([blk.t[i] for i in range(nl)],
                            [blk.t[nl + i] for i in range(nl)], 0, L, -1)
            if self._shared_index is not None:  # then peer pools via shared index
                ent = self._shared_index.lookup(hL)
                # Host-resident entries are skipped here: pinned host memory has no IPC
                # handle, so only the owning process can read it (ent.readable_here()).
                # The host fetch path handles those separately.
                if ent is not None and ent.n == L and ent.is_gpu:
                    if ent.dev in self.peer_park_pools:
                        k, v = self.peer_park_pools[ent.dev]
                        return k, v, ent.start, ent.n, ent.dev
        return None

    def _gather_copy_park_to_local(self, src_k, src_v, park_start, existing, n, dst_idx) -> None:
        """Copy park-pool slots [park_start+existing : park_start+n] (a local OR peer
        prefill's park pool) -> local KV-pool slots dst_idx (this P's GPU), all layers.
        src_k/src_v are that pool's (possibly IPC-mapped) buffers. Async by default: the
        copy is enqueued on the default stream (SGLang's forward_stream.wait_stream orders
        it before the model read); the source was synchronized at park time."""
        local_dev = f"cuda:{self.gpu_id}"
        lo, hi = park_start + existing, park_start + n
        # Split the two halves of each layer's work. On the ASYNC path this loop measured
        # ~80 ms per fetch, which is impossible for pure enqueue and is the number that
        # decided the longctx result -- so it has to be attributable, not just totalled.
        #   to_ms       peer-GPU slice -> local tensor. Allocates a temporary on the local
        #               device every layer (64 per fetch). Under KV-pool pressure the
        #               caching allocator can block the host thread here to free blocks,
        #               which would be an allocator stall wearing a transfer's clothes.
        #   scatter_ms  indexed write into the KV pool (index_put), a real kernel launch.
        # If to_ms dominates, the cost is allocation, not the link -- and that is fixable.
        t_to = t_sc = 0.0
        for layer in range(len(self.k_buffer)):
            a = time.perf_counter()
            tk = src_k[layer][lo:hi].to(local_dev)
            tv = src_v[layer][lo:hi].to(local_dev)
            b = time.perf_counter()
            self.k_buffer[layer][dst_idx] = tk
            self.v_buffer[layer][dst_idx] = tv
            c = time.perf_counter()
            t_to += b - a
            t_sc += c - b
        self._last_copy_to_ms = t_to * 1000.0
        self._last_copy_scatter_ms = t_sc * 1000.0
        if self.sync_fetch:
            s = time.perf_counter()
            torch.cuda.synchronize(self.gpu_id)
            self._last_copy_sync_ms = (time.perf_counter() - s) * 1000.0
        else:
            self._last_copy_sync_ms = 0.0

    def _trace_fetch(self, tier: str, tokens: int, ms: float, src_gpu: int,
                     phases: dict | None = None) -> None:
        """Append one row to the per-fetch cost trace (no-op unless enabled).

        `sync` is recorded per row and not per file because SYNC_FETCH is read at
        construction: a trace that silently mixed enqueue times with transfer times
        would fit a bandwidth of several TB/s and look like a discovery."""
        if not FETCH_TRACE_DIR:
            return
        try:
            if self._trace_fh is None:
                os.makedirs(FETCH_TRACE_DIR, exist_ok=True)
                path = os.path.join(FETCH_TRACE_DIR, f"fetch_gpu{self.gpu_id}.csv")
                new = not os.path.exists(path) or os.path.getsize(path) == 0
                self._trace_fh = open(path, "a", buffering=1)
                if new:
                    self._trace_fh.write(
                        "ts,tier,src_gpu,tokens,bytes,ms,sync," + ",".join(_PHASES) + "\n")
            ph = phases or {}
            self._trace_fh.write(
                f"{time.time():.3f},{tier},{src_gpu},{tokens},"
                f"{tokens * self._bytes_per_token()},{ms:.3f},"
                f"{1 if self.sync_fetch else 0}," +
                ",".join(f"{ph.get(k, 0.0):.3f}" for k in _PHASES) + "\n")
        except Exception:  # noqa: BLE001
            pass

    def maybe_fetch(self, req: "Req") -> int:
        """P: before a request enters prefill, pull its parked prefix from an idle-GPU
        pool back into the local KV pool + radix, so the scheduler prefix-hits instead
        of recomputing. Returns #tokens fetched (0 if none). Runs on the scheduler main
        thread (allocator-safe). Park-GPU mode only (slice 4b)."""
        if self.role != "prefill" or not self._pools:
            return 0
        # Retire landed parks first. A prefix parked one pass ago is findable only after
        # its publish, so skipping this would report a miss for KV that is already resident
        # and would understate the hit rate by exactly the publish lag.
        self._drain_pending_parks()
        try:
            token_ids = list(getattr(req, "origin_input_ids", None) or [])
        except Exception:  # noqa: BLE001
            return 0
        if not token_ids:
            return 0
        _t_find = time.perf_counter()
        hit = self._find_fetch_source(token_ids)
        _ms_find = (time.perf_counter() - _t_find) * 1000.0
        if hit is None:
            self._fetch_miss += 1
            self._miss_find_ms_sum += _ms_find   # a miss still pays the index scan
            return 0
        src_k, src_v, start, n, src_gpu = hit
        is_cross = src_gpu in self.peer_park_pools  # parked by a PEER prefill
        is_host = src_gpu < 0                       # CPU DRAM overflow tier (priority 3)
        if is_host:
            self._fetch_host_hits += 1
        key = RadixKey(token_ids[:n], extra_key=None)
        _t = time.perf_counter()
        existing = len(self.tree_cache.match_prefix(key).device_indices)
        _ms_match = (time.perf_counter() - _t) * 1000.0
        if existing >= n:
            self._fetch_already += 1  # P still has it (natural hit); nothing to stage
            return 0
        _ms_evict = 0.0
        _t = time.perf_counter()
        dst = self.token_to_kv_pool_allocator.alloc(n)
        _ms_alloc = (time.perf_counter() - _t) * 1000.0
        if dst is None:
            # evict-to-room (like hicache): the P GPU pool is full, but the cold
            # entries we evict are safe -- their KV is still in the park pool (or
            # cheaply recomputable). LRU-evict enough to stage this (hotter) prefix,
            # then retry. This is exactly what the scheduler does under pressure.
            _t = time.perf_counter()
            try:
                self.tree_cache.evict(n)
            except Exception as e:  # noqa: BLE001
                logger.debug("Idle KV parking [prefill]: evict-to-room failed: %r", e)
            dst = self.token_to_kv_pool_allocator.alloc(n)
            _ms_evict = (time.perf_counter() - _t) * 1000.0
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
            self._gather_copy_park_to_local(src_k, src_v, start, existing, n, dst64[existing:n])
            ms = (time.perf_counter() - t0) * 1000.0
            _t = time.perf_counter()
            new_prefix_len = self.tree_cache.insert(key, dst64)
            _ms_insert = (time.perf_counter() - _t) * 1000.0
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
        if is_cross:
            self._fetch_cross_hits += 1
        # Charge the cost to the tier that served it. `fetched` (not n) is the number of
        # tokens actually copied, so ms/token stays comparable across tiers even when the
        # local radix already held part of the prefix.
        tier = "host" if is_host else ("peer" if is_cross else "local")
        self._fetch_ms_tier[tier] += ms
        self._fetch_tok_tier[tier] += fetched
        self._fetch_n_tier[tier] += 1
        # Phase breakdown of the whole scheduler-thread cost, not just the copy. `ms`
        # above covers only _gather_copy_park_to_local; find/match/alloc/evict/insert are
        # also serial on this thread and also block every other request in the batch.
        ph = {"find": _ms_find, "match": _ms_match, "alloc": _ms_alloc,
              "evict": _ms_evict, "copy_to": self._last_copy_to_ms,
              "copy_scatter": self._last_copy_scatter_ms,
              "copy_sync": self._last_copy_sync_ms, "insert": _ms_insert}
        for k, v in ph.items():
            self._fetch_phase_ms[k] += v
        g = str(src_gpu)
        self._fetch_src_ms[g] = self._fetch_src_ms.get(g, 0.0) + ms
        self._fetch_src_tok[g] = self._fetch_src_tok.get(g, 0) + fetched
        self._fetch_src_n[g] = self._fetch_src_n.get(g, 0) + 1
        self._trace_fetch(tier, fetched, ms, src_gpu, ph)
        if self._fetch_hits <= 5 or self._fetch_hits % 50 == 0:
            logger.info(
                "Idle KV parking [prefill gpu%s]-fetch: rid=%s pulled %d tok "
                "(P had %d of %d) from %s%s in %.1fms. hits=%d (cross-P=%d, host=%d), "
                "tokens=%d, avg=%.1fms.",
                self.gpu_id, getattr(req, "rid", "?"), fetched, existing, n,
                "CPU DRAM" if is_host else f"gpu{src_gpu}",
                " [cross-P]" if is_cross else "", ms,
                self._fetch_hits, self._fetch_cross_hits, self._fetch_host_hits,
                self._fetched_tokens, self._fetch_ms_sum / max(1, self._fetch_hits),
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
            self._park_to_gpu(token_ids, src_indices, n, prefix_len=msg.get("prefix_len", n),
                              src_gpu=msg.get("src_gpu", -1))
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
            "FETCH: hits=%d(cross-P=%d,host=%d,evict-to-room=%d) tok=%d avg=%.1fms | "
            "miss=%d already=%d nospace=%d (of %d) | pools[live/N(idx)]: %s%s",
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
            self._fetch_cross_hits,
            self._fetch_host_hits,
            self._fetch_evicted,
            self._fetched_tokens,
            self._fetch_ms_sum / max(1, self._fetch_hits),
            self._fetch_miss,
            self._fetch_already,
            self._fetch_nospace,
            fetch_attempts,
            pool_occ,
            f" | {self._host.stats()}" if self._host is not None else "",
        )

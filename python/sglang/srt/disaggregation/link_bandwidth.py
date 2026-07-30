from __future__ import annotations

"""Measured link bandwidth for KV-aware placement (Phase 3).

Placement must NOT assume "a GPU is always better than the host". On a 4x RTX A6000
node we measured, in one address space with one access path:

    local HBM ................. 331 GB/s
    NVLink-bridged peer HBM ....  27-53 GB/s
    CPU DRAM (pinned) ..........  26.3 GB/s   <-- H2D; D2H 26.4
    NON-bridged peer HBM .......   3.3 GB/s   <-- 7-8x SLOWER THAN THE HOST

So a peer GPU reachable only over PCIe (cuDeviceCanAccessPeer says 1 for every pair,
which says nothing about speed) is a far worse parking target than CPU DRAM. The
placement policy therefore ranks candidate locations by *measured* bandwidth, and a
peer only outranks the host if it is actually faster.

Topology differs per node, so nothing here is hardcoded. One process measures the
matrix once at startup and publishes it to /dev/shm; the other PD processes attach to
the same file instead of each re-measuring (4 processes x 12 pairs would otherwise add
seconds to startup and contend for the links while measuring).

Usage:
    from sglang.srt.disaggregation.link_bandwidth import LinkBandwidth
    lb = LinkBandwidth.get(devices=[0, 1, 2, 3])
    lb.peer_bw(src=2, dst=0)   # GB/s reading dev2's memory from dev0
    lb.host_bw()               # GB/s H2D from pinned host memory
    lb.better_than_host(src=2, dst=0)
"""

import fcntl
import json
import logging
import os
import time
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Where the measured matrix is published. Same directory as the park rendezvous files.
_DIR = os.environ.get("SGLANG_KV_PARK_DIR", "/dev/shm/sglang_kv_parking")
_PATH = os.path.join(_DIR, "link_bw.json")
# Buffer size for the probe copies. Large enough to reach steady-state bandwidth
# (see benchmark/vmm_probe.py: flat from 32 MiB up), small enough to be cheap.
_PROBE_BYTES = int(os.environ.get("SGLANG_KV_PARK_BW_PROBE_MB", "64")) * 2**20
_ITERS = int(os.environ.get("SGLANG_KV_PARK_BW_PROBE_ITERS", "3"))
# A cached matrix older than this is re-measured (topology can change across reboots
# and /dev/shm may outlive them on some systems).
_TTL_S = float(os.environ.get("SGLANG_KV_PARK_BW_TTL", "3600"))
# A peer must beat the host by this factor to be preferred. 1.0 = strictly faster.
_MARGIN = float(os.environ.get("SGLANG_KV_PARK_BW_MARGIN", "1.0"))


def _bw_gbps(dst: torch.Tensor, src: torch.Tensor, iters: int) -> float:
    """GB/s of dst.copy_(src), synchronizing outside the timing loop."""
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return (src.numel() * iters / dt / 1e9) if dt > 0 else 0.0


def _measure(devices: List[int]) -> Dict:
    """Measure peer-read bandwidth for every ordered pair plus pinned H2D/D2H.

    'Peer read' means: with dev `dst` current, copy FROM a buffer resident on `src`.
    That is the direction a fetch pays, which is what placement cares about."""
    n = _PROBE_BYTES
    peer: Dict[str, float] = {}
    host_h2d = host_d2h = 0.0
    prev = torch.cuda.current_device()
    try:
        for dst in devices:
            try:
                torch.cuda.set_device(dst)
                local = torch.empty(n, dtype=torch.uint8, device=f"cuda:{dst}")
            except Exception as e:  # noqa: BLE001
                logger.warning("link_bw: cannot allocate on dev %d (%r); skipping", dst, e)
                continue
            for src in devices:
                if src == dst:
                    continue
                try:
                    remote = torch.empty(n, dtype=torch.uint8, device=f"cuda:{src}")
                    peer[f"{src}->{dst}"] = round(_bw_gbps(local, remote, _ITERS), 2)
                    del remote
                except Exception as e:  # noqa: BLE001
                    # No P2P path at all -> unusable as a parking target for this dst.
                    logger.debug("link_bw: %d->%d unavailable (%r)", src, dst, e)
                    peer[f"{src}->{dst}"] = 0.0
            if not host_h2d:      # measure the host link once, from the first device
                try:
                    pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
                    host_h2d = round(_bw_gbps(local, pinned, _ITERS), 2)
                    host_d2h = round(_bw_gbps(pinned, local, _ITERS), 2)
                    del pinned
                except Exception as e:  # noqa: BLE001
                    logger.warning("link_bw: pinned host probe failed (%r)", e)
            del local
            torch.cuda.empty_cache()
    finally:
        try:
            torch.cuda.set_device(prev)
        except Exception:  # noqa: BLE001
            pass
    return {"ts": time.time(), "devices": list(devices), "peer": peer,
            "host_h2d": host_h2d, "host_d2h": host_d2h,
            "probe_bytes": n, "measured_by": os.getpid()}


class LinkBandwidth:
    """Measured-once, shared-by-all view of the node's link bandwidths."""

    _inst: Optional["LinkBandwidth"] = None

    def __init__(self, data: Dict):
        self.data = data
        self.peer = data.get("peer", {})
        self._host = float(data.get("host_h2d") or 0.0)

    # ---------------------------------------------------------------- construction

    @classmethod
    def get(cls, devices: List[int], force: bool = False) -> "LinkBandwidth":
        """Attach to the published matrix, measuring it if this is the first process.

        The flock makes exactly one process measure: whoever gets the exclusive lock
        first writes the file, and the others then read it. Measuring concurrently
        would both waste startup time and corrupt the numbers through contention."""
        if cls._inst is not None and not force:
            return cls._inst
        os.makedirs(_DIR, exist_ok=True)
        fd = os.open(_PATH, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            data = None
            if not force:
                try:
                    raw = os.pread(fd, os.fstat(fd).st_size, 0)
                    if raw:
                        cand = json.loads(raw.decode())
                        fresh = time.time() - float(cand.get("ts", 0)) < _TTL_S
                        same = set(cand.get("devices", [])) >= set(devices)
                        if fresh and same:
                            data = cand
                except Exception:  # noqa: BLE001
                    data = None                      # unreadable/stale -> re-measure
            if data is None:
                t0 = time.perf_counter()
                data = _measure(devices)
                blob = json.dumps(data).encode()
                os.ftruncate(fd, 0)
                os.pwrite(fd, blob, 0)
                logger.info("link_bw: measured in %.0f ms -> %s",
                            (time.perf_counter() - t0) * 1e3, _PATH)
            cls._inst = cls(data)
            return cls._inst
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    # -------------------------------------------------------------------- queries

    def peer_bw(self, src: int, dst: int) -> float:
        """GB/s for reading memory resident on `src` from `dst`. 0.0 = no usable path."""
        if src == dst:
            return float("inf")          # local: never worse than any alternative
        return float(self.peer.get(f"{src}->{dst}", 0.0))

    def host_bw(self) -> float:
        """GB/s H2D from pinned host memory. Falls back to a conservative constant if
        the probe failed, so a failed measurement never makes the host look infinitely
        good (which would send everything to CPU DRAM)."""
        return self._host or 20.0

    def better_than_host(self, src: int, dst: int, margin: float = _MARGIN) -> bool:
        """Is parking on `src` (to be fetched by `dst`) faster than CPU DRAM?
        This is the test that keeps a PCIe-only peer (3.3 GB/s measured) from being
        preferred over the host (26.3 GB/s)."""
        if src == dst:
            return True
        return self.peer_bw(src, dst) >= self.host_bw() * margin

    def describe(self) -> str:
        pairs = ", ".join(f"{k}={v:g}" for k, v in sorted(self.peer.items()))
        return (f"host_h2d={self.data.get('host_h2d')} host_d2h={self.data.get('host_d2h')} "
                f"GB/s | peer: {pairs}")

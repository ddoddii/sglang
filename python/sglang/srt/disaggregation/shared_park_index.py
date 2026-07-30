from __future__ import annotations

"""Shared-memory cross-node park index (Phase 2 slice-2, piece 1).

In 2P2D idle-KV-parking, a decode-finished conversation's KV is parked onto whichever
prefill GPU is momentarily idle (pressure-aware). The next turn may be prefilled by a
DIFFERENT prefill node, which must be able to DISCOVER that the KV was parked (and on
which GPU) so it can fetch it. A per-process in-memory index cannot serve that: the
discovery must be shared across the independently-launched prefill processes.

SharedParkIndex is a fixed-capacity open-addressing hash table living in a /dev/shm
mmap file that every prefill process attaches to. Key = the polynomial prefix hash of
the parked token_ids (same hash idle_kv_parking._prefix_hash produces); value =
(gpu, start, n) locating the KV in that GPU's park pool. Concurrency is guarded by
fcntl.flock on the backing fd (shared lock for lookup, exclusive for insert/remove);
at park/fetch rates (~10s/s) the lock cost is negligible.

Entries carry a timestamp for LRU eviction within the bounded probe window. Correctness
note: when a park pool's ring buffer overwrites slots, the writing node must remove the
stale entries (it knows their hashes from its own local index) via remove(); otherwise a
lookup could return a slot now holding different tokens. The n==L length guard in the
caller plus exact-hash match makes a false hit astronomically unlikely, but stale
same-length entries are still removed for correctness.

LOCATION (KV-aware unified memory): an entry also records WHERE the KV physically
lives -- LOC_GPU (dev = CUDA ordinal) or LOC_HOST (dev = the owning process's pid,
because host allocations cannot be shared across processes: pinned host memory has
no IPC handle, so only the process that allocated it can read it). A fetcher must
therefore check `loc` before deciding how to read:

    LOC_GPU  -> peer park pool via the CUDA IPC mapping (any node)
    LOC_HOST -> readable ONLY if entry.dev == os.getpid(); otherwise treat as a miss

Keeping host entries in the shared index anyway (rather than in a private table) is
deliberate: another node learning "this prefix exists but is host-resident elsewhere"
can choose to recompute instead of re-parking a duplicate.
"""

import fcntl
import mmap
import os
import struct
import time
from typing import NamedTuple, Optional

# Physical residency of a parked KV. Values are persisted in the shm file, so do not
# renumber them without bumping _MAGIC.
LOC_GPU = 0    # dev = CUDA device ordinal; readable by any node via IPC
LOC_HOST = 1   # dev = owning pid; readable only in that process (no IPC for pinned host)

_MAGIC = 0x504B4959  # 'PKIY' -- bumped from PKIX when `loc` was added to the entry, so
                     # an old-format file is reinitialized rather than misparsed.
_HEADER = struct.Struct("<qq")            # magic, capacity
_ENTRY = struct.Struct("<qiiiid")         # hash(nonzero), loc, dev, start, n, ts
_ENTRY_SIZE = _ENTRY.size                 # 32 bytes
_WINDOW = 64                              # bounded linear-probe window
_EMPTY = 0                                # hash sentinel for an empty slot


class ParkEntry(NamedTuple):
    """Where a parked prefix lives. `dev` means a CUDA ordinal when loc == LOC_GPU and
    the owning pid when loc == LOC_HOST (see the module docstring)."""

    loc: int
    dev: int
    start: int
    n: int

    @property
    def is_gpu(self) -> bool:
        return self.loc == LOC_GPU

    def readable_here(self) -> bool:
        """Host-resident KV is only reachable inside the process that allocated it."""
        return self.loc == LOC_GPU or self.dev == os.getpid()


class SharedParkIndex:
    def __init__(self, path: str, capacity: int = 16384):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Create-or-attach with the header written exactly once under an exclusive lock.
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            st = os.fstat(fd)
            need = _HEADER.size + capacity * _ENTRY_SIZE

            def reinit():
                """Truncate AND zero the entry region, then write the header.

                Zeroing is not optional. ftruncate alone leaves the old bytes in place
                whenever the file was already at least this large -- e.g. a file from a
                previous run with a different _MAGIC or _ENTRY layout. A leftover
                nonzero leading 8 bytes reads as a valid hash, so lookup() could match
                it and return a (dev, start, n) decoded from a different record layout,
                i.e. hand the caller someone else's KV slots."""
                os.ftruncate(fd, need)
                os.pwrite(fd, b"\0" * (capacity * _ENTRY_SIZE), _HEADER.size)
                os.pwrite(fd, _HEADER.pack(_MAGIC, capacity), 0)

            if st.st_size < _HEADER.size:
                reinit()
                self.capacity = capacity
            else:
                magic, cap = _HEADER.unpack(os.pread(fd, _HEADER.size, 0))
                if magic != _MAGIC:
                    # foreign file, or one written by an older entry layout -> reinit
                    reinit()
                    self.capacity = capacity
                else:
                    self.capacity = cap
                    want = _HEADER.size + self.capacity * _ENTRY_SIZE
                    if st.st_size < want:
                        os.ftruncate(fd, want)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
        self.fd = fd
        self.size = _HEADER.size + self.capacity * _ENTRY_SIZE
        self.mm = mmap.mmap(fd, self.size)

    # --- internal slot access (mm must be valid; caller holds the lock) ---
    def _read(self, i: int):
        off = _HEADER.size + i * _ENTRY_SIZE
        return _ENTRY.unpack(self.mm[off:off + _ENTRY_SIZE])

    def _write(self, i: int, h: int, loc: int, dev: int, start: int, n: int, ts: float):
        off = _HEADER.size + i * _ENTRY_SIZE
        self.mm[off:off + _ENTRY_SIZE] = _ENTRY.pack(h, loc, dev, start, n, ts)

    @staticmethod
    def _norm(h: int) -> int:
        # reserve 0 for empty; also fold into unsigned 63-bit range for stable modulo.
        h &= (1 << 63) - 1
        return h or 1

    def insert(self, h: int, dev: int, start: int, n: int, loc: int = LOC_GPU) -> None:
        """Record that prefix `h` (n tokens) is parked at `start` on `dev`.
        loc=LOC_GPU -> dev is a CUDA ordinal; loc=LOC_HOST -> dev must be os.getpid()."""
        h = self._norm(h)
        now = time.time()
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            base = h % self.capacity
            free_i = -1
            lru_i = base
            lru_ts = float("inf")
            for w in range(_WINDOW):
                i = (base + w) % self.capacity
                eh, _el, _ed, _es, _en, ets = self._read(i)
                if eh == _EMPTY:
                    free_i = i
                    break
                if eh == h:  # update existing key in place
                    self._write(i, h, loc, dev, start, n, now)
                    return
                if ets < lru_ts:
                    lru_ts, lru_i = ets, i
            idx = free_i if free_i >= 0 else lru_i  # empty slot, else evict LRU in window
            self._write(idx, h, loc, dev, start, n, now)
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def lookup(self, h: int) -> Optional[ParkEntry]:
        """Return a ParkEntry(loc, dev, start, n) for hash h, or None. The caller must
        still require an exact length match (entry.n == L) and, for host-resident
        entries, check entry.readable_here().

        NOTE: this used to return a bare (gpu, start, n) 3-tuple. It now returns a
        4-field NamedTuple, so any old `gpu, start, n = lookup(...)` unpack raises
        immediately instead of silently mis-assigning."""
        h = self._norm(h)
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        try:
            base = h % self.capacity
            for w in range(_WINDOW):
                i = (base + w) % self.capacity
                eh, el, ed, es, en, _ets = self._read(i)
                if eh == _EMPTY:
                    return None            # probe stops at first empty slot
                if eh == h:
                    return ParkEntry(el, ed, es, en)
            return None
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def remove(self, h: int) -> bool:
        """Remove hash h if present. Returns True if removed. Uses backshift so the
        probe-stop-on-empty invariant of lookup() stays correct."""
        h = self._norm(h)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        try:
            base = h % self.capacity
            found = -1
            for w in range(_WINDOW):
                i = (base + w) % self.capacity
                eh = self._read(i)[0]
                if eh == _EMPTY:
                    break
                if eh == h:
                    found = i
                    break
            if found < 0:
                return False
            # backward-shift deletion within the window so no reachable key is orphaned.
            self._write(found, _EMPTY, 0, 0, 0, 0, 0.0)
            j = found
            for w in range(1, _WINDOW):
                k = (found + w) % self.capacity
                eh, el, ed, es, en, ets = self._read(k)
                if eh == _EMPTY:
                    break
                home = eh % self.capacity
                # is k's key reachable from home through the freed slot j?
                if self._in_range(home, j, k):
                    self._write(j, eh, el, ed, es, en, ets)
                    self._write(k, _EMPTY, 0, 0, 0, 0, 0.0)
                    j = k
            return True
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def _in_range(self, home: int, j: int, k: int) -> bool:
        # true if slot j lies in the circular interval [home, k]; used by backshift.
        if home <= k:
            return home <= j <= k
        return j >= home or j <= k

    def close(self):
        try:
            self.mm.close()
        finally:
            os.close(self.fd)

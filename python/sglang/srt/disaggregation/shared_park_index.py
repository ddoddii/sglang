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
"""

import fcntl
import mmap
import os
import struct
import time
from typing import Optional, Tuple

_MAGIC = 0x504B4958  # 'PKIX'
_HEADER = struct.Struct("<qq")            # magic, capacity
_ENTRY = struct.Struct("<qiiid")          # hash(nonzero), gpu, start, n, ts
_ENTRY_SIZE = _ENTRY.size                 # 28 bytes
_WINDOW = 64                              # bounded linear-probe window
_EMPTY = 0                                # hash sentinel for an empty slot


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
            if st.st_size < _HEADER.size:
                os.ftruncate(fd, need)
                os.pwrite(fd, _HEADER.pack(_MAGIC, capacity), 0)
                self.capacity = capacity
            else:
                magic, cap = _HEADER.unpack(os.pread(fd, _HEADER.size, 0))
                if magic != _MAGIC:
                    # foreign/garbage file -> reinit
                    os.ftruncate(fd, need)
                    os.pwrite(fd, _HEADER.pack(_MAGIC, capacity), 0)
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

    def _write(self, i: int, h: int, gpu: int, start: int, n: int, ts: float):
        off = _HEADER.size + i * _ENTRY_SIZE
        self.mm[off:off + _ENTRY_SIZE] = _ENTRY.pack(h, gpu, start, n, ts)

    @staticmethod
    def _norm(h: int) -> int:
        # reserve 0 for empty; also fold into unsigned 63-bit range for stable modulo.
        h &= (1 << 63) - 1
        return h or 1

    def insert(self, h: int, gpu: int, start: int, n: int) -> None:
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
                eh, eg, es, en, ets = self._read(i)
                if eh == _EMPTY:
                    free_i = i
                    break
                if eh == h:  # update existing key in place
                    self._write(i, h, gpu, start, n, now)
                    return
                if ets < lru_ts:
                    lru_ts, lru_i = ets, i
            idx = free_i if free_i >= 0 else lru_i  # empty slot, else evict LRU in window
            self._write(idx, h, gpu, start, n, now)
        finally:
            fcntl.flock(self.fd, fcntl.LOCK_UN)

    def lookup(self, h: int) -> Optional[Tuple[int, int, int]]:
        """Return (gpu, start, n) for hash h, or None. Requires exact-length match by the
        caller (n) — the value's n is returned so the caller can assert it."""
        h = self._norm(h)
        fcntl.flock(self.fd, fcntl.LOCK_SH)
        try:
            base = h % self.capacity
            for w in range(_WINDOW):
                i = (base + w) % self.capacity
                eh, eg, es, en, ets = self._read(i)
                if eh == _EMPTY:
                    return None            # probe stops at first empty slot
                if eh == h:
                    return (eg, es, en)
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
                eh, _, _, _, _ = self._read(i)
                if eh == _EMPTY:
                    break
                if eh == h:
                    found = i
                    break
            if found < 0:
                return False
            # backward-shift deletion within the window so no reachable key is orphaned.
            self._write(found, _EMPTY, 0, 0, 0, 0.0)
            j = found
            for w in range(1, _WINDOW):
                k = (found + w) % self.capacity
                eh, eg, es, en, ets = self._read(k)
                if eh == _EMPTY:
                    break
                home = eh % self.capacity
                # is k's key reachable from home through the freed slot j?
                if self._in_range(home, j, k):
                    self._write(j, eh, eg, es, en, ets)
                    self._write(k, _EMPTY, 0, 0, 0, 0.0)
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

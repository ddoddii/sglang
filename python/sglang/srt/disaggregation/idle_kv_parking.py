from __future__ import annotations

"""Idle KV parking (PD disaggregation).

During tool-call idle windows in agentic multi-turn workloads, a decode (D) node's
conversation prefix KV is parked into an idle prefill (P) node's GPU radix cache over
NVLink, so the next turn prefix-hits on P instead of recomputing. Overflow demotes to
the host DRAM tier. See docs/developer_guide/idle_kv_parking_design.md.

This module is being built incrementally:
  slice 1 (this file): scaffolding + flag wiring (no-op).
  slice 2: NIXL reverse D->P VRAM channel.
  slice 3: D-side park trigger on idle.
  slice 4: P-side receive + HiRadixCache insert.
"""

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.server_args import ServerArgs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


class IdleKVParkManager:
    """Manage idle-time KV parking between decode and prefill nodes.

    One instance lives on each PD node; ``role`` ("prefill" or "decode") selects the
    side-specific behavior. Slice 1 only wires construction; park/receive are no-ops.
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

        logger.info(
            "Idle KV parking enabled (role=%s) [scaffold: transfer/insert not yet wired]",
            role,
        )

    # --- decode side ------------------------------------------------------------
    def park(self, req: "Req") -> bool:
        """Park a finished/idle request's prefix KV to the peer prefill node.

        Slice 3 will implement the D->P push. No-op for now.
        """
        return False

    # --- prefill side -----------------------------------------------------------
    def poll_incoming(self) -> None:
        """Receive parked KV and insert into the local radix cache.

        Slice 4 will implement receive + HiRadixCache insert. No-op for now.
        """
        return None

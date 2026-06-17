"""Sequential load/run/unload cycle with warm-model caching (S12.2).

The scheduler tracks which adapter (model) is warm on which CUDA node. A
request for an adapter that is already warm on all requested nodes is a
no-op; otherwise any different warm adapter on those nodes is unloaded
(once - ``unload()`` releases the whole model, so its other nodes free too)
before the requested adapter is loaded.

Adapters are duck-typed: only ``name``, ``load()``, ``unload()`` and
``reported_vram_gb`` are used.
"""

from __future__ import annotations

import logging

from anvesha.core.adapters.base import LLMClient
from anvesha.core.gpu.manager import GpuManager

logger = logging.getLogger(__name__)


class GpuScheduler:
    """Implements the S12.2 sequential agent call cycle for a GpuManager."""

    def __init__(self, manager: GpuManager) -> None:
        self.manager = manager
        #: node id -> the adapter currently warm on that node.
        self._warm: dict[int, LLMClient] = {}

    def ensure_loaded(self, adapter: LLMClient, nodes: list[int]) -> None:
        """Make ``adapter`` resident on ``nodes``, evicting other models first."""
        if nodes and all(
            node in self._warm and self._warm[node].name == adapter.name
            for node in nodes
        ):
            logger.debug("%s already warm on nodes %s", adapter.name, nodes)
            return

        # Evict each different warm adapter exactly once. unload() releases
        # the whole model, so it goes cold on every node it occupied.
        evicted: list[LLMClient] = []
        for node in nodes:
            current = self._warm.get(node)
            if current is None or current.name == adapter.name:
                continue
            if any(current is seen for seen in evicted):
                continue
            evicted.append(current)
        for current in evicted:
            occupied = sorted(n for n, a in self._warm.items() if a is current)
            current.unload()
            for node in occupied:
                del self._warm[node]
                logger.info("unloaded %s from node %d", current.name, node)

        adapter.load()
        for node in nodes:
            self._warm[node] = adapter
            logger.info("loaded %s on node %d", adapter.name, node)

    def release_all(self) -> None:
        """Unload every warm adapter (each exactly once) and clear the registry."""
        remaining: list[LLMClient] = []
        for adapter in self._warm.values():
            if not any(adapter is seen for seen in remaining):
                remaining.append(adapter)
        for adapter in remaining:
            occupied = sorted(n for n, a in self._warm.items() if a is adapter)
            adapter.unload()
            for node in occupied:
                logger.info("unloaded %s from node %d", adapter.name, node)
        self._warm.clear()

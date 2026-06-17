"""Programmatic MCP access for Phase 1 (spec S2 search, S8 PDF acquisition).

Stages 2 and 8 call MCP tools directly -- NOT through an LLM tool-call loop --
so they need a deterministic name -> client lookup. ``ToolRouter`` builds that
map from the injected MCP clients' advertised tools (namespaced names like
``"papersflow.search"`` or ``"paper-search.download_with_fallback"``).

Boundary rule (charter S12): the router only consumes injected
:class:`MCPClient` instances; it never constructs MCP clients itself.
"""

from __future__ import annotations

import logging
from typing import Sequence

from anvesha.core.adapters.base import MCPClient
from anvesha.core.exceptions import AnveshaError

logger = logging.getLogger(__name__)


class ToolRouter:
    """Maps namespaced tool names to the injected MCP client that provides them.

    When more than one client advertises the same tool name, the first client
    in injection order wins (deterministic, and the duplicate is logged).
    """

    def __init__(self, mcps: Sequence[MCPClient]):
        self._clients: dict[str, MCPClient] = {}
        self._tool_names: list[str] = []
        for client in mcps:
            try:
                tools = client.list_tools()
            except Exception as exc:  # noqa: BLE001 - degrade gracefully (EH-2)
                # A client that cannot enumerate its tools is skipped rather
                # than aborting router construction; missing coverage is
                # handled at the stage level.
                logger.warning("MCP client list_tools() failed; skipping: %s", exc)
                continue
            for tool in tools:
                if tool.name in self._clients:
                    logger.warning(
                        "Duplicate MCP tool %r; keeping first-registered client",
                        tool.name,
                    )
                    continue
                self._clients[tool.name] = client
                self._tool_names.append(tool.name)

    @property
    def tool_names(self) -> list[str]:
        """All registered tool names, in first-seen (injection) order."""
        return list(self._tool_names)

    def has(self, tool_name: str) -> bool:
        """True if some injected client provides ``tool_name``."""
        return tool_name in self._clients

    def call(self, tool_name: str, arguments: dict) -> str:
        """Dispatch ``tool_name`` to its client and return the raw JSON string.

        Raises :class:`AnveshaError` if no injected client provides the tool;
        Stage 2 treats a missing optional source as degraded coverage (EH-2),
        but a total absence of search tools is fatal at the orchestrator (EH-3).
        """
        client = self._clients.get(tool_name)
        if client is None:
            available = ", ".join(sorted(self._clients)) or "(none)"
            raise AnveshaError(
                f"No MCP client provides tool {tool_name!r}; available: {available}"
            )
        return client.call_tool(tool_name, arguments)

"""PipelineHooks - pre/post agent callbacks (research_gap_pipeline S11 layout).

Observability seam for the orchestrator: callbacks fire around each agent
invocation, in registration order. Exceptions propagate to the caller.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

HookFn = Callable[[str, dict], None]


class PipelineHooks:
    """Registry of pre/post agent callbacks, invoked in registration order."""

    def __init__(self) -> None:
        self._pre_agent: list[HookFn] = []
        self._post_agent: list[HookFn] = []

    def add_pre_agent(self, fn: HookFn) -> None:
        self._pre_agent.append(fn)

    def add_post_agent(self, fn: HookFn) -> None:
        self._post_agent.append(fn)

    def fire_pre_agent(self, agent_name: str, payload: dict) -> None:
        logger.debug("pre-agent hooks for %s", agent_name)
        for fn in self._pre_agent:
            fn(agent_name, payload)

    def fire_post_agent(self, agent_name: str, payload: dict) -> None:
        logger.debug("post-agent hooks for %s", agent_name)
        for fn in self._post_agent:
            fn(agent_name, payload)

"""FallbackAdapter: local-first with cloud fallback on failure triggers
(ANVESHA.md S10.2, research_gap_pipeline_implementation.md S15.4).

The same input is sent to the fallback adapter unchanged when a trigger
condition is met. If the fallback also fails, the ORIGINAL primary error is
raised - the fallback failure never swallows it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient
from anvesha.core.exceptions import (
    AdapterParseError,
    AdapterTimeoutError,
    GpuOomError,
    LowConfidenceError,
)

logger = logging.getLogger(__name__)

FallbackTrigger = Literal[
    "timeout", "parse_error", "gpu_oom", "low_confidence", "explicit"
]

#: Exception class each (non-explicit) trigger keys on.
_TRIGGER_EXCEPTIONS: dict[str, type[Exception]] = {
    "timeout": AdapterTimeoutError,
    "parse_error": AdapterParseError,
    "gpu_oom": GpuOomError,
    "low_confidence": LowConfidenceError,
}


class FallbackAdapterConfig(BaseModel):
    """Configuration for :class:`FallbackAdapter` (S15.4)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    #: Local adapter, tried first.
    primary: LLMClient
    #: Cloud adapter, tried on a matched trigger.
    fallback: LLMClient
    fallback_on: list[FallbackTrigger] = Field(
        default_factory=lambda: ["timeout", "parse_error", "gpu_oom"]
    )
    log_fallback_events: bool = True
    #: Budget for the fallback call (S15.4). Not enforced at this layer:
    #: ``LLMClient.run`` takes no timeout, so the fallback backend's own
    #: ``timeout_seconds`` config governs. Kept for factory wiring.
    fallback_timeout_seconds: int = 60


class FallbackAdapter(LLMClient):
    """LLMClient that tries ``primary`` and falls back to ``fallback`` when
    the primary raises an error matching a configured trigger.

    The ``force_fallback`` keyword on :meth:`run` / :meth:`structured_run`
    routes straight to the fallback when the ``"explicit"`` trigger is
    enabled in ``fallback_on``; otherwise it is ignored.
    """

    def __init__(self, config: FallbackAdapterConfig):
        self.config = config

    @property
    def name(self) -> str:  # type: ignore[override]
        return self.config.primary.name

    @property
    def reported_vram_gb(self) -> float | None:  # type: ignore[override]
        return self.config.primary.reported_vram_gb

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
        *,
        force_fallback: bool = False,
    ) -> LLMResponse:
        return self._call(
            "run", force_fallback, prompt, mcps=mcps, input_files=input_files
        )

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
        *,
        force_fallback: bool = False,
    ) -> dict:
        return self._call(
            "structured_run",
            force_fallback,
            prompt,
            json_schema,
            mcps=mcps,
            input_files=input_files,
        )

    def _call(
        self, method: str, force_fallback: bool, *args: Any, **kwargs: Any
    ) -> Any:
        if force_fallback and "explicit" in self.config.fallback_on:
            self._log_event(method, "explicit", None)
            return getattr(self.config.fallback, method)(*args, **kwargs)

        try:
            return getattr(self.config.primary, method)(*args, **kwargs)
        except Exception as primary_exc:
            trigger = self._match_trigger(primary_exc)
            if trigger is None:
                raise
            self._log_event(method, trigger, primary_exc)
            try:
                # Same input, unchanged (S15.4).
                return getattr(self.config.fallback, method)(*args, **kwargs)
            except Exception as fallback_exc:
                if self.config.log_fallback_events:
                    logger.warning(
                        "fallback %s also failed (%s); raising the original "
                        "primary error",
                        self.config.fallback.name,
                        fallback_exc,
                    )
                # S15.4: raise the original error, do not swallow it. The
                # fallback failure stays visible as __context__.
                raise primary_exc

    def _match_trigger(self, exc: Exception) -> str | None:
        for trigger in self.config.fallback_on:
            exc_cls = _TRIGGER_EXCEPTIONS.get(trigger)
            if exc_cls is not None and isinstance(exc, exc_cls):
                return trigger
        return None

    def _log_event(
        self, method: str, trigger: str, primary_exc: Exception | None
    ) -> None:
        if not self.config.log_fallback_events:
            return
        logger.warning(
            "fallback event: %s -> %s (trigger=%s, method=%s, error=%s)",
            self.config.primary.name,
            self.config.fallback.name,
            trigger,
            method,
            primary_exc,
        )

"""Generic failure classification and retry primitives (S14.5).

This module provides only the phase-agnostic machinery: exception ->
failure_type classification, exponential-backoff retries, and FailureRecord
construction. The per-failure-type pipeline actions in S14.5 that need phase
context (fallback adapter routing, schema-reminder retries, context-length
reduction, search skipping) are composed from these primitives at phase
level, not here (ANVESHA.md S12.2).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from anvesha.core.checkpoint.manager import FailureRecord
from anvesha.core.exceptions import (
    AdapterParseError,
    AdapterTimeoutError,
    GpuOomError,
)

logger = logging.getLogger(__name__)


def classify_failure(exc: Exception) -> str:
    """Map an exception to an S14.5 failure_type string."""
    if isinstance(exc, AdapterTimeoutError):
        return "timeout"
    if isinstance(exc, AdapterParseError):
        return "parse_error"
    if isinstance(exc, GpuOomError):
        return "gpu_oom"
    return "unknown"


def with_retries(
    fn: Callable[[], Any],
    *,
    max_retries: int = 3,
    backoff_seconds: float = 5,
    retry_on: tuple = (AdapterTimeoutError,),
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Call ``fn`` with up to ``max_retries`` retries and exponential backoff.

    Delay before retry N (0-based) is ``backoff_seconds * 2**N``. Exceptions
    not in ``retry_on`` propagate immediately; retryable exceptions are
    re-raised once retries are exhausted.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retry_on as exc:
            if attempt >= max_retries:
                raise
            delay = backoff_seconds * 2**attempt
            logger.warning(
                "Retryable failure (%s) on attempt %d/%d; retrying in %.1fs",
                type(exc).__name__,
                attempt + 1,
                max_retries + 1,
                delay,
            )
            sleep(delay)


def make_failure_record(
    agent_name: str,
    exc: Exception,
    recovery_action: str = "",
    gap_id: str | None = None,
) -> FailureRecord:
    """Build a FailureRecord for the checkpoint failure log (S14.3)."""
    return FailureRecord(
        failed_at=datetime.now(timezone.utc).isoformat(),
        agent_name=agent_name,
        failure_type=classify_failure(exc),
        error_message=str(exc),
        recovery_action=recovery_action,
        gap_id=gap_id,
    )

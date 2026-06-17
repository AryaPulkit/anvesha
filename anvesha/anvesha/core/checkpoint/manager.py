"""Checkpoint state schema and manager (S14.3, S14.5, S14.6).

Core stays phase-agnostic (ANVESHA.md S12.2): ``pipeline_state`` and
``resume_hint`` are opaque dicts owned by the calling phase, and
``input_hashes`` is a generic name -> sha256 mapping.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, Field

from anvesha.core.config import CheckpointConfig
from anvesha.core.checkpoint.serializer import StateSerializer
from anvesha.core.exceptions import CheckpointError

logger = logging.getLogger(__name__)

KEEP_VERSIONS = 5  # S14.6: keep last 5 checkpoints per run_id

FailureType = Literal[
    "timeout",
    "parse_error",
    "gpu_oom",
    "search_failure",
    "checkpoint_write_failure",
    "unknown",
]


class FailureRecord(BaseModel):
    """One failure entry in the checkpoint failure log (S14.3)."""

    failed_at: str
    agent_name: str
    failure_type: FailureType
    error_message: str
    recovery_action: str = ""
    gap_id: str | None = None


class CheckpointState(BaseModel):
    """Persisted pipeline snapshot (S14.3, generic form)."""

    pipeline_run_id: str
    checkpoint_version: int
    created_at: str
    last_saved_at: str
    granularity: str
    pipeline_state: dict
    resume_hint: dict
    input_hashes: dict[str, str]
    failure_log: list[FailureRecord] = Field(default_factory=list)


class CheckpointManager:
    """Versioned save/load of CheckpointState with S14.6 retention."""

    def __init__(self, config: CheckpointConfig, directory: Path) -> None:
        self.config = config
        self._serializer = StateSerializer(config.storage, directory)

    def save(
        self,
        run_id: str,
        pipeline_state: dict,
        resume_hint: dict,
        input_hashes: dict[str, str],
        failure_log: Sequence[FailureRecord] = (),
    ) -> int | None:
        """Write the next checkpoint version and prune old ones.

        Returns the version written, or None when checkpointing is disabled
        or the write fails. Per S14.5 (checkpoint_write_failure) this method
        NEVER raises: a write failure is logged as a warning and execution
        continues with state still in memory.
        """
        if not self.config.enabled:
            return None
        now = datetime.now(timezone.utc).isoformat()
        try:
            versions = self._serializer.list_versions(run_id)
            version = versions[-1] + 1 if versions else 1
            created_at = now
            if versions:
                try:
                    created_at = self._serializer.load(run_id, versions[-1]).get(
                        "created_at", now
                    )
                except Exception:  # corrupt previous checkpoint must not block saves
                    logger.warning(
                        "Could not read previous checkpoint for run %s; "
                        "resetting created_at",
                        run_id,
                    )
            state = CheckpointState(
                pipeline_run_id=run_id,
                checkpoint_version=version,
                created_at=created_at,
                last_saved_at=now,
                granularity=self.config.granularity,
                pipeline_state=pipeline_state,
                resume_hint=resume_hint,
                input_hashes=input_hashes,
                failure_log=list(failure_log),
            )
            self._serializer.save(run_id, version, state.model_dump())
        except Exception:
            logger.warning(
                "Checkpoint write failed for run %s; continuing without "
                "checkpoint (S14.5)",
                run_id,
                exc_info=True,
            )
            return None

        # The write is durable. Pruning is best-effort and must not mask a
        # successful save by reporting write-failure (S14.5).
        try:
            self._serializer.prune(run_id, keep=KEEP_VERSIONS)
        except Exception:
            logger.warning(
                "Checkpoint prune failed for run %s; keeping all versions",
                run_id,
                exc_info=True,
            )
        return version

    def load_latest(self, run_id: str) -> CheckpointState | None:
        """Return the most recent CheckpointState, or None if none exists."""
        try:
            payload = self._serializer.load(run_id)
        except CheckpointError:
            return None
        try:
            return CheckpointState.model_validate(payload)
        except Exception as exc:
            raise CheckpointError(
                f"Checkpoint for run {run_id!r} could not be reconstructed: {exc}"
            ) from exc

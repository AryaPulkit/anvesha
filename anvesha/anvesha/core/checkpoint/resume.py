"""Resume decision logic (S14.4) and input hash validation (S14.7).

Input hashes are a generic name -> sha256 mapping so core stays
phase-agnostic; phases decide which named inputs participate.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Literal

from pydantic import BaseModel

from anvesha.core.checkpoint.manager import CheckpointManager, CheckpointState

logger = logging.getLogger(__name__)


def hash_inputs(named_contents: dict[str, str]) -> dict[str, str]:
    """SHA-256 hex digest of each named input's content (S14.7)."""
    return {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in named_contents.items()
    }


class ResumePlan(BaseModel):
    """Outcome of the S14.4 resume decision sequence."""

    action: Literal["fresh", "resume"]
    checkpoint: CheckpointState | None = None
    inputs_changed: bool = False
    reason: str


class ResumePlanner:
    """Decides between starting fresh and resuming from a checkpoint."""

    def __init__(self, manager: CheckpointManager) -> None:
        self.manager = manager

    def plan(self, run_id: str, current_input_hashes: dict[str, str]) -> ResumePlan:
        checkpoint = self.manager.load_latest(run_id)
        if checkpoint is None:
            return ResumePlan(
                action="fresh",
                reason=f"No checkpoint found for run {run_id!r}; starting fresh.",
            )
        if checkpoint.input_hashes != current_input_hashes:
            # S14.7: inputs changed since the checkpoint was written. Default
            # is start fresh; the caller should warn the user and may offer
            # resume-anyway using the attached checkpoint.
            return ResumePlan(
                action="fresh",
                checkpoint=checkpoint,
                inputs_changed=True,
                reason=(
                    f"Input hashes for run {run_id!r} differ from checkpoint "
                    f"version {checkpoint.checkpoint_version}; warn the user "
                    "and start fresh by default (resume-anyway is opt-in)."
                ),
            )
        logger.info(
            "Resuming run %s from checkpoint version %d",
            run_id,
            checkpoint.checkpoint_version,
        )
        return ResumePlan(
            action="resume",
            checkpoint=checkpoint,
            reason=(
                f"Checkpoint version {checkpoint.checkpoint_version} matches "
                "current inputs; resuming."
            ),
        )

"""Pydantic schemas for the research project workspace (ANVESHA.md S6.2).

Kept deliberately minimal: these mirror the on-disk workspace structures,
they do not duplicate the full config schema in ``core/config``.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ProjectMeta(BaseModel):
    """Identity of a research project workspace."""

    name: str
    root: Path


class WorkspaceConfig(BaseModel):
    """Mirror of the ``project:`` block in ``.anvesha/config.yaml`` (ANVESHA.md S8.2)."""

    name: str = "research-project"
    #: Path to the researcher's separate code repository (never written to).
    code_repo: str | None = None


class PhaseFilePaths(BaseModel):
    """Workspace-relative input/output paths for a single phase."""

    phase_id: int
    inputs: list[str] = Field(default_factory=list)
    output: str = ""

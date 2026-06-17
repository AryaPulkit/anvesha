"""Research project workspace root: location, validation, skeleton creation.

Implements the workspace layout from ANVESHA.md S6.2 /
research_gap_pipeline_implementation.md S11.2. Per the boundary rules
(ANVESHA.md S12.6), phase entry points receive a :class:`ResearchProject`
instead of raw paths.
"""

from __future__ import annotations

import logging
from pathlib import Path

from anvesha.core.exceptions import WorkspaceError

logger = logging.getLogger(__name__)

_CONFIG_RELPATH = Path(".anvesha") / "config.yaml"

_GITIGNORE_CONTENT = ".anvesha/\ndata/\n"

_VERSION_LOG_HEADER = (
    "# Version Log\n"
    "\n"
    "| timestamp | phase | output | run_id | note |\n"
    "|---|---|---|---|---|\n"
)


class ResearchProject:
    """A researcher-owned workspace directory (ANVESHA.md S6.2)."""

    def __init__(self, root: Path | str) -> None:
        self.root: Path = Path(root).resolve()

    @classmethod
    def find(cls, start: Path | str = ".") -> "ResearchProject":
        """Walk up from ``start`` to the directory containing ``.anvesha/config.yaml``."""
        start_path = Path(start).resolve()
        for candidate in (start_path, *start_path.parents):
            if (candidate / _CONFIG_RELPATH).is_file():
                return cls(candidate)
        raise WorkspaceError(
            f"No Anvesha workspace found at or above {start_path} "
            "(missing .anvesha/config.yaml)"
        )

    def validate(self) -> None:
        """Raise :class:`WorkspaceError` unless this looks like a valid workspace."""
        if not self.config_path.is_file():
            raise WorkspaceError(
                f"Not an Anvesha workspace: {self.root} (missing {_CONFIG_RELPATH})"
            )

    @property
    def config_path(self) -> Path:
        return self.root / _CONFIG_RELPATH

    @property
    def phases_dir(self) -> Path:
        return self.root / "phases"

    @property
    def checkpoints_dir(self) -> Path:
        return self.root / ".anvesha" / "checkpoints"

    @property
    def logs_dir(self) -> Path:
        return self.root / ".anvesha" / "logs"

    @property
    def version_log_path(self) -> Path:
        return self.root / "_VERSION_LOG.md"

    def resolve(self, relpath: str) -> Path:
        """Resolve a workspace-relative path against the project root."""
        return self.root / relpath

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ResearchProject(root={str(self.root)!r})"


def init_workspace(root: Path | str, name: str) -> ResearchProject:
    """Create the ANVESHA.md S6.2 workspace skeleton under ``root``.

    Raises :class:`WorkspaceError` if ``root`` already contains
    ``.anvesha/config.yaml``. Pre-existing researcher files (README.md,
    .gitignore) are never overwritten.
    """
    project = ResearchProject(root)
    if project.config_path.exists():
        raise WorkspaceError(
            f"{project.root} is already an Anvesha workspace "
            f"({_CONFIG_RELPATH} exists)"
        )

    for directory in (
        project.checkpoints_dir,
        project.logs_dir,
        project.phases_dir,
        project.root / "data",
        project.root / "refs" / "pdfs",
        project.root / "runs",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    readme = project.root / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {name}\n\n## Topic Statement\n\n<!-- topic statement -->\n",
            encoding="utf-8",
        )

    if not project.version_log_path.exists():
        project.version_log_path.write_text(_VERSION_LOG_HEADER, encoding="utf-8")

    project.config_path.write_text(
        "# Anvesha project configuration.\n"
        "# Minimal config - see ANVESHA.md S8 for the full structure and defaults.\n"
        "project:\n"
        f'  name: "{name}"\n'
        "  code_repo: null\n",
        encoding="utf-8",
    )

    gitignore = project.root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(_GITIGNORE_CONTENT, encoding="utf-8")

    logger.info("Initialised Anvesha workspace %r at %s", name, project.root)
    return project

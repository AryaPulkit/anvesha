"""All phase file I/O flows through this module (ANVESHA.md S12.3).

Agents and loops receive Markdown content as strings; only this module
touches the filesystem. Versioned re-runs follow ANVESHA.md S11: the base
path is used on the first run, then ``_v2``, ``_v3``, ... suffixes are
inserted before the file extension.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from anvesha.core.exceptions import WorkspaceError
from anvesha.workspace.project import ResearchProject

logger = logging.getLogger(__name__)


def read_text(project: ResearchProject, relpath: str) -> str:
    """Read a workspace file; :class:`WorkspaceError` if it does not exist."""
    path = project.resolve(relpath)
    if not path.is_file():
        raise WorkspaceError(f"Workspace file not found: {relpath} (in {project.root})")
    return path.read_text(encoding="utf-8")


def exists(project: ResearchProject, relpath: str) -> bool:
    """True if ``relpath`` exists in the workspace."""
    return project.resolve(relpath).exists()


def next_versioned_path(project: ResearchProject, relpath: str) -> Path:
    """Next unused output path per ANVESHA.md S11.

    Base path if unused; otherwise ``_v2``, ``_v3``, ... inserted before the
    suffix, picking max existing version + 1 (suffix-agnostic, so
    ``08_results.txt`` versions correctly too).
    """
    base = project.resolve(relpath)
    if not base.exists():
        return base
    max_version = max(_existing_versions(base), default=1)
    return base.with_name(f"{base.stem}_v{max_version + 1}{base.suffix}")


def write_phase_output(project: ResearchProject, relpath: str, content: str) -> Path:
    """Write ``content`` to the next versioned path; returns the path written."""
    path = next_versioned_path(project, relpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote phase output %s", path)
    return path


def write_text_file(project: ResearchProject, relpath: str, content: str) -> Path:
    """Write ``content`` to an EXACT workspace path (no versioning).

    Used by phases whose versioning happens at a coarser level than the
    individual file - e.g. Phase 1 versions its whole output directory
    (ANVESHA.md S11) and writes the index/PDFs at known paths inside it.
    """
    path = project.resolve(relpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote workspace file %s", path)
    return path


def write_bytes_file(project: ResearchProject, relpath: str, data: bytes) -> Path:
    """Write binary ``data`` to an EXACT workspace path (e.g. a downloaded PDF)."""
    path = project.resolve(relpath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    logger.info("Wrote workspace file %s (%d bytes)", path, len(data))
    return path


def list_versions(project: ResearchProject, relpath: str) -> list[Path]:
    """Existing versions of ``relpath``, base path first, then _v2, _v3, ..."""
    base = project.resolve(relpath)
    versions: list[Path] = [base] if base.exists() else []
    numbered = sorted(_existing_versions(base))
    versions.extend(
        base.with_name(f"{base.stem}_v{n}{base.suffix}") for n in numbered if n >= 2
    )
    return versions


def _existing_versions(base: Path) -> list[int]:
    """Version numbers N for existing ``<stem>_vN<suffix>`` siblings of ``base``."""
    if not base.parent.is_dir():
        return []
    pattern = re.compile(
        rf"^{re.escape(base.stem)}_v(\d+){re.escape(base.suffix)}$"
    )
    found: list[int] = []
    for sibling in base.parent.iterdir():
        match = pattern.match(sibling.name)
        if match:
            found.append(int(match.group(1)))
    return found

"""_VERSION_LOG.md read/write (ANVESHA.md S11).

The log is a parseable markdown table; one row is appended per recorded
phase output. Downstream phases use :meth:`VersionLog.current` to find the
active version of an output instead of the base path in config (ANVESHA.md
S8.5).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from anvesha.workspace.project import ResearchProject

logger = logging.getLogger(__name__)

_HEADER = (
    "# Version Log\n"
    "\n"
    "| timestamp | phase | output | run_id | note |\n"
    "|---|---|---|---|---|\n"
)


def _clean_cell(value: str) -> str:
    """Make a value safe to embed in a markdown table cell."""
    return value.replace("|", "/").replace("\n", " ").strip()


class VersionLog:
    """Append-only record of phase outputs in ``_VERSION_LOG.md``."""

    def __init__(self, project: ResearchProject) -> None:
        self._project = project
        self._path = project.version_log_path

    def record(
        self, phase_id: int, output_path: str, run_id: str = "", note: str = ""
    ) -> None:
        """Append a row for ``phase_id``; creates the log file if missing.

        ``output_path`` is stored workspace-relative; absolute paths under
        the project root are relativised.
        """
        rel = self._relativize(output_path)
        timestamp = datetime.now(timezone.utc).isoformat()
        row = (
            f"| {timestamp} | {phase_id} | {_clean_cell(rel)} "
            f"| {_clean_cell(run_id)} | {_clean_cell(note)} |\n"
        )
        if not self._path.exists():
            self._path.write_text(_HEADER, encoding="utf-8")
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(row)
        logger.info("Recorded phase %d output %s in version log", phase_id, rel)

    def current(self, phase_id: int) -> str | None:
        """Workspace-relative output path from the LAST row for ``phase_id``."""
        result: str | None = None
        for line in self.read().splitlines():
            parsed = self._parse_row(line)
            if parsed is not None and parsed[0] == phase_id:
                result = parsed[1]
        return result

    def read(self) -> str:
        """Raw markdown contents; empty string if the log file is missing."""
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8")

    def _relativize(self, output_path: str) -> str:
        path = Path(output_path)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(self._project.root)
            except ValueError:
                logger.warning(
                    "Output path %s is outside the workspace %s; storing as given",
                    output_path,
                    self._project.root,
                )
        return path.as_posix()

    @staticmethod
    def _parse_row(line: str) -> tuple[int, str] | None:
        """Return ``(phase_id, output)`` for a data row, else None.

        Tolerates hand-edited files: header, separator, and malformed rows
        are skipped silently.
        """
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            return None
        cells = [cell.strip() for cell in stripped[1:-1].split("|")]
        if len(cells) < 3:
            return None
        try:
            phase_id = int(cells[1])
        except ValueError:
            return None  # header, separator, or hand-edited junk
        output = cells[2]
        if not output:
            return None
        return phase_id, output

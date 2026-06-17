"""Checkpoint persistence backends (S14.6).

Two storage modes:

- ``"filesystem"``: one JSON file per version named
  ``{run_id}_{version:05d}.checkpoint.json`` inside ``directory``.
- ``"sqlite"``: a single ``anvesha_checkpoints.sqlite`` file in ``directory``
  with table ``checkpoints(run_id, version, payload, created_at)``; writes are
  atomic via sqlite transactions (safer for "agent" granularity).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from anvesha.core.exceptions import CheckpointError, ConfigError

logger = logging.getLogger(__name__)

_DB_FILENAME = "anvesha_checkpoints.sqlite"
_FILE_SUFFIX = ".checkpoint.json"


class StateSerializer:
    """Reads and writes versioned checkpoint payloads (opaque dicts)."""

    def __init__(self, storage: str, directory: Path) -> None:
        if storage not in ("filesystem", "sqlite"):
            raise ConfigError(f"Unknown checkpoint storage: {storage!r}")
        self.storage = storage
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if storage == "sqlite":
            with self._connect() as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS checkpoints ("
                    "run_id TEXT, version INTEGER, payload TEXT, created_at TEXT, "
                    "PRIMARY KEY (run_id, version))"
                )

    # -- public API ---------------------------------------------------------

    def save(self, run_id: str, version: int, payload: dict) -> None:
        text = json.dumps(payload)
        if self.storage == "filesystem":
            path = self._file_path(run_id, version)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, path)
        else:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO checkpoints "
                    "(run_id, version, payload, created_at) VALUES (?, ?, ?, ?)",
                    (run_id, version, text, datetime.now(timezone.utc).isoformat()),
                )

    def load(self, run_id: str, version: int | None = None) -> dict:
        """Return the payload for ``version`` (latest when None).

        Raises CheckpointError when no matching checkpoint exists.
        """
        if version is None:
            versions = self.list_versions(run_id)
            if not versions:
                raise CheckpointError(f"No checkpoint found for run {run_id!r}")
            version = versions[-1]
        if self.storage == "filesystem":
            path = self._file_path(run_id, version)
            if not path.exists():
                raise CheckpointError(
                    f"Checkpoint version {version} not found for run {run_id!r}"
                )
            return json.loads(path.read_text(encoding="utf-8"))
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM checkpoints WHERE run_id = ? AND version = ?",
                (run_id, version),
            ).fetchone()
        if row is None:
            raise CheckpointError(
                f"Checkpoint version {version} not found for run {run_id!r}"
            )
        return json.loads(row[0])

    def list_versions(self, run_id: str) -> list[int]:
        if self.storage == "filesystem":
            versions = []
            for path in self.directory.glob(f"{run_id}_*{_FILE_SUFFIX}"):
                stem = path.name[: -len(_FILE_SUFFIX)]
                prefix, _, ver = stem.rpartition("_")
                if prefix == run_id and ver.isdigit():
                    versions.append(int(ver))
            return sorted(versions)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version FROM checkpoints WHERE run_id = ? ORDER BY version",
                (run_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def prune(self, run_id: str, keep: int = 5) -> None:
        """Keep only the ``keep`` most recent versions (S14.6 retention)."""
        stale = self.list_versions(run_id)[:-keep] if keep > 0 else self.list_versions(run_id)
        if not stale:
            return
        if self.storage == "filesystem":
            for version in stale:
                self._file_path(run_id, version).unlink(missing_ok=True)
        else:
            with self._connect() as conn:
                conn.executemany(
                    "DELETE FROM checkpoints WHERE run_id = ? AND version = ?",
                    [(run_id, version) for version in stale],
                )
        logger.debug("Pruned %d checkpoint(s) for run %s", len(stale), run_id)

    # -- internals ----------------------------------------------------------

    def _file_path(self, run_id: str, version: int) -> Path:
        return self.directory / f"{run_id}_{version:05d}{_FILE_SUFFIX}"

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.directory / _DB_FILENAME)

"""Tests for the anvesha CLI (ANVESHA.md S9)."""

from __future__ import annotations

import pytest

from anvesha.cli import main


def test_init_creates_workspace(tmp_path, capsys):
    rc = main(["init", "--name", "demo", "--path", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / ".anvesha" / "config.yaml").is_file()
    assert (tmp_path / "phases").is_dir()
    assert (tmp_path / "_VERSION_LOG.md").is_file()
    assert (tmp_path / ".gitignore").is_file()
    out = capsys.readouterr().out
    assert "demo" in out


def test_double_init_errors(tmp_path):
    assert main(["init", "--name", "demo", "--path", str(tmp_path)]) == 0
    rc = main(["init", "--name", "demo", "--path", str(tmp_path)])
    assert rc == 1  # WorkspaceError -> exit 1


def test_status_fresh_workspace(tmp_path, monkeypatch, capsys):
    main(["init", "--name", "demo", "--path", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    rc = main(["status"])
    assert rc == 0
    assert "no phases recorded" in capsys.readouterr().out


def test_status_with_recorded_phase(tmp_path, monkeypatch, capsys):
    from anvesha.workspace import ResearchProject, VersionLog

    main(["init", "--name", "demo", "--path", str(tmp_path)])
    VersionLog(ResearchProject(tmp_path)).record(3, "phases/03_research_gaps.md")
    monkeypatch.chdir(tmp_path)
    rc = main(["status"])
    assert rc == 0
    assert "03_research_gaps.md" in capsys.readouterr().out


def test_run_unimplemented_phase_exits_2(tmp_path, monkeypatch, capsys):
    main(["init", "--name", "demo", "--path", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "--phase", "3"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not implemented" in err
    assert "Research Gaps" in err


def test_resume_unimplemented_phase_exits_2(tmp_path, monkeypatch):
    main(["init", "--name", "demo", "--path", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    assert main(["resume", "--phase", "3"]) == 2


def test_run_outside_workspace_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = main(["run", "--phase", "3"])
    assert rc == 1  # WorkspaceError -> exit 1


def test_logs_missing_exits_1(tmp_path, monkeypatch, capsys):
    main(["init", "--name", "demo", "--path", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    rc = main(["logs", "--phase", "3"])
    assert rc == 1
    assert "No log file" in capsys.readouterr().err


def test_logs_prints_existing(tmp_path, monkeypatch, capsys):
    main(["init", "--name", "demo", "--path", str(tmp_path)])
    log = tmp_path / ".anvesha" / "logs" / "phase_03.log"
    log.write_text("hello phase 3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    rc = main(["logs", "--phase", "3"])
    assert rc == 0
    assert "hello phase 3" in capsys.readouterr().out


def test_unimplemented_phase_does_not_require_backend(tmp_path, monkeypatch):
    """A run on an unimplemented phase must not touch backend config (no
    default_backend is set here, yet the command returns 2 cleanly)."""
    main(["init", "--name", "demo", "--path", str(tmp_path)])
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--phase", "5"]) == 2

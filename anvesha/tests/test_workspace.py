"""Tests for anvesha.workspace - project, version_log, phase_io (offline, tmp_path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from anvesha.core.exceptions import WorkspaceError
from anvesha.workspace import (
    ResearchProject,
    VersionLog,
    exists,
    init_workspace,
    list_versions,
    next_versioned_path,
    read_text,
    write_phase_output,
)
from anvesha.workspace.schemas import PhaseFilePaths, ProjectMeta, WorkspaceConfig


@pytest.fixture()
def project(tmp_path: Path) -> ResearchProject:
    return init_workspace(tmp_path, "test-project")


# --- init_workspace / ResearchProject ---------------------------------------


def test_init_workspace_creates_skeleton(tmp_path: Path) -> None:
    project = init_workspace(tmp_path, "my-research")
    root = tmp_path.resolve()
    assert project.root == root

    for directory in (
        root / ".anvesha" / "checkpoints",
        root / ".anvesha" / "logs",
        root / "phases",
        root / "data",
        root / "refs" / "pdfs",
        root / "runs",
    ):
        assert directory.is_dir(), directory

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "# my-research" in readme
    assert "<!-- topic statement -->" in readme

    version_log = (root / "_VERSION_LOG.md").read_text(encoding="utf-8")
    assert "| timestamp | phase | output | run_id | note |" in version_log
    assert "|---|---|---|---|---|" in version_log

    config = (root / ".anvesha" / "config.yaml").read_text(encoding="utf-8")
    assert 'name: "my-research"' in config
    assert "code_repo: null" in config
    assert "ANVESHA.md S8" in config

    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert ".anvesha/" in gitignore
    assert "data/" in gitignore


def test_init_workspace_twice_raises(tmp_path: Path) -> None:
    init_workspace(tmp_path, "p")
    with pytest.raises(WorkspaceError):
        init_workspace(tmp_path, "p")


def test_init_workspace_preserves_existing_readme(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("existing\n", encoding="utf-8")
    init_workspace(tmp_path, "p")
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "existing\n"


def test_validate_and_properties(project: ResearchProject) -> None:
    project.validate()  # does not raise
    assert project.config_path == project.root / ".anvesha" / "config.yaml"
    assert project.phases_dir == project.root / "phases"
    assert project.checkpoints_dir == project.root / ".anvesha" / "checkpoints"
    assert project.logs_dir == project.root / ".anvesha" / "logs"
    assert project.version_log_path == project.root / "_VERSION_LOG.md"
    assert project.resolve("phases/x.md") == project.root / "phases/x.md"


def test_validate_raises_outside_workspace(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        ResearchProject(tmp_path / "nowhere").validate()


def test_find_walks_up_from_nested_dir(project: ResearchProject) -> None:
    nested = project.root / "phases" / "deep" / "deeper"
    nested.mkdir(parents=True)
    found = ResearchProject.find(nested)
    assert found.root == project.root


def test_find_fails_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path / "not_a_workspace"
    outside.mkdir()
    with pytest.raises(WorkspaceError):
        ResearchProject.find(outside)


# --- phase_io ----------------------------------------------------------------


def test_read_text_and_exists(project: ResearchProject) -> None:
    path = project.phases_dir / "02_literature_survey.md"
    path.write_text("survey content", encoding="utf-8")
    assert exists(project, "phases/02_literature_survey.md")
    assert read_text(project, "phases/02_literature_survey.md") == "survey content"
    assert not exists(project, "phases/missing.md")


def test_read_text_missing_raises(project: ResearchProject) -> None:
    with pytest.raises(WorkspaceError):
        read_text(project, "phases/missing.md")


@pytest.mark.parametrize("relpath", ["phases/03_research_gaps.md", "phases/08_results.txt"])
def test_versioning_sequence(project: ResearchProject, relpath: str) -> None:
    base = project.resolve(relpath)
    stem, suffix = base.stem, base.suffix

    assert next_versioned_path(project, relpath) == base
    p1 = write_phase_output(project, relpath, "run 1")
    assert p1 == base

    assert next_versioned_path(project, relpath).name == f"{stem}_v2{suffix}"
    p2 = write_phase_output(project, relpath, "run 2")
    assert p2.name == f"{stem}_v2{suffix}"

    p3 = write_phase_output(project, relpath, "run 3")
    assert p3.name == f"{stem}_v3{suffix}"

    assert p1.read_text(encoding="utf-8") == "run 1"
    assert p3.read_text(encoding="utf-8") == "run 3"
    assert list_versions(project, relpath) == [base, p2, p3]


def test_next_versioned_path_picks_max_plus_one(project: ResearchProject) -> None:
    relpath = "phases/05_ideas.md"
    project.resolve(relpath).write_text("base", encoding="utf-8")
    project.resolve("phases/05_ideas_v4.md").write_text("v4", encoding="utf-8")
    assert next_versioned_path(project, relpath).name == "05_ideas_v5.md"
    # _v4 exists but _v2/_v3 do not: list returns only what exists, base first
    versions = list_versions(project, relpath)
    assert [p.name for p in versions] == ["05_ideas.md", "05_ideas_v4.md"]


def test_write_phase_output_creates_parents(project: ResearchProject) -> None:
    path = write_phase_output(project, "phases/sub/dir/note.md", "hello")
    assert path == project.root / "phases/sub/dir/note.md"
    assert path.read_text(encoding="utf-8") == "hello"


def test_list_versions_empty_when_nothing_exists(project: ResearchProject) -> None:
    assert list_versions(project, "phases/nope.md") == []


# --- VersionLog --------------------------------------------------------------


def test_version_log_round_trip(project: ResearchProject) -> None:
    log = VersionLog(project)
    log.record(3, "phases/03_research_gaps.md", run_id="abc123")
    log.record(2, "phases/02_literature_survey.md")
    log.record(3, "phases/03_research_gaps_v2.md", run_id="def456", note="re-run")

    assert log.current(3) == "phases/03_research_gaps_v2.md"
    assert log.current(2) == "phases/02_literature_survey.md"
    assert log.current(7) is None

    raw = log.read()
    assert "| timestamp | phase | output | run_id | note |" in raw
    assert "abc123" in raw
    assert "re-run" in raw


def test_version_log_relativizes_absolute_paths(project: ResearchProject) -> None:
    log = VersionLog(project)
    absolute = project.root / "phases" / "04_base_paper.md"
    log.record(4, str(absolute))
    assert log.current(4) == "phases/04_base_paper.md"


def test_version_log_creates_file_if_missing(project: ResearchProject) -> None:
    project.version_log_path.unlink()
    log = VersionLog(project)
    assert log.read() == ""
    assert log.current(1) is None
    log.record(1, "filtered_literature_v1_x/01_filtered_literature.md", run_id="r1")
    assert log.current(1) == "filtered_literature_v1_x/01_filtered_literature.md"
    assert log.read().startswith("# Version Log")


def test_version_log_tolerates_hand_edited_rows(project: ResearchProject) -> None:
    with project.version_log_path.open("a", encoding="utf-8") as fh:
        fh.write("some prose the researcher added\n")
        fh.write("| broken row |\n")
        fh.write("| ts | not-a-number | phases/x.md | r | n |\n")
        fh.write("| 2026-06-12T00:00:00+00:00 | 5 | phases/05_ideas.md | abc |  |\n")
    log = VersionLog(project)
    assert log.current(5) == "phases/05_ideas.md"
    assert log.current(99) is None


# --- schemas -----------------------------------------------------------------


def test_schemas_construct(tmp_path: Path) -> None:
    meta = ProjectMeta(name="p", root=tmp_path)
    assert meta.root == tmp_path
    cfg = WorkspaceConfig()
    assert cfg.code_repo is None
    paths = PhaseFilePaths(phase_id=3, inputs=["phases/02_literature_survey.md"], output="phases/03_research_gaps.md")
    assert paths.phase_id == 3

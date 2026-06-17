"""End-to-end tests for the Phase 1 pipeline wiring (orchestrator + entry point).

Offline only: a real tmp ``ResearchProject`` workspace (``init_workspace``), a
fake ``LLMClient`` (scripted queries / relevance assessments / summaries) and
fake ``MCPClient`` instances feeding a real :class:`ToolRouter` scripted search
results + PDF bytes. No network, no MCP servers, no GPU.

Asserts the S4.1 output-directory name, that ``01_filtered_literature.md`` is
present and passes ``validate_document`` (S4.5), the version log records the
index path (Stage 11), the PRISMA funnel is monotonic, the empty-result path
(EH-7) does not crash, EH-3 (no search source) is fatal, and a second run
produces a ``_v2`` directory (S4.1 / EH-8).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.exceptions import AnveshaError, PhaseValidationError
from anvesha.core.phase import PhaseSpec
from anvesha.phases.phase01_filter.pipeline import FilterLiteraturePipeline
from anvesha.phases.phase01_filter.stages.assemble import validate_document
from anvesha.workspace.project import init_workspace
from anvesha.workspace.version_log import VersionLog


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeLLM(LLMClient):
    """Scripted LLM.

    ``structured_run`` dispatches by the JSON schema: a ``queries`` schema gets
    the scripted query list; a ``domain_matches`` schema gets the per-domain
    assessment scripted for the paper title in the prompt (keyed by a marker
    substring), falling back to a deterministic default. ``run`` returns a fixed
    summary for every call.
    """

    name = "fake"

    def __init__(
        self,
        queries: list[str],
        assessments_by_marker: dict[str, dict[str, Any]],
        default_assessment: dict[str, Any],
        summary: str = "x " * 70,
    ) -> None:
        self._queries = queries
        self._assessments = assessments_by_marker
        self._default_assessment = default_assessment
        self._summary = summary.strip()
        self.run_calls: list[str] = []
        self.structured_calls: list[str] = []

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        self.run_calls.append(prompt)
        return LLMResponse(content=self._summary)

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> dict:
        self.structured_calls.append(prompt)
        props = json_schema.get("properties", {})
        if "queries" in props:
            return {"queries": list(self._queries)}
        if "domain_matches" in props:
            for marker, assessment in self._assessments.items():
                if marker in prompt:
                    return assessment
            return self._default_assessment
        raise AssertionError(f"unexpected schema: {sorted(props)}")


class FakeMCP(MCPClient):
    """Scripted MCP client: advertises ``tool_names`` and replays ``responses``."""

    def __init__(
        self,
        tool_names: Sequence[str],
        responses: dict[str, str] | None = None,
    ) -> None:
        self._tool_names = list(tool_names)
        self._responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        return [MCPToolDef(name=n, description="", input_schema={}) for n in self._tool_names]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self._responses.get(name, "[]")


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
def _make_project(tmp_path: Path):
    return init_workspace(tmp_path / "ws", "Test Project")


def _spec(**overrides: Any) -> PhaseSpec:
    filters = overrides.pop(
        "filters",
        {
            "conferences": ["NeurIPS", "ICML"],
            "domains": ["LLM Agents", "Retrieval-Augmented Generation"],
            "years": {"start": 2023, "end": 2025},
        },
    )
    options = overrides.pop(
        "options",
        {
            "max_papers": 40,
            "relevance_threshold": 0.50,
            "pdf_download": True,
            "pdf_max_retries": 1,
            "summary_max_retries": 1,
        },
    )
    return PhaseSpec(
        phase_id=1,
        name="Filter Literature",
        inputs=[],
        optional_inputs=["README.md"],
        output="filtered_literature/01_filtered_literature.md",
        tools_permitted=[
            "papersflow.*",
            "paper-search.search_papers",
            "paper-search.download_with_fallback",
        ],
        filters=filters,
        options=options,
    )


# Two papers, each strongly on a configured domain, with verbatim evidence.
_PAPER_A = {
    "title": "LLM Agents for Tool Use",
    "abstract": "We propose LLM Agents for tool use in retrieval settings.",
    "authors": ["A. One", "B. Two"],
    "venue": "Advances in Neural Information Processing Systems",
    "year": 2024,
    "doi": "10.1/neurips.a",
    "language": "en",
}
_PAPER_B = {
    "title": "Retrieval-Augmented Generation at Scale",
    "abstract": "A study of Retrieval-Augmented Generation across large corpora.",
    "authors": ["C. Three"],
    "venue": "International Conference on Machine Learning",
    "year": 2025,
    "arxiv_id": "2501.00002",
    "language": "en",
    "code": "https://github.com/example/rag",
}

_ASSESS_A = {
    "domain_matches": [
        {
            "domain": "LLM Agents",
            "match_level": "central",
            "evidence_quote": "LLM Agents for tool use",
            "evidence_location": "abstract",
        },
        {"domain": "Retrieval-Augmented Generation", "match_level": "absent"},
    ],
    "topic_alignment": "no_topic_statement",
}
_ASSESS_B = {
    "domain_matches": [
        {"domain": "LLM Agents", "match_level": "absent"},
        {
            "domain": "Retrieval-Augmented Generation",
            "match_level": "central",
            "evidence_quote": "Retrieval-Augmented Generation across large corpora",
            "evidence_location": "abstract",
        },
    ],
    "topic_alignment": "no_topic_statement",
}
_ASSESS_ABSENT = {
    "domain_matches": [
        {"domain": "LLM Agents", "match_level": "absent"},
        {"domain": "Retrieval-Augmented Generation", "match_level": "absent"},
    ],
    "topic_alignment": "no_topic_statement",
}


def _search_response(papers: list[dict]) -> str:
    return json.dumps({"results": papers})


def _pdf_response() -> str:
    return json.dumps({"content": base64.b64encode(b"%PDF-1.4 fake").decode("ascii")})


def _build_llm(summary: str = "x " * 70) -> FakeLLM:
    return FakeLLM(
        queries=["llm agents tool use", "retrieval augmented generation"],
        assessments_by_marker={
            _PAPER_A["title"]: _ASSESS_A,
            _PAPER_B["title"]: _ASSESS_B,
        },
        default_assessment=_ASSESS_ABSENT,
        summary=summary,
    )


def _build_mcps(papers: list[dict], with_pdf: bool = True) -> list[FakeMCP]:
    papersflow = FakeMCP(
        ["papersflow.search"],
        {"papersflow.search": _search_response(papers)},
    )
    ps_responses = {"paper-search.search_papers": _search_response([])}
    ps_tools = ["paper-search.search_papers"]
    if with_pdf:
        ps_tools.append("paper-search.download_with_fallback")
        ps_responses["paper-search.download_with_fallback"] = _pdf_response()
    paper_search = FakeMCP(ps_tools, ps_responses)
    return [papersflow, paper_search]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_run_creates_versioned_dir_and_valid_index(tmp_path):
    project = _make_project(tmp_path)
    llm = _build_llm()
    mcps = _build_mcps([_PAPER_A, _PAPER_B])
    pipeline = FilterLiteraturePipeline(project, VersionLog(project), llm, _spec())

    written = pipeline.run(mcps=mcps)

    # S4.1 directory name: filtered_literature_v1_{domain_slug}_{years}.
    expected_dir = "filtered_literature_v1_llm_agents_2023_2025"
    assert written.parent.name == expected_dir
    assert written.name == "01_filtered_literature.md"
    assert written.is_absolute() and written.is_file()

    content = written.read_text(encoding="utf-8")
    assert validate_document(content) == []
    # Both papers retained.
    assert content.count("paper_id: paper_") == 2
    assert "papers_total: 2" in content


def test_version_log_records_index_path(tmp_path):
    project = _make_project(tmp_path)
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec()
    )
    written = pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))

    rel = written.relative_to(project.root).as_posix()
    log = VersionLog(project)
    assert log.current(1) == rel
    assert rel == "filtered_literature_v1_llm_agents_2023_2025/01_filtered_literature.md"


def test_funnel_is_monotonic_and_consistent(tmp_path):
    project = _make_project(tmp_path)
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec()
    )
    written = pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
    content = written.read_text(encoding="utf-8")

    import yaml

    manifest = yaml.safe_load(content.split("---", 2)[1])
    counts = manifest["counts"]
    assert (
        counts["identified"]
        >= counts["after_deduplication"]
        >= counts["after_hard_filter"]
        >= counts["screened"]
        >= counts["retained"]
    )
    # 2 queries x 1 available source (papersflow) x 2 papers each = 4 raw,
    # collapsed to 2 unique after dedup.
    assert counts["identified"] == 4
    assert counts["after_deduplication"] == 2
    assert counts["retained"] == 2
    assert counts["pdfs_downloaded"] == 2
    assert counts["pdfs_failed"] == 0
    # Paper B has a github code link; paper A does not.
    assert counts["with_code"] == 1


def test_pdf_files_written_and_paths_resolve(tmp_path):
    project = _make_project(tmp_path)
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec()
    )
    written = pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
    run_dir = written.parent

    # Every non-null pdf_path points to an existing file in pdfs/ (S4.5 bullet 5).
    import yaml

    blocks = written.read_text(encoding="utf-8").split("---")
    pdf_paths = []
    for raw in blocks:
        raw = raw.strip()
        if "paper_id:" not in raw:
            continue
        rec = yaml.safe_load(raw)
        if rec.get("pdf_path"):
            pdf_paths.append(rec["pdf_path"])
    assert pdf_paths  # at least one downloaded
    for rel in pdf_paths:
        assert (run_dir / rel).is_file()


def test_rerun_produces_v2_directory(tmp_path):
    project = _make_project(tmp_path)
    spec = _spec()

    first = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), spec
    ).run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
    assert first.parent.name == "filtered_literature_v1_llm_agents_2023_2025"

    second = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), spec
    ).run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
    assert second.parent.name == "filtered_literature_v2_llm_agents_2023_2025"
    assert first.parent != second.parent
    # Both directories exist on disk.
    assert first.is_file() and second.is_file()


def test_overwrite_reuses_v1_directory(tmp_path):
    project = _make_project(tmp_path)
    options = {
        "pdf_download": False,
        "pdf_max_retries": 1,
        "summary_max_retries": 1,
        "overwrite": True,
    }
    spec = _spec(options=options)

    first = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), spec
    ).run(mcps=_build_mcps([_PAPER_A, _PAPER_B], with_pdf=False))
    second = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), spec
    ).run(mcps=_build_mcps([_PAPER_A, _PAPER_B], with_pdf=False))
    assert first.parent == second.parent
    assert first.parent.name == "filtered_literature_v1_llm_agents_2023_2025"


def test_empty_result_does_not_crash(tmp_path):
    project = _make_project(tmp_path)
    # All candidates classified absent -> all below the 0.50 threshold -> EH-7.
    llm = FakeLLM(
        queries=["llm agents"],
        assessments_by_marker={},
        default_assessment=_ASSESS_ABSENT,
    )
    pipeline = FilterLiteraturePipeline(project, VersionLog(project), llm, _spec())
    written = pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))

    content = written.read_text(encoding="utf-8")
    assert "papers_total: 0" in content
    assert "## Notes" in content
    assert "Recommendation: relax" in content
    assert validate_document(content) == []
    # No paper records.
    assert "paper_id: paper_" not in content


def test_empty_result_after_hard_filter_recommends_conferences(tmp_path):
    project = _make_project(tmp_path)
    # A paper whose venue is not in the configured conferences and not a preprint
    # -> removed at the hard filter -> after_hard_filter == 0 -> recommend conferences.
    off_topic = {
        "title": "Some Unrelated Database Paper",
        "abstract": "A paper about databases.",
        "venue": "VLDB",
        "year": 2024,
        "language": "en",
        "doi": "10.1/vldb.x",
    }
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec()
    )
    written = pipeline.run(mcps=_build_mcps([off_topic]))
    content = written.read_text(encoding="utf-8")
    assert "papers_total: 0" in content
    assert "Recommendation: relax conferences." in content


def test_no_search_source_is_fatal(tmp_path):
    project = _make_project(tmp_path)
    # MCP client advertises only a PDF tool, no search tool -> EH-3 fatal.
    only_pdf = FakeMCP(
        ["paper-search.download_with_fallback"],
        {"paper-search.download_with_fallback": _pdf_response()},
    )
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec()
    )
    with pytest.raises(AnveshaError):
        pipeline.run(mcps=[only_pdf])


def test_degraded_single_source_still_runs(tmp_path):
    project = _make_project(tmp_path)
    # Only PapersFlow available (paper-search absent except the PDF tool); search
    # still runs against the one source (EH-2 degraded coverage).
    papersflow = FakeMCP(
        ["papersflow.search"],
        {"papersflow.search": _search_response([_PAPER_A, _PAPER_B])},
    )
    pdf_only = FakeMCP(
        ["paper-search.download_with_fallback"],
        {"paper-search.download_with_fallback": _pdf_response()},
    )
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec()
    )
    written = pipeline.run(mcps=[papersflow, pdf_only])
    content = written.read_text(encoding="utf-8")
    assert validate_document(content) == []
    assert "papers_total: 2" in content


def test_topic_statement_read_from_readme(tmp_path):
    project = _make_project(tmp_path)
    # Overwrite the README with a real topic statement (not the placeholder).
    (project.root / "README.md").write_text(
        "# Topic\n\nWe study retrieval-augmented LLM agents.\n", encoding="utf-8"
    )
    llm = _build_llm()
    pipeline = FilterLiteraturePipeline(project, VersionLog(project), llm, _spec())
    pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))

    # The topic statement must reach the relevance prompts (Stage 5).
    assert any(
        "We study retrieval-augmented LLM agents." in prompt
        for prompt in llm.structured_calls
    )


def test_placeholder_readme_is_not_a_topic_statement(tmp_path):
    project = _make_project(tmp_path)
    # The init skeleton README contains only the placeholder; topic must be None,
    # so relevance prompts say "none provided".
    llm = _build_llm()
    pipeline = FilterLiteraturePipeline(project, VersionLog(project), llm, _spec())
    pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
    assert any("none provided" in prompt for prompt in llm.structured_calls)


def test_invalid_config_is_fatal_before_output(tmp_path):
    project = _make_project(tmp_path)
    bad = _spec(
        filters={
            "conferences": [],
            "domains": ["X"],
            "years": {"start": 2025, "end": 2023},  # start > end -> EH-1
        }
    )
    pipeline = FilterLiteraturePipeline(project, VersionLog(project), _build_llm(), bad)
    with pytest.raises(Exception):  # pydantic ValidationError (EH-1)
        pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
    # No output directory should have been created.
    assert not any(p.name.startswith("filtered_literature") for p in project.root.iterdir())


def test_pdf_download_disabled_marks_skipped(tmp_path):
    project = _make_project(tmp_path)
    options = {
        "pdf_download": False,
        "pdf_max_retries": 1,
        "summary_max_retries": 1,
        "relevance_threshold": 0.50,
    }
    pipeline = FilterLiteraturePipeline(
        project, VersionLog(project), _build_llm(), _spec(options=options)
    )
    written = pipeline.run(mcps=_build_mcps([_PAPER_A, _PAPER_B], with_pdf=False))
    content = written.read_text(encoding="utf-8")
    assert validate_document(content) == []
    assert content.count("pdf_status: skipped") == 2
    assert "pdfs_downloaded: 0" in content
    # No pdfs/ files written.
    assert not (written.parent / "pdfs").exists() or not list(
        (written.parent / "pdfs").iterdir()
    )


def test_determinism_same_inputs_same_output(tmp_path):
    # Two independent runs with identical inputs differ only in generated_at;
    # strip that line and the rest of the document must be byte-identical.
    def _run(root_name: str) -> str:
        project = init_workspace(tmp_path / root_name, "Test")
        written = FilterLiteraturePipeline(
            project, VersionLog(project), _build_llm(), _spec()
        ).run(mcps=_build_mcps([_PAPER_A, _PAPER_B]))
        return written.read_text(encoding="utf-8")

    def _strip_ts(doc: str) -> str:
        return "\n".join(
            line for line in doc.splitlines() if not line.startswith("generated_at:")
        )

    assert _strip_ts(_run("a")) == _strip_ts(_run("b"))

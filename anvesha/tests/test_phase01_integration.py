"""Phase 1 (Filter Literature) integration + determinism + registration tests.

Scope (this file's assignment): exercise the *whole* Phase 1 pipeline end to end
over a real tmp ``ResearchProject`` workspace, driven by injected fakes only --
no network, no MCP servers, no GPU. It complements ``test_phase01_pipeline.py``
(which checks the wiring/funnel/EH-paths) by asserting the *consumer contract*:

  1. HAPPY PATH -- the produced ``01_filtered_literature.md`` parses under an
     INDEPENDENT implementation of the S4.4 grammar (a parser written here from
     the spec, deliberately NOT the producer's own splitter), the S4.5 exit
     criteria hold, and the manifest counts are internally consistent.
  2. DETERMINISM (NFR-1) -- two runs with byte-identical inputs and identical
     scripted fake responses, into two separate workspaces, produce documents
     that are byte-identical EXCEPT the single ``generated_at:`` manifest line.
  3. EH-7 -- a filter/scoring configuration that retains zero papers still
     writes a valid manifest-only document with a ``## Notes`` section and
     ``papers_total: 0``, and does NOT raise.
  4. REGISTRATION -- ``anvesha.phases.PHASE_REGISTRY[1] is
     FilterLiteraturePipeline``, ``import anvesha.cli`` works (no circular
     import), and unimplemented phases stay absent from the registry.

Determinism caveat (documented, deliberate): the run manifest records a wall
clock ``generated_at`` (S4.3, NFR-3) captured once per run. Two independent runs
happen at different instants, so that one line legitimately differs; NFR-1's
"byte-identical" guarantee is over the *content derived from inputs + tool
responses*, not over the timestamp. The determinism test therefore strips the
single ``generated_at:`` line from both documents before comparing -- everything
else (field order, ranking, scores, summaries, counts) must match to the byte.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.phase import PhaseSpec
from anvesha.phases.phase01_filter.pipeline import FilterLiteraturePipeline
from anvesha.phases.phase01_filter.stages.assemble import validate_document
from anvesha.workspace.project import init_workspace
from anvesha.workspace.version_log import VersionLog


# --------------------------------------------------------------------------- #
# Fakes (offline scripted LLM + MCP clients)
# --------------------------------------------------------------------------- #
class FakeLLM(LLMClient):
    """Scripted, deterministic LLM.

    ``structured_run`` dispatches by JSON schema: a ``queries`` schema returns
    the scripted query list; a ``domain_matches`` schema returns the per-paper
    assessment keyed by a title marker found in the prompt (default otherwise).
    ``run`` returns one fixed summary. No clock, no randomness -- identical
    prompts always map to identical replies, which is what makes the pipeline's
    determinism testable.
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

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        return LLMResponse(content=self._summary)

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> dict:
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
        return [
            MCPToolDef(name=n, description="", input_schema={})
            for n in self._tool_names
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self._responses.get(name, "[]")


# --------------------------------------------------------------------------- #
# Scripted fixtures (two clearly-relevant papers + their assessments)
# --------------------------------------------------------------------------- #
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
    return json.dumps(
        {"content": base64.b64encode(b"%PDF-1.4 fake").decode("ascii")}
    )


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


def _run_pipeline(
    project,
    papers: list[dict],
    *,
    llm: FakeLLM | None = None,
    spec: PhaseSpec | None = None,
    with_pdf: bool = True,
) -> Path:
    """Drive the full pipeline (Stages 0-11) and return the index path."""
    pipeline = FilterLiteraturePipeline(
        project,
        VersionLog(project),
        llm or _build_llm(),
        spec or _spec(),
    )
    return pipeline.run(mcps=_build_mcps(papers, with_pdf=with_pdf))


# --------------------------------------------------------------------------- #
# Independent S4.4 grammar parser (consumer contract; NOT the producer's code)
# --------------------------------------------------------------------------- #
def parse_s44(content: str) -> tuple[dict, list[tuple[dict, str]]]:
    """Parse ``01_filtered_literature.md`` per the S4.4 grammar from scratch.

    Grammar (spec S4.4)::

        FILE         := RUN_MANIFEST PAPER_RECORD*
        RUN_MANIFEST := "---" NL YAML_BLOCK "---" NL
        PAPER_RECORD := "---" NL YAML_BLOCK "---" NL "## LLM Summary" NL SUMMARY
        SUMMARY      := free text until the next line that is exactly "---" or EOF

    Returns ``(manifest, records)`` where ``manifest`` is the first YAML block
    parsed to a dict, and ``records`` is a list of ``(yaml_dict, summary_text)``
    tuples in file order.

    Disambiguation (S4.4): the FIRST YAML block is the run manifest -- it has
    ``document_type`` and no ``paper_id``; every later block is a paper record
    and has ``paper_id``. This parser is written independently of the producer's
    ``assemble._split_yaml_blocks`` so the test genuinely exercises the published
    consumer contract rather than the implementation's own round-trip.

    Raises ``AssertionError`` on any structural violation, so a malformed file
    fails the test loudly.
    """
    lines = content.splitlines()
    i = 0
    n = len(lines)

    def _read_yaml_block(start: int) -> tuple[dict, int]:
        # ``start`` must be a line that is exactly "---" (opening fence).
        assert lines[start] == "---", (
            f"expected opening '---' fence at line {start}, got {lines[start]!r}"
        )
        body: list[str] = []
        j = start + 1
        while j < n and lines[j] != "---":
            body.append(lines[j])
            j += 1
        assert j < n, "YAML block was not closed by a '---' fence before EOF"
        loaded = yaml.safe_load("\n".join(body))
        assert isinstance(loaded, dict), "YAML block did not parse to a mapping"
        return loaded, j + 1  # index just past the closing fence

    # 1) RUN_MANIFEST -- skip any leading blank lines, then the first block.
    while i < n and lines[i].strip() == "":
        i += 1
    assert i < n, "file is empty; no run manifest"
    manifest, i = _read_yaml_block(i)
    assert "document_type" in manifest, "first block lacks document_type"
    assert "paper_id" not in manifest, "first block unexpectedly has paper_id"

    # 2) PAPER_RECORD* -- each is a YAML block, then '## LLM Summary', then text.
    records: list[tuple[dict, str]] = []
    while i < n:
        # Tolerate blank separator lines between records.
        if lines[i].strip() == "":
            i += 1
            continue
        if lines[i] == "---":
            rec, i = _read_yaml_block(i)
            assert "paper_id" in rec, "paper record block lacks paper_id"
            # Skip blank lines up to the '## LLM Summary' heading.
            while i < n and lines[i].strip() == "":
                i += 1
            assert i < n and lines[i].strip() == "## LLM Summary", (
                "paper record is not followed by a '## LLM Summary' heading"
            )
            i += 1
            # Summary text runs until the next line that is exactly '---' or EOF.
            summary_lines: list[str] = []
            while i < n and lines[i] != "---":
                summary_lines.append(lines[i])
                i += 1
            records.append((rec, "\n".join(summary_lines).strip()))
        else:
            # Any other non-blank content after the manifest with no records is
            # the EH-7 '## Notes' section; the caller handles that separately.
            break
    return manifest, records


# --------------------------------------------------------------------------- #
# 1. End-to-end happy path -- parses under the independent S4.4 grammar
# --------------------------------------------------------------------------- #
def test_happy_path_parses_under_s44_grammar(tmp_path):
    project = init_workspace(tmp_path / "ws", "Integration")
    written = _run_pipeline(project, [_PAPER_A, _PAPER_B])

    assert written.is_absolute() and written.is_file()
    assert written.name == "01_filtered_literature.md"
    content = written.read_text(encoding="utf-8")

    manifest, records = parse_s44(content)

    # Manifest is the first block: document_type set, no paper_id (S4.4).
    assert manifest["document_type"] == "filtered_literature_index"
    assert manifest["schema_version"] == "2.0"
    assert manifest["generator"] == "anvesha.phase01_filter"
    assert "paper_id" not in manifest

    # Two relevant papers retained -> two records, each with a paper_id and a
    # non-empty summary body after '## LLM Summary'.
    assert len(records) == 2
    assert manifest["papers_total"] == len(records) == 2
    for rec, summary in records:
        assert rec["paper_id"].startswith("paper_")
        assert summary  # '## LLM Summary' body present and non-empty

    # The producer's own validator must also accept the document (S4.5).
    assert validate_document(content) == []


def test_happy_path_exit_criteria_and_count_consistency(tmp_path):
    project = init_workspace(tmp_path / "ws", "Integration")
    written = _run_pipeline(project, [_PAPER_A, _PAPER_B])
    content = written.read_text(encoding="utf-8")
    run_dir = written.parent

    manifest, records = parse_s44(content)
    counts = manifest["counts"]

    # S4.5: papers_total == number of paper records.
    assert manifest["papers_total"] == len(records)

    # S4.5: counts monotonic identified >= dedup >= hard >= screened >= retained.
    assert (
        counts["identified"]
        >= counts["after_deduplication"]
        >= counts["after_hard_filter"]
        >= counts["screened"]
        >= counts["retained"]
    )
    assert counts["retained"] == len(records)

    # S4.5: rank is the contiguous ascending sequence 1..retained, in file order.
    assert [rec["rank"] for rec, _ in records] == list(range(1, len(records) + 1))

    # S4.5: every non-null pdf_path points to an existing file under pdfs/.
    pdf_seen = False
    for rec, _ in records:
        # S4.3: every field present (nulls allowed).
        for field in (
            "paper_id", "title", "authors", "conference", "year", "domain",
            "paper_url", "pdf_path", "pdf_status", "code_url", "git_exists",
            "relevance_score", "rank", "doi", "arxiv_id", "source",
        ):
            assert field in rec, f"record missing required field {field!r}"
        # S8 invariant: git_exists == (code_url != null).
        assert bool(rec["git_exists"]) == (rec["code_url"] is not None)
        if rec["pdf_path"] is not None:
            pdf_seen = True
            assert (run_dir / rec["pdf_path"]).is_file()
            assert rec["pdf_status"] == "ok"
        else:
            assert rec["pdf_status"] in {"failed", "skipped"}
    assert pdf_seen  # PDFs were downloaded in the happy path

    # Funnel accounting is consistent with the scripted inputs:
    # 2 queries x 1 live source (papersflow) x 2 papers = 4 identified,
    # collapsed to 2 unique, both pass the gate, both scored, both retained.
    assert counts["identified"] == 4
    assert counts["after_deduplication"] == 2
    assert counts["retained"] == 2
    assert counts["pdfs_downloaded"] == 2
    assert counts["pdfs_failed"] == 0
    # Exactly paper B carries a github code link.
    assert counts["with_code"] == 1
    assert sum(1 for rec, _ in records if rec["git_exists"]) == counts["with_code"]


# --------------------------------------------------------------------------- #
# 2. Determinism (NFR-1)
# --------------------------------------------------------------------------- #
def _strip_generated_at(doc: str) -> str:
    """Drop the single wall-clock ``generated_at:`` manifest line (see caveat)."""
    return "\n".join(
        line for line in doc.splitlines() if not line.startswith("generated_at:")
    )


def test_determinism_byte_identical_except_timestamp(tmp_path):
    # Two fully independent runs: separate workspaces, freshly built (but
    # identically scripted) fakes, identical spec/inputs.
    def _one(root: str) -> str:
        project = init_workspace(tmp_path / root, "Determinism")
        written = _run_pipeline(project, [_PAPER_A, _PAPER_B])
        return written.read_text(encoding="utf-8")

    doc_a = _one("run_a")
    doc_b = _one("run_b")

    # Exactly one line differs and it is the generated_at line: that line is the
    # ONLY non-input-derived datum in the document (a wall-clock timestamp,
    # S4.3/NFR-3). After stripping it the documents are byte-identical (NFR-1).
    diff = [
        (x, y)
        for x, y in zip(doc_a.splitlines(), doc_b.splitlines())
        if x != y
    ]
    assert len(doc_a.splitlines()) == len(doc_b.splitlines())
    assert all(
        x.startswith("generated_at:") and y.startswith("generated_at:")
        for x, y in diff
    ), f"unexpected non-timestamp differences: {diff}"
    assert _strip_generated_at(doc_a) == _strip_generated_at(doc_b)


# --------------------------------------------------------------------------- #
# 3. EH-7 empty result -- valid manifest-only document, no crash
# --------------------------------------------------------------------------- #
def test_eh7_empty_result_manifest_only_and_valid(tmp_path):
    project = init_workspace(tmp_path / "ws", "Empty")
    # Every candidate is classified ``absent`` -> all score 0.00 -> all below the
    # 0.50 threshold -> zero retained -> EH-7 guided empty-result document.
    llm = FakeLLM(
        queries=["llm agents"],
        assessments_by_marker={},
        default_assessment=_ASSESS_ABSENT,
    )
    written = _run_pipeline(project, [_PAPER_A, _PAPER_B], llm=llm)
    content = written.read_text(encoding="utf-8")

    # Does not crash; a valid manifest-only document is written (S4.5 passes).
    assert validate_document(content) == []

    manifest, records = parse_s44(content)
    assert records == []  # no paper records
    assert manifest["papers_total"] == 0
    assert manifest["counts"]["retained"] == 0
    assert "paper_id: paper_" not in content

    # EH-7 requires a '## Notes' section reporting the funnel + a relax hint.
    assert "## Notes" in content
    assert "Recommendation: relax" in content


def test_eh7_hard_filter_wipeout_recommends_conferences(tmp_path):
    project = init_workspace(tmp_path / "ws", "Empty")
    # A non-preprint paper whose venue is not in the configured conferences is
    # removed at the hard filter -> after_hard_filter == 0 -> recommend
    # relaxing 'conferences' (the active strictest gate).
    off_topic = {
        "title": "Some Unrelated Database Paper",
        "abstract": "A paper about databases.",
        "venue": "VLDB",
        "year": 2024,
        "language": "en",
        "doi": "10.1/vldb.x",
    }
    written = _run_pipeline(project, [off_topic])
    content = written.read_text(encoding="utf-8")
    assert validate_document(content) == []
    manifest, records = parse_s44(content)
    assert records == []
    assert manifest["papers_total"] == 0
    assert manifest["counts"]["after_hard_filter"] == 0
    assert "Recommendation: relax conferences." in content


# --------------------------------------------------------------------------- #
# 4. Registration + no circular import
# --------------------------------------------------------------------------- #
def test_phase1_registered_to_pipeline_class():
    import anvesha.phases as phases

    assert phases.PHASE_REGISTRY[1] is FilterLiteraturePipeline


def test_importing_cli_works_no_circular_import():
    # A clean import of the CLI (which imports the phase registry transitively)
    # must succeed without a circular-import error and must see Phase 1 wired.
    import importlib

    cli = importlib.import_module("anvesha.cli")
    assert cli is not None
    import anvesha.phases as phases

    assert phases.PHASE_REGISTRY[1] is FilterLiteraturePipeline


def test_unimplemented_phases_absent_from_registry():
    import anvesha.phases as phases

    # Only Phase 1 is implemented; phases 2..11 must NOT be registered yet, so
    # the CLI reports them as "not implemented" rather than mis-dispatching.
    assert sorted(phases.PHASE_REGISTRY) == [1]
    for phase_id in range(2, 12):
        assert phase_id not in phases.PHASE_REGISTRY

"""Tests for the Phase 1 data layer: config, candidate, counts, record, tools.

Offline only -- no network, no MCP servers. The ToolRouter tests use a fake
``MCPClient`` returning scripted tool defs and JSON strings.
"""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from anvesha.core.adapters.base import MCPToolDef
from anvesha.core.exceptions import AnveshaError
from anvesha.phases.phase01_filter.schemas.candidate import RetainedPaper
from anvesha.phases.phase01_filter.schemas.config import (
    FilterConfig,
    Options,
    YearRange,
    build_filter_config,
    build_options,
)
from anvesha.phases.phase01_filter.schemas.counts import FilterCounts
from anvesha.phases.phase01_filter.schemas.record import (
    GENERATOR,
    PAPER_FIELD_ORDER,
    SCHEMA_VERSION,
    manifest_counts,
    render_document,
)
from anvesha.phases.phase01_filter.tools import ToolRouter


# --------------------------------------------------------------------------- #
# config.py
# --------------------------------------------------------------------------- #
def test_yearrange_accepts_equal_and_ordered() -> None:
    assert YearRange(start=2023, end=2025).end == 2025
    assert YearRange(start=2024, end=2024).start == 2024


def test_yearrange_rejects_inverted() -> None:
    with pytest.raises(ValidationError) as exc:
        YearRange(start=2025, end=2023)
    assert "years.start" in str(exc.value)


def test_build_filter_config_tolerates_year_mapping() -> None:
    cfg = build_filter_config(
        {
            "conferences": ["NeurIPS", "ICML"],
            "domains": ["LLM Agents"],
            "years": {"start": 2023, "end": 2025},
        }
    )
    assert isinstance(cfg, FilterConfig)
    assert cfg.years.start == 2023 and cfg.years.end == 2025
    assert cfg.conferences == ["NeurIPS", "ICML"]


def test_build_filter_config_empty_conferences_allowed() -> None:
    cfg = build_filter_config({"domains": ["RAG"], "years": {"start": 2020, "end": 2021}})
    assert cfg.conferences == []


def test_build_filter_config_inverted_years_raises() -> None:
    with pytest.raises(ValidationError):
        build_filter_config(
            {"domains": ["X"], "years": {"start": 2030, "end": 2020}}
        )


def test_build_options_defaults_when_none() -> None:
    opts = build_options(None)
    assert isinstance(opts, Options)
    assert opts.max_papers == 40
    assert opts.relevance_threshold == 0.50
    assert opts.dedup_threshold == 0.90
    assert opts.include_preprints is True
    assert opts.overwrite is False


def test_build_options_partial_override_keeps_defaults() -> None:
    opts = build_options({"max_papers": 10, "relevance_threshold": None})
    assert opts.max_papers == 10
    assert opts.relevance_threshold is None
    # untouched fields keep defaults
    assert opts.dedup_threshold == 0.90
    assert opts.language == "en"


# --------------------------------------------------------------------------- #
# counts.py / manifest_counts
# --------------------------------------------------------------------------- #
def test_manifest_counts_order_and_values() -> None:
    counts = FilterCounts(
        identified=412,
        after_deduplication=318,
        after_hard_filter=96,
        screened=96,
        retained=40,
        pdfs_downloaded=38,
        pdfs_failed=2,
        with_code=23,
    )
    out = manifest_counts(counts)
    assert list(out.keys()) == [
        "identified",
        "after_deduplication",
        "after_hard_filter",
        "screened",
        "retained",
        "pdfs_downloaded",
        "pdfs_failed",
        "with_code",
    ]
    assert out["identified"] == 412
    assert out["with_code"] == 23


# --------------------------------------------------------------------------- #
# record.py - golden document
# --------------------------------------------------------------------------- #
def _two_paper_fixture() -> tuple[dict, list[RetainedPaper]]:
    manifest = {
        "generated_at": "2026-01-15T09:42:11Z",
        "filters": {
            "conferences": ["NeurIPS", "ICML", "CVPR"],
            "domains": ["LLM Agents", "Retrieval-Augmented Generation"],
            "years": {"start": 2023, "end": 2025},
        },
        "counts": FilterCounts(
            identified=412,
            after_deduplication=318,
            after_hard_filter=96,
            screened=96,
            retained=2,
            pdfs_downloaded=1,
            pdfs_failed=1,
            with_code=1,
        ),
        "papers_total": 2,
        "output_dir": "filtered_literature_v1_llm_agents_2023_2025",
    }
    p1 = RetainedPaper(
        paper_id="paper_001",
        rank=1,
        title="Toolformer-Style Self-Supervised Tool Use for Retrieval-Augmented Agents",
        authors=["A. Researcher", "B. Scientist"],
        conference="NeurIPS",
        year=2024,
        domain=["LLM Agents", "Retrieval-Augmented Generation"],
        paper_url="https://example.org/abs/2024.00001",
        pdf_path="pdfs/paper_001.pdf",
        pdf_status="ok",
        code_url="https://github.com/example/toolformer-rag",
        git_exists=True,
        relevance_score=0.94,
        doi="10.0000/neurips.2024.00001",
        arxiv_id="2401.00001",
        source="merged",
        summary="Problem: tool use. Methodology: self-supervision. Contributions: scheme.",
        summary_status="ok",
    )
    p2 = RetainedPaper(
        paper_id="paper_002",
        rank=2,
        title="Memory Compression Strategies for Long-Horizon LLM Agents",
        authors=["C. Author"],
        conference="ICML",
        year=2025,
        domain=["LLM Agents"],
        paper_url="https://example.org/abs/2025.00042",
        pdf_path=None,
        pdf_status="failed",
        code_url=None,
        git_exists=False,
        relevance_score=0.8,  # must render as 0.80
        doi=None,
        arxiv_id="2505.00042",
        source="paper_search",
        summary="Problem: context overflow. Methodology: compression comparison.",
        summary_status="ok",
    )
    return manifest, [p1, p2]


def test_render_document_golden_string() -> None:
    manifest, papers = _two_paper_fixture()
    doc = render_document(manifest, papers)
    expected = (
        "---\n"
        "document_type: filtered_literature_index\n"
        'schema_version: "2.0"\n'
        "generated_at: 2026-01-15T09:42:11Z\n"
        "generator: anvesha.phase01_filter\n"
        "filters:\n"
        "  conferences:\n"
        "    - NeurIPS\n"
        "    - ICML\n"
        "    - CVPR\n"
        "  domains:\n"
        "    - LLM Agents\n"
        "    - Retrieval-Augmented Generation\n"
        "  years:\n"
        "    start: 2023\n"
        "    end: 2025\n"
        "counts:\n"
        "  identified: 412\n"
        "  after_deduplication: 318\n"
        "  after_hard_filter: 96\n"
        "  screened: 96\n"
        "  retained: 2\n"
        "  pdfs_downloaded: 1\n"
        "  pdfs_failed: 1\n"
        "  with_code: 1\n"
        "papers_total: 2\n"
        "output_dir: filtered_literature_v1_llm_agents_2023_2025\n"
        "---\n"
        "\n"
        "---\n"
        "paper_id: paper_001\n"
        "title: Toolformer-Style Self-Supervised Tool Use for Retrieval-Augmented Agents\n"
        "authors:\n"
        "  - A. Researcher\n"
        "  - B. Scientist\n"
        "conference: NeurIPS\n"
        "year: 2024\n"
        "domain:\n"
        "  - LLM Agents\n"
        "  - Retrieval-Augmented Generation\n"
        "paper_url: https://example.org/abs/2024.00001\n"
        "pdf_path: pdfs/paper_001.pdf\n"
        "pdf_status: ok\n"
        "code_url: https://github.com/example/toolformer-rag\n"
        "git_exists: true\n"
        "relevance_score: 0.94\n"
        "rank: 1\n"
        "doi: 10.0000/neurips.2024.00001\n"
        'arxiv_id: "2401.00001"\n'
        "source: merged\n"
        "---\n"
        "\n"
        "## LLM Summary\n"
        "\n"
        "Problem: tool use. Methodology: self-supervision. Contributions: scheme.\n"
        "\n"
        "---\n"
        "paper_id: paper_002\n"
        "title: Memory Compression Strategies for Long-Horizon LLM Agents\n"
        "authors:\n"
        "  - C. Author\n"
        "conference: ICML\n"
        "year: 2025\n"
        "domain:\n"
        "  - LLM Agents\n"
        "paper_url: https://example.org/abs/2025.00042\n"
        "pdf_path: null\n"
        "pdf_status: failed\n"
        "code_url: null\n"
        "git_exists: false\n"
        "relevance_score: 0.80\n"
        "rank: 2\n"
        "doi: null\n"
        'arxiv_id: "2505.00042"\n'
        "source: paper_search\n"
        "---\n"
        "\n"
        "## LLM Summary\n"
        "\n"
        "Problem: context overflow. Methodology: compression comparison.\n"
    )
    assert doc == expected


def test_render_document_is_deterministic() -> None:
    manifest, papers = _two_paper_fixture()
    assert render_document(manifest, papers) == render_document(manifest, papers)


def test_render_document_two_decimal_score() -> None:
    manifest, papers = _two_paper_fixture()
    doc = render_document(manifest, papers)
    assert "relevance_score: 0.80\n" in doc  # 0.8 -> 0.80
    assert "relevance_score: 0.94\n" in doc


def test_render_document_null_literals_for_none() -> None:
    manifest, papers = _two_paper_fixture()
    doc = render_document(manifest, papers)
    assert "code_url: null\n" in doc
    assert "doi: null\n" in doc
    assert "pdf_path: null\n" in doc


def _split_yaml_blocks(doc: str) -> list[dict]:
    """Parse every '---'-delimited YAML frontmatter block per the S4.4 grammar."""
    blocks: list[dict] = []
    lines = doc.split("\n")
    i = 0
    while i < len(lines):
        if lines[i] == "---":
            j = i + 1
            buf: list[str] = []
            while j < len(lines) and lines[j] != "---":
                buf.append(lines[j])
                j += 1
            blocks.append(yaml.safe_load("\n".join(buf)))
            i = j + 1
        else:
            i += 1
    return blocks


def test_render_document_blocks_are_valid_yaml() -> None:
    manifest, papers = _two_paper_fixture()
    doc = render_document(manifest, papers)
    blocks = _split_yaml_blocks(doc)
    # 1 manifest + 2 paper records
    assert len(blocks) == 3
    run_manifest = blocks[0]
    assert run_manifest["document_type"] == "filtered_literature_index"
    assert "paper_id" not in run_manifest
    assert run_manifest["schema_version"] == SCHEMA_VERSION
    assert run_manifest["generator"] == GENERATOR
    assert run_manifest["papers_total"] == 2
    assert run_manifest["counts"]["identified"] == 412
    assert run_manifest["filters"]["years"] == {"start": 2023, "end": 2025}

    rec1, rec2 = blocks[1], blocks[2]
    assert rec1["paper_id"] == "paper_001"
    assert rec1["git_exists"] is True
    assert rec1["relevance_score"] == 0.94
    assert rec1["authors"] == ["A. Researcher", "B. Scientist"]
    # nulls parse back to None
    assert rec2["code_url"] is None
    assert rec2["doi"] is None
    assert rec2["pdf_path"] is None
    assert rec2["git_exists"] is False
    # numeric-looking identifiers stay strings (not parsed as YAML floats)
    assert rec1["arxiv_id"] == "2401.00001"
    assert isinstance(rec1["arxiv_id"], str)
    assert rec2["arxiv_id"] == "2505.00042"
    assert isinstance(rec2["arxiv_id"], str)


def test_render_paper_field_order_exact() -> None:
    """The keys in each record appear in the exact S4.3 canonical order."""
    manifest, papers = _two_paper_fixture()
    doc = render_document(manifest, papers)
    # Extract the first paper record's key order from the raw text.
    start = doc.index("paper_id: paper_001")
    end = doc.index("\n---", start)
    record_text = doc[start:end]
    keys = [
        line.split(":", 1)[0]
        for line in record_text.split("\n")
        if line and not line.startswith(" ") and ":" in line
    ]
    assert keys == list(PAPER_FIELD_ORDER)


def test_render_document_quotes_ambiguous_title() -> None:
    """A title that looks like a YAML special value is quoted and round-trips."""
    manifest, papers = _two_paper_fixture()
    papers[0].title = "null"  # would parse as None if emitted plain
    papers[1].title = "Yes: a study of agents"  # contains ': '
    doc = render_document(manifest, papers)
    blocks = _split_yaml_blocks(doc)
    assert blocks[1]["title"] == "null"
    assert blocks[2]["title"] == "Yes: a study of agents"


def test_render_document_empty_paper_list() -> None:
    """EH-7 empty result: manifest renders with papers_total 0 and no records."""
    manifest, _ = _two_paper_fixture()
    manifest["papers_total"] = 0
    doc = render_document(manifest, [])
    blocks = _split_yaml_blocks(doc)
    assert len(blocks) == 1
    assert blocks[0]["papers_total"] == 0
    assert "## LLM Summary" not in doc


def test_render_document_accepts_filterconfig_object() -> None:
    """manifest['filters'] may be a FilterConfig (model_dump path)."""
    manifest, papers = _two_paper_fixture()
    manifest["filters"] = FilterConfig(
        conferences=["NeurIPS"],
        domains=["LLM Agents"],
        years=YearRange(start=2023, end=2025),
    )
    doc = render_document(manifest, papers)
    blocks = _split_yaml_blocks(doc)
    assert blocks[0]["filters"]["conferences"] == ["NeurIPS"]
    assert blocks[0]["filters"]["years"] == {"start": 2023, "end": 2025}


# --------------------------------------------------------------------------- #
# tools.py - ToolRouter
# --------------------------------------------------------------------------- #
class _FakeMCP:
    """Scripted MCPClient: advertises tool defs, echoes calls as JSON."""

    def __init__(self, names: list[str], responses: dict | None = None):
        self._names = names
        self._responses = responses or {}
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        return [MCPToolDef(name=n, description="", input_schema={}) for n in self._names]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return self._responses.get(name, '{"results": []}')


class _BrokenMCP:
    """A client whose list_tools raises -- should be skipped gracefully (EH-2)."""

    def list_tools(self) -> list[MCPToolDef]:
        raise RuntimeError("server down")

    def call_tool(self, name: str, arguments: dict) -> str:  # pragma: no cover
        return "{}"


def test_toolrouter_maps_names_to_clients() -> None:
    a = _FakeMCP(["papersflow.search", "papersflow.search_literature"])
    b = _FakeMCP(["paper-search.search_papers", "paper-search.download_with_fallback"])
    router = ToolRouter([a, b])
    assert router.has("papersflow.search")
    assert router.has("paper-search.download_with_fallback")
    assert not router.has("nonexistent.tool")
    assert set(router.tool_names) == {
        "papersflow.search",
        "papersflow.search_literature",
        "paper-search.search_papers",
        "paper-search.download_with_fallback",
    }


def test_toolrouter_call_dispatches_to_owning_client() -> None:
    a = _FakeMCP(["papersflow.search"], {"papersflow.search": '{"hits": 3}'})
    b = _FakeMCP(["paper-search.search_papers"])
    router = ToolRouter([a, b])
    out = router.call("papersflow.search", {"query": "agents"})
    assert out == '{"hits": 3}'
    assert a.calls == [("papersflow.search", {"query": "agents"})]
    assert b.calls == []


def test_toolrouter_call_missing_tool_raises_clear_error() -> None:
    router = ToolRouter([_FakeMCP(["papersflow.search"])])
    with pytest.raises(AnveshaError) as exc:
        router.call("paper-search.download_with_fallback", {})
    msg = str(exc.value)
    assert "paper-search.download_with_fallback" in msg
    assert "papersflow.search" in msg  # lists what IS available


def test_toolrouter_first_registered_wins_on_duplicate() -> None:
    a = _FakeMCP(["paper-search.search_papers"], {"paper-search.search_papers": "A"})
    b = _FakeMCP(["paper-search.search_papers"], {"paper-search.search_papers": "B"})
    router = ToolRouter([a, b])
    assert router.call("paper-search.search_papers", {}) == "A"


def test_toolrouter_skips_broken_client() -> None:
    good = _FakeMCP(["papersflow.search"])
    router = ToolRouter([_BrokenMCP(), good])
    assert router.has("papersflow.search")
    assert router.tool_names == ["papersflow.search"]

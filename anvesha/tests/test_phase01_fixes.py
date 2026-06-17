"""Regression tests for Phase 1 review fixes (B1, B2, B3, F1, language edge).

Each test pins a confirmed bug fix so it cannot silently regress.
"""

from __future__ import annotations

from typing import Sequence

import pytest

from anvesha.core.adapters.base import MCPClient, MCPToolDef
from anvesha.core.exceptions import AnveshaError
from anvesha.phases.phase01_filter.schemas.candidate import (
    PaperCandidate,
    RelevanceAssessment,
    RetainedPaper,
    ScoredCandidate,
)
from anvesha.phases.phase01_filter.schemas.config import Options
from anvesha.phases.phase01_filter.schemas.record import (
    _sanitize_summary,
    render_paper_block,
)
from anvesha.phases.phase01_filter.stages.dedup import deduplicate, title_similarity
from anvesha.phases.phase01_filter.stages.hard_filter import _primary_subtag
from anvesha.phases.phase01_filter.stages.rank import rank_and_select
from anvesha.phases.phase01_filter.stages.relevance import (
    compute_raw_relevance_score,
    compute_relevance_score,
)
from anvesha.phases.phase01_filter.stages.search import search_candidates
from anvesha.phases.phase01_filter.tools import ToolRouter


# --- B1: title-less papers must not merge on title similarity ----------------

def test_title_similarity_zero_when_a_title_is_missing():
    assert title_similarity("", "") == 0.0
    assert title_similarity("Deep Learning for X", "") == 0.0
    # real near-duplicates still merge
    assert title_similarity("Deep Learning for X", "Deep Learning for X") == 1.0


def test_dedup_does_not_merge_distinct_titleless_papers():
    a = PaperCandidate(candidate_id="c1", title="", source="papersflow", doi="10.1/aaa", year=2024)
    b = PaperCandidate(candidate_id="c2", title="", source="paper_search", doi="10.1/bbb", year=2025)
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 2  # distinct papers, not collapsed into one


# --- B2: a bare "---" line inside a summary must not fence the S4.4 grammar ---

def test_sanitize_summary_neutralizes_fence_lines():
    out = _sanitize_summary("intro\n---\ntail")
    assert "---" not in out.split("\n")  # no line is exactly the fence
    assert out.split("\n") == ["intro", "———", "tail"]


def test_render_paper_block_summary_does_not_add_fence():
    paper = RetainedPaper(paper_id="paper_001", rank=1, title="T", summary="a\n---\nb")
    block = render_paper_block(paper)
    fence_lines = [ln for ln in block.split("\n") if ln.strip() == "---"]
    # exactly the two frontmatter delimiters; the summary's "---" was neutralized
    assert len(fence_lines) == 2


# --- B3: EH-3 - all discovery sources failing at call time is fatal ----------

class _FakeMCP(MCPClient):
    def __init__(self, tools: Sequence[str], *, raise_on_call: set[str] | None = None,
                 responses: dict[str, str] | None = None):
        self._tools = list(tools)
        self._raise = raise_on_call or set()
        self._responses = responses or {}

    def list_tools(self) -> list[MCPToolDef]:
        return [MCPToolDef(name=n, description="", input_schema={}) for n in self._tools]

    def call_tool(self, name: str, arguments: dict) -> str:
        if name in self._raise:
            raise RuntimeError(f"{name} down")
        return self._responses.get(name, "[]")


def test_search_raises_when_all_sources_fail_at_call_time():
    pf = _FakeMCP(["papersflow.search"], raise_on_call={"papersflow.search"})
    ps = _FakeMCP(["paper-search.search_papers"], raise_on_call={"paper-search.search_papers"})
    router = ToolRouter([pf, ps])
    with pytest.raises(AnveshaError):
        search_candidates(router, ["agents"], max_results_per_source=10)


def test_search_one_source_down_degrades_not_fatal():
    # papersflow fails every call, paper-search succeeds (empty) -> EH-2, no raise
    pf = _FakeMCP(["papersflow.search"], raise_on_call={"papersflow.search"})
    ps = _FakeMCP(["paper-search.search_papers"], responses={"paper-search.search_papers": "[]"})
    router = ToolRouter([pf, ps])
    assert search_candidates(router, ["agents"], max_results_per_source=10) == []


# --- F1: S6.3 tie-break rule 1 uses the full-precision (unrounded) score -----

def test_raw_score_is_unrounded_and_emitted_is_rounded():
    # peripheral (best=0.35) + partial topic (0.9) -> raw 0.315, emitted 0.32 (or .31)
    assessment = RelevanceAssessment(domain_matches=[], topic_alignment="no_topic_statement")
    # all-absent path: raw 0.0
    assert compute_raw_relevance_score(assessment) == 0.0
    assert compute_relevance_score(assessment) == 0.0


def _scored(raw: float, *, title: str, year: int = 2024) -> ScoredCandidate:
    return ScoredCandidate(
        candidate=PaperCandidate(
            candidate_id=f"c-{raw}", title=title, source="papersflow",
            year=year, citation_count=5,
        ),
        assessment=RelevanceAssessment(domain_matches=[], topic_alignment="no_topic_statement"),
        relevance_score=round(raw, 2),
        raw_relevance_score=raw,
    )


def test_rank_breaks_equal_rounded_score_by_full_precision():
    # Both round to 0.84 but differ in raw; higher raw must rank first regardless
    # of input order. Identical year/citations/title isolates rule 1.
    lo = _scored(0.841, title="same title")
    hi = _scored(0.844, title="same title")
    assert round(0.841, 2) == round(0.844, 2) == 0.84  # same emitted score
    opts = Options(relevance_threshold=None)
    for order in ([lo, hi], [hi, lo]):
        ranked = rank_and_select(order, opts)
        assert ranked[0].rank == 1
        # rounded scores tie; rule 1 (full precision) puts the higher raw first
        assert ranked[0].relevance_score == ranked[1].relevance_score == 0.84
        assert ranked[0].paper_id == "paper_001"


# --- language edge: underscore locale must not drop an English paper ---------

def test_primary_subtag_handles_underscore_locale():
    assert _primary_subtag("en_US") == "en"
    assert _primary_subtag("en-GB") == "en"
    assert _primary_subtag("EN") == "en"

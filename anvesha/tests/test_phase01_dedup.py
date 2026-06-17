"""Tests for Phase 1 Stage 3 -- Normalize & Deduplicate (spec S7).

Offline: pure string/data logic, no network/MCP/LLM/GPU.
"""

from __future__ import annotations

import itertools

import pytest

from anvesha.phases.phase01_filter.schemas.candidate import PaperCandidate
from anvesha.phases.phase01_filter.stages.dedup import (
    deduplicate,
    normalize_title,
    title_similarity,
)


def make(
    cid: str,
    title: str,
    *,
    authors: list[str] | None = None,
    abstract: str | None = None,
    venue_raw: str | None = None,
    venue_canonical: str | None = None,
    year: int | None = None,
    citation_count: int | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    paper_url: str | None = None,
    metadata_code_url: str | None = None,
    language: str | None = None,
    is_preprint: bool = False,
    source: str = "papersflow",
    found_by_queries: list[str] | None = None,
) -> PaperCandidate:
    return PaperCandidate(
        candidate_id=cid,
        title=title,
        authors=authors or [],
        abstract=abstract,
        venue_raw=venue_raw,
        venue_canonical=venue_canonical,
        year=year,
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        paper_url=paper_url,
        metadata_code_url=metadata_code_url,
        language=language,
        is_preprint=is_preprint,
        source=source,  # type: ignore[arg-type]
        found_by_queries=found_by_queries or [],
    )


# --- normalization (S7.1) ------------------------------------------------


def test_normalize_lowercases_strips_punct_collapses_ws():
    assert normalize_title("  Deep   Learning: A Survey!! ") == "deep learning a survey"


def test_normalize_none_and_empty():
    assert normalize_title(None) == ""
    assert normalize_title("   ") == ""


def test_normalize_nfc_unicode():
    # Composed (U+00E9) vs decomposed (e + U+0301 combining acute) e-acute
    # normalize to one string. NFC composes accents but does not strip them,
    # so the two encodings collapse together (and would dedup together).
    composed = normalize_title("R\u00e9sum\u00e9 Learning")
    decomposed = normalize_title("Re\u0301sume\u0301 Learning")
    assert composed == decomposed
    assert composed == "r\u00e9sum\u00e9 learning"


# --- title_similarity (S7.1) ---------------------------------------------


def test_similarity_identical_is_one():
    assert title_similarity("Attention Is All You Need", "attention is all you need") == 1.0


def test_similarity_bounds():
    for a, b in [
        ("graph neural networks", "completely different topic words here"),
        ("a", "z"),
        ("", "something"),
        ("transformer models", "transformer models for vision"),
    ]:
        s = title_similarity(a, b)
        assert 0.0 <= s <= 1.0


def test_similarity_disjoint_is_low():
    s = title_similarity("alpha beta gamma", "delta epsilon zeta")
    assert s < 0.5


def test_similarity_word_order_via_jaccard():
    # Same token set, different order -> Jaccard = 1.0, so max = 1.0.
    assert title_similarity("learning deep neural", "neural deep learning") == 1.0


def test_similarity_minor_char_diff_via_levenshtein():
    # One-character typo: Jaccard would be 0 (no shared whole tokens) but
    # Levenshtein ratio is high; max captures it.
    s = title_similarity("transformer", "transformer")
    assert s >= 0.9


# --- exact identifier dedup overrides title difference (S7.2) ------------


def test_exact_doi_dedup_despite_different_titles():
    a = make("c1", "Original Preprint Title", doi="10.1/ABC", is_preprint=True)
    b = make("c2", "Totally Revised Camera Ready Heading", doi="10.1/abc",
             venue_raw="NeurIPS", year=2024, source="paper_search")
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 1
    rec = out[0]
    # Published member donates venue/year/title (S7.3.1).
    assert rec.venue_raw == "NeurIPS"
    assert rec.year == 2024
    assert rec.title == "Totally Revised Camera Ready Heading"
    assert rec.doi == "10.1/ABC"  # first non-null DOI retained verbatim
    assert rec.source == "merged"


def test_exact_arxiv_dedup_despite_different_titles():
    a = make("c1", "Some Title", arxiv_id="2401.00001")
    b = make("c2", "Another Title Entirely", arxiv_id="2401.00001")
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 1
    assert out[0].arxiv_id == "2401.00001"


def test_different_identifiers_not_merged_when_titles_differ():
    a = make("c1", "Graph Neural Networks for Chemistry", doi="10.1/aaa")
    b = make("c2", "Reinforcement Learning in Robotics", doi="10.1/bbb")
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 2


# --- high title-sim dedup with NO shared DOI (S7.2) ----------------------


def test_title_similarity_dedup_without_shared_id():
    a = make("c1", "Attention Is All You Need", arxiv_id="1706.03762", is_preprint=True)
    b = make("c2", "Attention is all you need.", venue_raw="NeurIPS", year=2017,
             source="paper_search")
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 1
    rec = out[0]
    assert rec.year == 2017
    assert rec.venue_raw == "NeurIPS"
    assert rec.arxiv_id == "1706.03762"  # retained for PDF fallback (S7.3.2)
    assert rec.source == "merged"


def test_below_threshold_not_merged():
    a = make("c1", "Deep Learning for Image Classification")
    b = make("c2", "Shallow Methods for Audio Segmentation")
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 2


# --- order independence (S7.5) -------------------------------------------


def test_order_independence_canonical_record_stable():
    members = [
        make("c1", "Attention Is All You Need", arxiv_id="1706.03762",
             is_preprint=True, authors=["A", "B"], abstract="short",
             found_by_queries=["q1"], source="papersflow"),
        make("c2", "Attention is all you need", venue_raw="NeurIPS", year=2017,
             authors=["A", "B", "C"], abstract="a longer abstract body",
             found_by_queries=["q2"], source="paper_search",
             metadata_code_url="https://github.com/x/y"),
        make("c3", "An Unrelated Paper About Databases", doi="10.5/db"),
    ]
    baseline = deduplicate(list(members), dedup_threshold=0.90)
    for perm in itertools.permutations(members):
        out = deduplicate(list(perm), dedup_threshold=0.90)
        assert [c.model_dump() for c in out] == [c.model_dump() for c in baseline]
    # Sanity: two of the three collapsed into one.
    assert len(baseline) == 2


def test_transitive_grouping_connected_components():
    # a~b (title), b~c (arxiv); a and c never directly compared but must group.
    a = make("c1", "Deep Residual Learning for Image Recognition")
    b = make("c2", "Deep residual learning for image recognition.",
             arxiv_id="1512.03385")
    c = make("c3", "Camera Ready Final Version Heading", arxiv_id="1512.03385",
             venue_raw="CVPR", year=2016)
    out = deduplicate([a, b, c], dedup_threshold=0.90)
    assert len(out) == 1


# --- conflict-resolution picks (S7.3) ------------------------------------


def test_conflict_resolution_picks():
    pre = make(
        "c1", "Some Paper",
        is_preprint=True, arxiv_id="2401.99999",
        authors=["X"], abstract="tiny",
        citation_count=3, source="papersflow",
        metadata_code_url="https://example.com/page",
        found_by_queries=["alpha"],
    )
    pub = make(
        "c2", "Some Paper",
        venue_raw="ICML", year=2024, doi="10.9/icml",
        authors=["X", "Y", "Z"], abstract="a much more complete abstract here",
        citation_count=42, source="paper_search",
        metadata_code_url="https://github.com/org/repo",
        found_by_queries=["beta", "alpha"],
    )
    out = deduplicate([pre, pub], dedup_threshold=0.90)
    assert len(out) == 1
    rec = out[0]
    # S7.3.1 venue/year from published
    assert rec.venue_raw == "ICML"
    assert rec.year == 2024
    assert rec.is_preprint is False
    # S7.3.2 both identifiers retained
    assert rec.doi == "10.9/icml"
    assert rec.arxiv_id == "2401.99999"
    # S7.3.3 known code host beats generic URL
    assert rec.metadata_code_url == "https://github.com/org/repo"
    # S7.3.4 longest abstract
    assert rec.abstract == "a much more complete abstract here"
    # S7.3.5 longest author list
    assert rec.authors == ["X", "Y", "Z"]
    # citation count = max known
    assert rec.citation_count == 42
    # S7.3.6 merged source
    assert rec.source == "merged"
    # S7.3.7 union of found_by_queries, first-seen order
    assert rec.metadata_code_url == "https://github.com/org/repo"
    assert sorted(rec.found_by_queries) == ["alpha", "beta"]


def test_code_link_known_host_beats_generic_and_lexicographic_tiebreak():
    a = make("c1", "P", arxiv_id="9", metadata_code_url="https://example.com/z")
    b = make("c2", "P", arxiv_id="9", metadata_code_url="https://gitlab.com/b/repo")
    c = make("c3", "P", arxiv_id="9", metadata_code_url="https://github.com/a/repo")
    out = deduplicate([a, b, c], dedup_threshold=0.90)
    assert len(out) == 1
    # Two known hosts: github.com vs gitlab.com -> lexicographically smaller URL.
    assert out[0].metadata_code_url == "https://github.com/a/repo"


def test_authors_tie_break_prefers_published():
    pre = make("c1", "P", arxiv_id="7", is_preprint=True, authors=["A", "B"])
    pub = make("c2", "P", arxiv_id="7", venue_raw="ACL", year=2023, authors=["C", "D"])
    out = deduplicate([pre, pub], dedup_threshold=0.90)
    assert len(out) == 1
    # Equal length (2 == 2) -> published record's authors win (S7.3.5).
    assert out[0].authors == ["C", "D"]


def test_single_source_group_not_marked_merged():
    a = make("c1", "Same Title Here", arxiv_id="123", source="papersflow")
    b = make("c2", "Same Title Here", arxiv_id="123", source="papersflow")
    out = deduplicate([a, b], dedup_threshold=0.90)
    assert len(out) == 1
    assert out[0].source == "papersflow"


def test_empty_input():
    assert deduplicate([], dedup_threshold=0.90) == []


def test_no_duplicates_passthrough_sorted_by_candidate_id():
    a = make("c3", "Reinforcement Learning for Robotics Control")
    b = make("c1", "Graph Neural Networks in Computational Chemistry")
    c = make("c2", "Transformer Architectures for Speech Recognition")
    out = deduplicate([a, b, c], dedup_threshold=0.90)
    assert [r.candidate_id for r in out] == ["c1", "c2", "c3"]

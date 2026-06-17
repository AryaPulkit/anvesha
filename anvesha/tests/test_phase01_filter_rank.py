"""Tests for Phase 1 Stage 4 (hard filter) and Stage 6 (rank & select).

Offline and deterministic: no network, no MCP, no LLM. Construct candidates and
the venue map directly. Covers (per assignment):
  - conference hard-filter + canonicalization + empty-list skip + preprint exception
  - year bounds + null handling
  - language drop
  - threshold drop
  - tie-break determinism (equal scores -> deterministic, input-order-independent)
  - max_papers cap
  - paper_id assignment
"""

from __future__ import annotations

from anvesha.phases.phase01_filter.schemas.candidate import (
    PaperCandidate,
    RelevanceAssessment,
    ScoredCandidate,
)
from anvesha.phases.phase01_filter.schemas.config import (
    FilterConfig,
    Options,
    YearRange,
)
from anvesha.phases.phase01_filter.stages.hard_filter import hard_filter
from anvesha.phases.phase01_filter.stages.rank import rank_and_select
from anvesha.phases.phase01_filter.venues.alias_map import VenueAliasMap


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def make_candidate(
    cid: str,
    *,
    title: str = "A Paper",
    venue_raw: str | None = None,
    year: int | None = 2024,
    language: str | None = None,
    is_preprint: bool = False,
    citation_count: int | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    paper_url: str | None = None,
    authors: list[str] | None = None,
    source: str = "paper_search",
) -> PaperCandidate:
    return PaperCandidate(
        candidate_id=cid,
        title=title,
        authors=authors or [],
        venue_raw=venue_raw,
        year=year,
        language=language,
        is_preprint=is_preprint,
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        paper_url=paper_url,
        source=source,  # type: ignore[arg-type]
    )


def make_scored(
    cid: str,
    score: float,
    *,
    title: str = "A Paper",
    year: int | None = 2024,
    citation_count: int | None = None,
    matched_domains: list[str] | None = None,
    venue_canonical: str | None = "NeurIPS",
    doi: str | None = None,
    arxiv_id: str | None = None,
    paper_url: str | None = None,
    authors: list[str] | None = None,
    source: str = "paper_search",
) -> ScoredCandidate:
    cand = make_candidate(
        cid,
        title=title,
        year=year,
        citation_count=citation_count,
        doi=doi,
        arxiv_id=arxiv_id,
        paper_url=paper_url,
        authors=authors,
        source=source,
    )
    cand.venue_canonical = venue_canonical
    return ScoredCandidate(
        candidate=cand,
        assessment=RelevanceAssessment(
            domain_matches=[], topic_alignment="no_topic_statement"
        ),
        relevance_score=score,
        matched_domains=matched_domains or ["LLM Agents"],
    )


def filter_config(
    conferences: list[str] | None = None,
    domains: list[str] | None = None,
    start: int = 2023,
    end: int = 2025,
) -> FilterConfig:
    return FilterConfig(
        conferences=conferences if conferences is not None else ["NeurIPS", "ICML"],
        domains=domains or ["LLM Agents"],
        years=YearRange(start=start, end=end),
    )


# --------------------------------------------------------------------------- #
# Stage 4 - conference filter
# --------------------------------------------------------------------------- #
def test_conference_filter_keeps_matching_and_canonicalizes() -> None:
    amap = VenueAliasMap()
    cands = [
        make_candidate("c1", venue_raw="NeurIPS"),
        make_candidate(
            "c2",
            venue_raw="Advances in Neural Information Processing Systems",
        ),
        make_candidate("c3", venue_raw="CVPR"),  # not in configured conferences
    ]
    kept, removals = hard_filter(cands, filter_config(), Options(), amap)
    kept_ids = [c.candidate_id for c in kept]
    assert kept_ids == ["c1", "c2"]
    # Both surviving candidates carry the resolved canonical key.
    assert all(c.venue_canonical == "NeurIPS" for c in kept)
    assert removals["conference"] == 1


def test_conference_filter_alias_config_resolves() -> None:
    # A config that uses the "NIPS" alias must still match NeurIPS candidates.
    amap = VenueAliasMap()
    cands = [make_candidate("c1", venue_raw="NeurIPS")]
    cfg = filter_config(conferences=["NIPS"])
    kept, removals = hard_filter(cands, cfg, Options(), amap)
    assert [c.candidate_id for c in kept] == ["c1"]
    assert removals["conference"] == 0


def test_empty_conferences_skips_venue_filter() -> None:
    amap = VenueAliasMap()
    cands = [
        make_candidate("c1", venue_raw="Some Random Venue"),
        make_candidate("c2", venue_raw="CVPR"),
        make_candidate("c3", venue_raw=None),
    ]
    cfg = filter_config(conferences=[])
    kept, removals = hard_filter(cands, cfg, Options(), amap)
    assert [c.candidate_id for c in kept] == ["c1", "c2", "c3"]
    assert removals["conference"] == 0
    # venue_canonical is still resolved where possible even with the filter off.
    assert kept[1].venue_canonical == "CVPR"
    assert kept[0].venue_canonical is None


def test_conference_preprint_exception_keeps_unresolved_preprint() -> None:
    amap = VenueAliasMap()
    cands = [
        # Unresolved venue, preprint -> kept by exception.
        make_candidate("c1", venue_raw="arXiv", is_preprint=True),
        # Unresolved venue, not a preprint -> dropped.
        make_candidate("c2", venue_raw="Unknown Workshop", is_preprint=False),
        # Null venue, preprint -> kept by exception.
        make_candidate("c3", venue_raw=None, is_preprint=True),
    ]
    kept, removals = hard_filter(cands, filter_config(), Options(), amap)
    assert [c.candidate_id for c in kept] == ["c1", "c3"]
    assert removals["conference"] == 1


def test_conference_preprint_exception_disabled_when_include_preprints_false() -> None:
    amap = VenueAliasMap()
    cands = [make_candidate("c1", venue_raw="arXiv", is_preprint=True)]
    opts = Options(include_preprints=False)
    kept, removals = hard_filter(cands, filter_config(), opts, amap)
    assert kept == []
    assert removals["conference"] == 1


# --------------------------------------------------------------------------- #
# Stage 4 - year filter
# --------------------------------------------------------------------------- #
def test_year_bounds_inclusive() -> None:
    amap = VenueAliasMap()
    cfg = filter_config(conferences=[], start=2023, end=2025)
    cands = [
        make_candidate("lo", year=2023),  # inclusive lower
        make_candidate("hi", year=2025),  # inclusive upper
        make_candidate("below", year=2022),
        make_candidate("above", year=2026),
    ]
    kept, removals = hard_filter(cands, cfg, Options(), amap)
    assert [c.candidate_id for c in kept] == ["lo", "hi"]
    assert removals["year"] == 2


def test_null_year_kept_only_for_preprint() -> None:
    amap = VenueAliasMap()
    cfg = filter_config(conferences=[])
    cands = [
        make_candidate("preprint_null", year=None, is_preprint=True),
        make_candidate("nonpreprint_null", year=None, is_preprint=False),
    ]
    kept, removals = hard_filter(cands, cfg, Options(), amap)
    assert [c.candidate_id for c in kept] == ["preprint_null"]
    assert removals["year"] == 1


def test_null_year_preprint_dropped_when_include_preprints_false() -> None:
    amap = VenueAliasMap()
    cfg = filter_config(conferences=[])
    cands = [make_candidate("c1", year=None, is_preprint=True)]
    kept, removals = hard_filter(cands, cfg, Options(include_preprints=False), amap)
    assert kept == []
    assert removals["year"] == 1


# --------------------------------------------------------------------------- #
# Stage 4 - language filter
# --------------------------------------------------------------------------- #
def test_language_drop_known_mismatch_keep_unknown() -> None:
    amap = VenueAliasMap()
    cfg = filter_config(conferences=[])
    cands = [
        make_candidate("en", language="en"),
        make_candidate("en_region", language="en-US"),  # primary subtag matches
        make_candidate("de", language="de"),  # known mismatch -> dropped
        make_candidate("unknown", language=None),  # unknown -> kept
    ]
    kept, removals = hard_filter(cands, cfg, Options(language="en"), amap)
    assert [c.candidate_id for c in kept] == ["en", "en_region", "unknown"]
    assert removals["language"] == 1


# --------------------------------------------------------------------------- #
# Stage 4 - ordering of filters / counts consistency
# --------------------------------------------------------------------------- #
def test_filters_applied_in_order_and_counts_sum() -> None:
    amap = VenueAliasMap()
    cfg = filter_config(conferences=["NeurIPS"], start=2023, end=2025)
    cands = [
        make_candidate("ok", venue_raw="NeurIPS", year=2024, language="en"),
        make_candidate("bad_conf", venue_raw="CVPR", year=2024, language="en"),
        # Bad conference AND bad year: counted once, at the conference filter.
        make_candidate("conf_first", venue_raw="CVPR", year=2000, language="en"),
        make_candidate("bad_year", venue_raw="NeurIPS", year=2000, language="en"),
        make_candidate("bad_lang", venue_raw="NeurIPS", year=2024, language="fr"),
    ]
    kept, removals = hard_filter(cands, cfg, Options(), amap)
    assert [c.candidate_id for c in kept] == ["ok"]
    assert removals["conference"] == 2  # bad_conf, conf_first
    assert removals["year"] == 1  # bad_year
    assert removals["language"] == 1  # bad_lang
    total_removed = sum(removals.values())
    assert total_removed + len(kept) == len(cands)


# --------------------------------------------------------------------------- #
# Stage 6 - threshold
# --------------------------------------------------------------------------- #
def test_threshold_drops_below_keeps_at_boundary() -> None:
    scored = [
        make_scored("a", 0.70, title="Alpha"),
        make_scored("b", 0.50, title="Beta"),  # exactly at threshold -> kept
        make_scored("c", 0.49, title="Gamma"),  # below -> dropped
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=0.50))
    ids = [r.title for r in retained]
    assert ids == ["Alpha", "Beta"]


def test_threshold_none_keeps_all() -> None:
    scored = [
        make_scored("a", 0.10, title="Alpha"),
        make_scored("b", 0.00, title="Beta"),
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    assert len(retained) == 2


# --------------------------------------------------------------------------- #
# Stage 6 - sorting and tie-breaks
# --------------------------------------------------------------------------- #
def test_sort_descending_by_score() -> None:
    scored = [
        make_scored("a", 0.55, title="Low"),
        make_scored("b", 0.95, title="High"),
        make_scored("c", 0.75, title="Mid"),
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    assert [r.title for r in retained] == ["High", "Mid", "Low"]
    assert [r.relevance_score for r in retained] == [0.95, 0.75, 0.55]


def test_tie_break_year_then_citation_then_title() -> None:
    # All equal score 0.80. Expected order:
    #   year desc (null last), then citation desc (null=0), then title asc.
    scored = [
        make_scored("older", 0.80, title="Zeta", year=2023, citation_count=100),
        make_scored("newer_lowcite", 0.80, title="Yota", year=2025, citation_count=1),
        make_scored("newer_hicite", 0.80, title="Xray", year=2025, citation_count=50),
        make_scored("nullyear", 0.80, title="Aaa", year=None, citation_count=999),
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    # 2025 papers first (higher citation first), then 2023, then null-year last.
    assert [r.title for r in retained] == ["Xray", "Yota", "Zeta", "Aaa"]


def test_tie_break_citation_null_treated_as_zero() -> None:
    scored = [
        make_scored("hascite", 0.80, title="Beta", year=2024, citation_count=5),
        make_scored("nullcite", 0.80, title="Alpha", year=2024, citation_count=None),
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    # Same year; citation 5 beats null(=0) despite "Alpha" < "Beta" lexically.
    assert [r.title for r in retained] == ["Beta", "Alpha"]


def test_tie_break_title_lexicographic_final() -> None:
    scored = [
        make_scored("b", 0.80, title="Banana", year=2024, citation_count=10),
        make_scored("a", 0.80, title="Apple", year=2024, citation_count=10),
        make_scored("c", 0.80, title="Cherry", year=2024, citation_count=10),
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    assert [r.title for r in retained] == ["Apple", "Banana", "Cherry"]


def test_tie_break_determinism_independent_of_input_order() -> None:
    base = [
        make_scored("a", 0.80, title="Apple", year=2024, citation_count=10),
        make_scored("b", 0.80, title="Banana", year=2025, citation_count=10),
        make_scored("c", 0.80, title="Cherry", year=2024, citation_count=20),
        make_scored("d", 0.90, title="Date", year=2023, citation_count=1),
    ]
    opts = Options(relevance_threshold=None)
    order_one = [r.title for r in rank_and_select(list(base), opts)]
    order_two = [r.title for r in rank_and_select(list(reversed(base)), opts)]
    assert order_one == order_two
    # And the leading rank-1 is the highest-scoring paper regardless.
    assert order_one[0] == "Date"


# --------------------------------------------------------------------------- #
# Stage 6 - max_papers cap and rank/id assignment
# --------------------------------------------------------------------------- #
def test_max_papers_cap() -> None:
    scored = [make_scored(f"c{i}", 0.90 - i * 0.01, title=f"Paper {i:02d}") for i in range(10)]
    retained = rank_and_select(scored, Options(relevance_threshold=None, max_papers=3))
    assert len(retained) == 3
    assert [r.rank for r in retained] == [1, 2, 3]


def test_paper_id_and_rank_assignment() -> None:
    scored = [
        make_scored("a", 0.90, title="First"),
        make_scored("b", 0.80, title="Second"),
        make_scored("c", 0.70, title="Third"),
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    assert [r.paper_id for r in retained] == ["paper_001", "paper_002", "paper_003"]
    assert [r.rank for r in retained] == [1, 2, 3]


def test_retained_carries_candidate_fields() -> None:
    scored = [
        make_scored(
            "a",
            0.91,
            title="Carrier",
            year=2024,
            doi="10.1/x",
            arxiv_id="2401.00001",
            paper_url="http://example.com/p",
            authors=["Ada", "Bea"],
            matched_domains=["LLM Agents", "RAG"],
            venue_canonical="NeurIPS",
            source="merged",
        )
    ]
    retained = rank_and_select(scored, Options(relevance_threshold=None))
    r = retained[0]
    assert r.title == "Carrier"
    assert r.year == 2024
    assert r.doi == "10.1/x"
    assert r.arxiv_id == "2401.00001"
    assert r.paper_url == "http://example.com/p"
    assert r.authors == ["Ada", "Bea"]
    assert r.domain == ["LLM Agents", "RAG"]
    assert r.conference == "NeurIPS"
    assert r.source == "merged"
    assert r.relevance_score == 0.91


def test_empty_input_returns_empty() -> None:
    assert rank_and_select([], Options()) == []
    amap = VenueAliasMap()
    kept, removals = hard_filter([], filter_config(), Options(), amap)
    assert kept == []
    assert removals == {"conference": 0, "year": 0, "language": 0}

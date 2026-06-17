"""Internal candidate / scoring / retained types (spec Appendix B, S4.3, S6.1).

These are the data the deterministic stages pass between each other. Only the
fields in S4.3 are emitted (via ``record.py``); the rest are internal. All
models are pydantic v2.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

# Provenance of a candidate. Raw candidates carry the discovery source;
# "merged" appears only after deduplication merges records from >1 source.
Source = Literal["papersflow", "paper_search", "merged"]


class PaperCandidate(BaseModel):
    """Normalized discovery result (Appendix B, post-normalization).

    Missing source metadata is represented as ``None`` (EH-4); the candidate
    still proceeds through the pipeline. ``authors``, ``found_by_queries`` are
    source-order collections (never sorted).
    """

    candidate_id: str
    title: str
    authors: list[str] = []
    abstract: str | None = None
    venue_raw: str | None = None
    venue_canonical: str | None = None
    year: int | None = None
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    paper_url: str | None = None
    metadata_code_url: str | None = None
    language: str | None = None
    is_preprint: bool = False
    source: Source
    found_by_queries: list[str] = []


# How strongly a configured domain matches a paper (S6.1.2 rubric).
MatchLevel = Literal["central", "substantial", "peripheral", "absent"]


class DomainMatch(BaseModel):
    """LLM classification of one configured domain against title+abstract (S6.1.1).

    ``evidence_quote`` must be a verbatim substring of the stated
    ``evidence_location``; validation in ``stages/relevance.py`` discards
    invalid quotes and forces ``match_level = "absent"`` (S6.1.3).
    """

    domain: str
    match_level: MatchLevel
    evidence_quote: str | None = None
    evidence_location: Literal["title", "abstract"] | None = None


class RelevanceAssessment(BaseModel):
    """The LLM's anchored classification output for one paper (S6.1.1).

    Contains no numeric score: ``relevance_score`` is computed by code from a
    validated assessment (S6.1.4).
    """

    domain_matches: list[DomainMatch]
    topic_alignment: Literal["direct", "partial", "none", "no_topic_statement"]
    topic_evidence: str | None = None


class ScoredCandidate(BaseModel):
    """A candidate plus its validated assessment and computed score (Appendix B)."""

    candidate: PaperCandidate
    assessment: RelevanceAssessment
    #: The emitted score, rounded to 2 decimals (S6.1.4).
    relevance_score: float
    #: Full-precision (capped, unrounded) score for the S6.3 tie-break rule 1.
    raw_relevance_score: float = 0.0
    matched_domains: list[str] = []


# Outcome of a per-paper PDF acquisition attempt (S9).
PdfStatus = Literal["ok", "failed", "skipped"]


class RetainedPaper(BaseModel):
    """Post-rank record; basis for the emitted YAML record (S4.3).

    Holds the emitted values only -- the canonical emit order lives in
    ``record.py``. ``rank_field_alias_note`` is an inert placeholder kept for
    contract stability and is never emitted.
    """

    paper_id: str
    rank: int
    title: str
    authors: list[str] = []
    conference: str | None = None
    year: int | None = None
    domain: list[str] = []
    paper_url: str | None = None
    pdf_path: str | None = None
    pdf_status: PdfStatus = "skipped"
    code_url: str | None = None
    git_exists: bool = False
    relevance_score: float = 0.0
    rank_field_alias_note: None = None
    doi: str | None = None
    arxiv_id: str | None = None
    source: Source = "merged"
    summary: str = ""
    summary_status: Literal["ok", "failed"] = "ok"

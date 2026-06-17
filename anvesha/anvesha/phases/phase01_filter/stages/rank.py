"""Stage 6 - Rank & Select (deterministic), spec S5 Stage 6 / S6.2-6.3.

Takes the scored candidates from Stage 5 and produces the final ranked set of
``RetainedPaper`` records:

  1. Drop candidates below ``options.relevance_threshold`` (when it is set;
     ``None`` keeps every scored candidate). The comparison is at the candidate's
     stored ``relevance_score`` (S6.2).
  2. Sort by ``relevance_score`` descending, breaking ties with the fully
     deterministic S6.3 chain: higher score, then more recent year (null last),
     then higher citation_count (null treated as 0), then lexicographic
     normalized title ascending. Rule 5 is a total order on distinct papers, so
     the ranking is fully determined and independent of input order.
  3. Cap to ``options.max_papers`` (when > 0).
  4. Assign ``rank`` (1-based) and ``paper_id = "paper_%03d"`` strictly from the
     final ordering, then map each surviving ``ScoredCandidate`` to a
     ``RetainedPaper`` carrying the emitted-field values.

This module performs no I/O, constructs no clients, and imports only sibling
schema modules (charter S12). Stages 7-9 fill in repository/PDF/summary fields on
the returned records afterwards.
"""

from __future__ import annotations

import logging
import unicodedata

from anvesha.phases.phase01_filter.schemas.candidate import (
    RetainedPaper,
    ScoredCandidate,
)
from anvesha.phases.phase01_filter.schemas.config import Options

logger = logging.getLogger(__name__)


def rank_and_select(
    scored: list[ScoredCandidate], options: Options
) -> list[RetainedPaper]:
    """Drop below-threshold candidates, sort per S6.2-6.3, cap, and assign ranks.

    Returns the retained papers in ascending rank order (rank 1 = most
    relevant). ``paper_id`` and ``rank`` are derived solely from the final
    ordering, so identical inputs yield identical IDs (NFR-1).
    """
    threshold = options.relevance_threshold
    if threshold is not None:
        kept = [sc for sc in scored if sc.relevance_score >= threshold]
    else:
        kept = list(scored)

    kept.sort(key=_sort_key)

    if options.max_papers > 0:
        selected = kept[: options.max_papers]
    else:
        selected = kept

    retained: list[RetainedPaper] = []
    for index, sc in enumerate(selected):
        rank = index + 1
        retained.append(_to_retained(sc, rank))

    logger.info(
        "rank_and_select: %d scored -> %d above threshold -> %d retained",
        len(scored),
        len(kept),
        len(retained),
    )
    return retained


def _sort_key(sc: ScoredCandidate) -> tuple[float, float, int, int, int, str]:
    """Ascending sort key implementing the S6.3 tie-break chain (best first).

    Each component is arranged so that a smaller key sorts earlier (rank 1):
      1a. ``-relevance_score`` -> higher (emitted, 2-decimal) score first (S6.2).
      1b. ``-raw_relevance_score`` -> S6.3 rule 1: among equal emitted scores,
          higher FULL-PRECISION (pre-rounding) score first.
      2. year presence + ``-year`` -> more recent first, null year last.
      3. ``-(citation_count or 0)`` -> higher citations first, null treated as 0.
      4. normalized title ascending -> total order independent of input order.
    (S6.3 rule 4 explicitly does NOT use source priority for ordering.)
    """
    candidate = sc.candidate

    # Year: present years sort before a null year (bucket 0 vs 1); within the
    # present bucket, larger year first via negation.
    if candidate.year is None:
        year_bucket = 1
        year_neg = 0
    else:
        year_bucket = 0
        year_neg = -candidate.year

    citation_neg = -(candidate.citation_count or 0)
    title_norm = _normalize_title(candidate.title)

    return (
        -sc.relevance_score,
        -sc.raw_relevance_score,
        year_bucket,
        year_neg,
        citation_neg,
        title_norm,
    )


def _normalize_title(title: str) -> str:
    """Normalize a title for lexicographic tie-breaking (NFC, lower, collapsed ws)."""
    text = unicodedata.normalize("NFC", title).lower()
    return " ".join(text.split())


def _to_retained(sc: ScoredCandidate, rank: int) -> RetainedPaper:
    """Map a ScoredCandidate to a RetainedPaper, copying the emitted-field values.

    Repository (Stage 7), PDF (Stage 8), and summary (Stage 9) fields keep their
    schema defaults here; those stages populate them on the returned record.
    ``domain`` is the matched-domain subset from Stage 5; ``conference`` is the
    canonical venue resolved in Stage 4.
    """
    candidate = sc.candidate
    return RetainedPaper(
        paper_id=f"paper_{rank:03d}",
        rank=rank,
        title=candidate.title,
        authors=list(candidate.authors),
        conference=candidate.venue_canonical,
        year=candidate.year,
        domain=list(sc.matched_domains),
        paper_url=candidate.paper_url,
        relevance_score=sc.relevance_score,
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        source=candidate.source,
    )

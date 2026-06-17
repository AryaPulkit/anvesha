"""Stage 4 - Hard Filter Gate (rule-based), spec S4 / S5 Stage 4.

Applies the three hard filters in the fixed order conference -> year -> language,
recording removals by reason. The filter is deterministic: candidates are
processed in input order, survivors preserve that order, and every comparison is
a total, value-only decision (NFR-1).

Filter semantics (S5 Stage 4):
  1. Conference: resolve the candidate venue to a canonical key (S7.4) and keep
     it only if that key is among the configured conferences. An empty
     ``conferences`` list disables this filter. Preprint exception: an
     unresolved venue is kept when the paper is a preprint and
     ``include_preprints`` is true (it still faces relevance scoring later).
  2. Year: keep when ``years.start <= year <= years.end``. A null year is kept
     only for preprints when ``include_preprints`` is true.
  3. Language: drop only when the language is KNOWN and differs from
     ``options.language``; an unknown (null) language is always kept.

The surviving candidates carry their resolved ``venue_canonical`` (set during the
conference step, including the empty-``conferences`` case so downstream stages and
the emitted record see the canonical key when one exists).

This module imports only sibling schema/venue modules and the workspace-free
venue helpers; it performs no I/O and constructs no clients (charter S12).
"""

from __future__ import annotations

import logging

from anvesha.phases.phase01_filter.schemas.candidate import PaperCandidate
from anvesha.phases.phase01_filter.schemas.config import FilterConfig, Options
from anvesha.phases.phase01_filter.venues.alias_map import VenueAliasMap
from anvesha.phases.phase01_filter.venues.canonicalize import canonicalize_venue

logger = logging.getLogger(__name__)

# Removal-reason keys for the returned counts dict. Fixed strings so callers
# (orchestrator funnel/logging) can rely on them.
_REASON_CONFERENCE = "conference"
_REASON_YEAR = "year"
_REASON_LANGUAGE = "language"


def hard_filter(
    candidates: list[PaperCandidate],
    config: FilterConfig,
    options: Options,
    alias_map: VenueAliasMap,
) -> tuple[list[PaperCandidate], dict[str, int]]:
    """Apply the Stage 4 hard filters in order; return (kept, removals_by_reason).

    ``kept`` preserves input order and contains only candidates that pass all
    enabled filters; each kept candidate has ``venue_canonical`` set to its
    resolved key (or ``None`` when unresolved). ``removals_by_reason`` maps each
    reason (``"conference"``, ``"year"``, ``"language"``) to the number of
    candidates first removed by that filter; the three counts plus
    ``len(kept)`` sum to ``len(candidates)`` (each removed candidate is counted
    once, at the first filter it fails).
    """
    allowed_keys = _allowed_conference_keys(config.conferences, alias_map)
    conference_active = len(allowed_keys) > 0

    kept: list[PaperCandidate] = []
    removals: dict[str, int] = {
        _REASON_CONFERENCE: 0,
        _REASON_YEAR: 0,
        _REASON_LANGUAGE: 0,
    }

    for candidate in candidates:
        # Resolve and record the canonical venue once, regardless of whether the
        # conference filter is active, so survivors carry it downstream.
        canonical = canonicalize_venue(candidate.venue_raw, alias_map)
        candidate.venue_canonical = canonical

        if conference_active and not _passes_conference(
            candidate, canonical, allowed_keys, options
        ):
            removals[_REASON_CONFERENCE] += 1
            continue

        if not _passes_year(candidate, config, options):
            removals[_REASON_YEAR] += 1
            continue

        if not _passes_language(candidate, options):
            removals[_REASON_LANGUAGE] += 1
            continue

        kept.append(candidate)

    logger.info(
        "hard_filter: %d in -> %d kept (removed conference=%d year=%d language=%d)",
        len(candidates),
        len(kept),
        removals[_REASON_CONFERENCE],
        removals[_REASON_YEAR],
        removals[_REASON_LANGUAGE],
    )
    return kept, removals


def _allowed_conference_keys(
    conferences: list[str], alias_map: VenueAliasMap
) -> set[str]:
    """Canonicalize the configured conferences into a set of allowed keys.

    Each configured name is resolved through the same canonicalization the
    candidate venues use, so "NIPS" in the config matches a "NeurIPS" candidate.
    A configured name that does not resolve falls back to its raw string so the
    filter still behaves predictably for venues outside the built-in map.
    An empty ``conferences`` list yields an empty set (filter disabled).
    """
    keys: set[str] = set()
    for raw in conferences:
        resolved = canonicalize_venue(raw, alias_map)
        keys.add(resolved if resolved is not None else raw)
    return keys


def _passes_conference(
    candidate: PaperCandidate,
    canonical: str | None,
    allowed_keys: set[str],
    options: Options,
) -> bool:
    """Conference filter (S5 Stage 4.1).

    Pass when the resolved canonical key is in ``allowed_keys``. When the venue
    is unresolved, apply the preprint exception: keep a preprint candidate if
    ``include_preprints`` is true; otherwise drop it.
    """
    if canonical is not None:
        return canonical in allowed_keys
    # Unresolved venue: preprint exception only.
    return candidate.is_preprint and options.include_preprints


def _passes_year(
    candidate: PaperCandidate, config: FilterConfig, options: Options
) -> bool:
    """Year filter (S5 Stage 4.2).

    Pass when ``years.start <= year <= years.end``. A null year passes only for
    a preprint when ``include_preprints`` is true.
    """
    year = candidate.year
    if year is None:
        return candidate.is_preprint and options.include_preprints
    return config.years.start <= year <= config.years.end


def _passes_language(candidate: PaperCandidate, options: Options) -> bool:
    """Language filter (S5 Stage 4.3).

    Drop only when the language is known and differs from ``options.language``;
    an unknown (null) language always passes. Comparison is case-insensitive on
    the primary subtag, so "en-US" or "EN" matches a target of "en".
    """
    language = candidate.language
    if language is None:
        return True
    return _primary_subtag(language) == _primary_subtag(options.language)


def _primary_subtag(tag: str) -> str:
    """Lowercase primary language subtag of a BCP-47-ish tag ("en-US" -> "en").

    Tolerates the non-standard underscore locale form ("en_US") so an English
    paper tagged that way is not wrongly dropped by the language filter.
    """
    return tag.strip().lower().replace("_", "-").split("-", 1)[0]

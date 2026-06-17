"""Stage 1 - Query Builder (LLM), spec S5 Stage 1.

Generates up to ``max_queries`` diverse, topic-focused search query strings from
the configured ``domains`` (and the optional topic statement). Per S5 Stage 1 the
queries target the topic only: conference and year constraints are NEVER embedded
in query text -- they are applied later as hard filters (Stage 4). Each query is a
short phrase of 2-6 terms and must be meaningfully distinct from the others.

Decoding is greedy (the injected adapter defaults to temperature 0.0) so identical
inputs yield identical queries (NFR-1 determinism).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from anvesha.core.adapters.base import LLMClient

logger = logging.getLogger(__name__)

# JSON Schema the LLM is asked to fill: a flat list of query strings. No scores,
# no metadata -- the model only proposes topical phrasings (anti-hallucination is
# moot here since queries are not classifications, but keeping the output minimal
# and structured maximizes determinism under greedy decoding).
_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
        }
    },
    "required": ["queries"],
}

# A "term" for the 2-6 term rule is a whitespace-delimited token. We count tokens
# on the normalized query (collapsed whitespace), not characters.
_MIN_TERMS = 2
_MAX_TERMS = 6

# Year-like and venue-like tokens we strip if the model leaks them into a query
# (S5 Stage 1: queries are topic-only). A 4-digit token in the 1900-2099 range is
# treated as a year and dropped.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _build_prompt(domains: list[str], topic_statement: str | None, max_queries: int) -> str:
    """Render the deterministic instruction prompt for the query builder."""
    domain_lines = "\n".join(f"- {d}" for d in domains)
    topic_block = (
        f"\nResearcher topic statement (sharpens the queries):\n{topic_statement.strip()}\n"
        if topic_statement and topic_statement.strip()
        else "\n(No topic statement provided; use the domains as the sole topical signal.)\n"
    )
    return (
        "You generate search query strings for an academic literature search.\n"
        "\n"
        "Configured research domains:\n"
        f"{domain_lines}\n"
        f"{topic_block}"
        "\n"
        f"Produce up to {max_queries} distinct search queries. Rules:\n"
        "1. Each query targets the TOPIC only. Do NOT include any conference, "
        "venue, journal, or publication-year in the query text -- those are "
        "applied separately as filters.\n"
        "2. Each query is a short phrase of 2 to 6 terms (words).\n"
        "3. Queries must be meaningfully distinct: cover direct phrasing, "
        "synonyms, sub-topics, and method/dataset angles.\n"
        "4. Return plain topical phrases, no boolean operators, no quotes.\n"
        "Order the queries from most direct to most specialized."
    )


def _normalize_query(raw: str) -> str:
    """Strip leaked years, drop surrounding punctuation, collapse whitespace.

    Returns the cleaned query (possibly empty if nothing topical survives).
    """
    text = _YEAR_RE.sub(" ", raw)
    # Drop boolean glue and stray quoting/bracketing the model may emit; keep the
    # topical words. We do not lowercase (search backends are case-insensitive and
    # preserving the model's casing keeps the output stable and readable).
    text = text.replace('"', " ").replace("'", " ")
    text = re.sub(r"[()\[\]{}]", " ", text)
    # Collapse runs of whitespace to single spaces and trim.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _term_count(query: str) -> int:
    """Number of whitespace-delimited terms in a normalized query."""
    return len(query.split()) if query else 0


def _coerce_to_list(raw: object) -> list[str]:
    """Robustly pull a list of query strings out of the parsed LLM object.

    Accepts the canonical ``{"queries": [...]}`` shape, a bare list, or a dict
    whose first list-valued field holds the queries (tolerant of schema drift).
    Non-string items are coerced via ``str`` and empty/blank items dropped.
    """
    candidate: object = raw
    if isinstance(raw, dict):
        if "queries" in raw:
            candidate = raw["queries"]
        else:
            # Fall back to the first list-valued field, if any.
            candidate = next(
                (v for v in raw.values() if isinstance(v, list)),
                [],
            )
    if not isinstance(candidate, list):
        return []
    out: list[str] = []
    for item in candidate:
        if isinstance(item, str):
            out.append(item)
        elif item is not None:
            out.append(str(item))
    return out


def build_queries(
    llm: LLMClient,
    domains: list[str],
    topic_statement: str | None,
    max_queries: int,
) -> list[str]:
    """Generate up to ``max_queries`` distinct topic-only search queries (S5 Stage 1).

    Each returned query is normalized (years/venue tokens stripped, whitespace
    collapsed) and constrained to 2-6 terms. Queries are de-duplicated
    case-insensitively while preserving the model's emission order; the result is
    capped at ``max_queries``.

    If the LLM yields no usable query (parse failure, all-empty, all over/under
    length), a deterministic fallback is synthesized from the domains so Stage 2
    always has at least one query to run.
    """
    if max_queries <= 0:
        return []

    parsed: dict[str, Any] | None = None
    try:
        parsed = llm.structured_run(
            _build_prompt(domains, topic_statement, max_queries),
            _QUERY_SCHEMA,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to fallback, never crash Stage 1
        logger.warning("query builder LLM call failed; using domain fallback: %s", exc)

    raw_queries = _coerce_to_list(parsed) if parsed is not None else []

    selected: list[str] = []
    seen: set[str] = set()
    for raw in raw_queries:
        if len(selected) >= max_queries:
            break
        query = _normalize_query(raw)
        if not query:
            continue
        if not (_MIN_TERMS <= _term_count(query) <= _MAX_TERMS):
            # Out-of-range term count: skip rather than truncate, to keep queries
            # coherent. Over-long queries lose meaning if mechanically cut.
            logger.debug("dropping query with out-of-range term count: %r", query)
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        selected.append(query)

    if not selected:
        selected = _fallback_queries(domains, max_queries, seen)

    return selected


def _fallback_queries(
    domains: list[str], max_queries: int, seen: set[str]
) -> list[str]:
    """Deterministic topic-only queries derived from the domains (no LLM).

    Used when the model produces nothing usable. Each domain becomes one query
    (clamped to 2-6 terms); domains preserve config order so the fallback is
    byte-stable. A single-word domain is left as-is (1 term) only if no 2-6 term
    query is available, ensuring Stage 2 is never starved.
    """
    out: list[str] = []
    spare: list[str] = []
    for domain in domains:
        if len(out) >= max_queries:
            break
        query = _normalize_query(domain)
        if not query:
            continue
        key = query.casefold()
        if key in seen:
            continue
        terms = _term_count(query)
        if _MIN_TERMS <= terms <= _MAX_TERMS:
            seen.add(key)
            out.append(query)
        elif terms == 1:
            spare.append(query)
        # Domains longer than _MAX_TERMS terms are skipped from the primary pass
        # but never block the search: see the spare fallback below.
    if not out:
        # No domain fit the 2-6 term window. Emit single-term domains (or, failing
        # that, the truncated first domain) so search still runs.
        for query in spare:
            if len(out) >= max_queries:
                break
            key = query.casefold()
            if key not in seen:
                seen.add(key)
                out.append(query)
        if not out and domains:
            words = _normalize_query(domains[0]).split()
            if words:
                out.append(" ".join(words[:_MAX_TERMS]))
    return out

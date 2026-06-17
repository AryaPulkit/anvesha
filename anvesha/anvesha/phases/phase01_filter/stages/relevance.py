"""Stage 5 -- Relevance Scoring (the anti-hallucination core, spec S6.1).

The LLM emits a ``RelevanceAssessment`` consisting of constrained enums plus
VERBATIM evidence quotes only -- never a numeric score (S6.1.1). Code then:

1. validates every quote against the cited source text, forcing the
   conservative ``absent`` / ``none`` value whenever a quote cannot be found
   verbatim (S6.1.3);
2. computes ``relevance_score`` deterministically from the validated
   classifications (S6.1.4).

This guarantees the model cannot inflate relevance on text that is not present:
the worst case of a fabricated quote is that the match is discarded, never that
it raises the score. All comparisons use Unicode NFC + whitespace-collapse so
formatting differences in the quote do not cause false rejections.
"""

from __future__ import annotations

import logging
import re
import unicodedata

from anvesha.phases.phase01_filter.schemas.candidate import (
    DomainMatch,
    PaperCandidate,
    RelevanceAssessment,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

# Points per match level (S6.1.4). Order is fixed; the arithmetic below depends
# on these exact values.
_POINTS: dict[str, float] = {
    "central": 1.00,
    "substantial": 0.70,
    "peripheral": 0.35,
    "absent": 0.00,
}
# Levels that count toward n_strong (the multi-domain bonus, S6.1.4).
_STRONG_LEVELS = frozenset({"central", "substantial"})
# topic_alignment -> multiplicative factor (S6.1.4).
_TOPIC_FACTOR: dict[str, float] = {
    "direct": 1.00,
    "partial": 0.90,
    "none": 0.80,
    "no_topic_statement": 1.00,
}
# Topic alignments that require a verbatim quote (S6.1.3).
_TOPIC_NEEDS_EVIDENCE = frozenset({"direct", "partial"})

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Unicode NFC + collapse all whitespace runs to a single space, trimmed.

    This is the canonical form used for the verbatim-substring check (S6.1.3).
    Both the quote and the source text are normalized identically before
    comparison so that newline/indent/multi-space differences never cause a
    valid quote to be rejected.
    """
    normalized = unicodedata.normalize("NFC", text)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _quote_is_verbatim(
    quote: str | None,
    location: str | None,
    title: str,
    abstract: str | None,
) -> bool:
    """True iff ``quote`` is a non-empty verbatim substring of the cited source.

    ``location`` must be ``"title"`` or ``"abstract"`` and must name a source
    that actually contains the (normalized) quote. A null/empty quote, an
    unknown location, or an absent abstract all fail (S6.1.3).
    """
    if quote is None:
        return False
    norm_quote = _normalize(quote)
    if not norm_quote:
        return False
    if location == "title":
        source = title
    elif location == "abstract":
        source = abstract if abstract is not None else ""
    else:
        # Missing or invalid location cannot be trusted -> reject.
        return False
    return norm_quote in _normalize(source)


def validate_assessment(
    assessment: RelevanceAssessment,
    title: str,
    abstract: str | None,
    domains: list[str],
) -> RelevanceAssessment:
    """Enforce the S6.1.3 evidence-anchoring rule and normalize domain coverage.

    For every ``DomainMatch`` whose ``match_level`` is not ``absent``: if its
    ``evidence_quote`` is not a verbatim substring of the stated location, the
    match is forced to ``absent`` and the quote/location are dropped. The same
    rule applies to ``topic_evidence`` for ``topic_alignment`` in
    ``{direct, partial}`` -- an invalid quote forces ``topic_alignment = none``
    (its evidence is dropped).

    The returned assessment also contains EXACTLY one ``DomainMatch`` per
    configured domain, in ``domains`` (config) order: matches the LLM emitted
    for unknown domains are ignored, and any configured domain the LLM omitted
    is filled in as ``absent``. This guarantees the score computation and
    ``matched_domains`` see a complete, deterministic per-domain picture.

    A new ``RelevanceAssessment`` is returned; the input is not mutated.
    """
    # Index the LLM's matches by domain; first occurrence wins so the result is
    # deterministic even if the model emits a domain twice.
    emitted: dict[str, DomainMatch] = {}
    for match in assessment.domain_matches:
        if match.domain not in emitted:
            emitted[match.domain] = match

    validated_matches: list[DomainMatch] = []
    for domain in domains:
        match = emitted.get(domain)
        if match is None:
            # Domain the model never addressed -> conservative absent (S6.1.3).
            logger.debug("domain %r omitted by model; defaulting to absent", domain)
            validated_matches.append(DomainMatch(domain=domain, match_level="absent"))
            continue
        if match.match_level == "absent":
            # Already conservative; drop any stray evidence to keep records clean.
            validated_matches.append(DomainMatch(domain=domain, match_level="absent"))
            continue
        if _quote_is_verbatim(match.evidence_quote, match.evidence_location, title, abstract):
            validated_matches.append(
                DomainMatch(
                    domain=domain,
                    match_level=match.match_level,
                    evidence_quote=match.evidence_quote,
                    evidence_location=match.evidence_location,
                )
            )
        else:
            # Unverifiable quote -> force absent, discard the quote (S6.1.3).
            logger.debug(
                "domain %r quote not verbatim in %r; forcing absent",
                domain,
                match.evidence_location,
            )
            validated_matches.append(DomainMatch(domain=domain, match_level="absent"))

    # Topic alignment evidence validation (S6.1.3).
    topic_alignment = assessment.topic_alignment
    topic_evidence = assessment.topic_evidence
    if topic_alignment in _TOPIC_NEEDS_EVIDENCE:
        # topic_evidence may be quoted from title or abstract; accept either.
        if not (
            _quote_is_verbatim(topic_evidence, "title", title, abstract)
            or _quote_is_verbatim(topic_evidence, "abstract", title, abstract)
        ):
            logger.debug(
                "topic_evidence not verbatim; forcing topic_alignment none"
            )
            topic_alignment = "none"
            topic_evidence = None
    else:
        # none / no_topic_statement carry no evidence requirement; drop any quote.
        topic_evidence = None

    return RelevanceAssessment(
        domain_matches=validated_matches,
        topic_alignment=topic_alignment,
        topic_evidence=topic_evidence,
    )


def compute_raw_relevance_score(assessment: RelevanceAssessment) -> float:
    """Full-precision (capped at 1.0, UNROUNDED) score from a VALIDATED
    assessment (S6.1.4 arithmetic without the final rounding step).

    This is the value the S6.3 tie-break rule 1 ("higher relevance_score, full
    precision, before rounding") compares; the emitted score rounds it.
    """
    levels = [m.match_level for m in assessment.domain_matches]
    best = max((_POINTS[level] for level in levels), default=0.00)
    n_strong = sum(1 for level in levels if level in _STRONG_LEVELS)
    multi_bonus = 0.05 if n_strong >= 2 else 0.00
    topic_factor = _TOPIC_FACTOR[assessment.topic_alignment]
    raw = (best + multi_bonus) * topic_factor
    return min(1.0, raw)


def compute_relevance_score(assessment: RelevanceAssessment) -> float:
    """Compute the emitted ``relevance_score`` (S6.1.4, exact): the capped raw
    score rounded to 2 decimals.

    Assumes :func:`validate_assessment` has already run, so every
    ``match_level`` reflects only quotes that occur verbatim in the source.
    The arithmetic is deterministic and identical for identical input.
    """
    return round(compute_raw_relevance_score(assessment), 2)


def _matched_domains(assessment: RelevanceAssessment) -> list[str]:
    """Configured domains with a non-absent match, in assessment (config) order."""
    return [m.domain for m in assessment.domain_matches if m.match_level != "absent"]


# JSON Schema for the LLM's structured output. The schema deliberately omits any
# numeric score field so the model cannot emit one (S6.1.1). evidence_quote /
# evidence_location are nullable because they are required only for non-absent
# levels; code (not the schema) enforces that coupling during validation.
_ASSESSMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "domain_matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "domain": {"type": "string"},
                    "match_level": {
                        "type": "string",
                        "enum": ["central", "substantial", "peripheral", "absent"],
                    },
                    "evidence_quote": {"type": ["string", "null"]},
                    "evidence_location": {
                        "type": ["string", "null"],
                        "enum": ["title", "abstract", None],
                    },
                },
                "required": ["domain", "match_level"],
            },
        },
        "topic_alignment": {
            "type": "string",
            "enum": ["direct", "partial", "none", "no_topic_statement"],
        },
        "topic_evidence": {"type": ["string", "null"]},
    },
    "required": ["domain_matches", "topic_alignment"],
}


def _build_prompt(
    candidate: PaperCandidate,
    domains: list[str],
    topic_statement: str | None,
) -> str:
    """Build the greedy-decoded scoring prompt for one candidate (S6.1.1-S6.1.2).

    The prompt instructs the model to (a) classify each configured domain
    against the fixed rubric, (b) emit a VERBATIM quote for any non-absent
    level, and (c) classify topic alignment. It explicitly forbids emitting any
    numeric score and instructs conservative classification (choose the lower
    level when uncertain), per S6.1.1.
    """
    domain_lines = "\n".join(f"  - {d}" for d in domains)
    abstract = candidate.abstract if candidate.abstract is not None else "(no abstract provided)"
    if topic_statement:
        topic_block = (
            "TOPIC STATEMENT (assess topic_alignment against this specific "
            f"research question):\n{topic_statement}\n"
        )
    else:
        topic_block = (
            "TOPIC STATEMENT: none provided. Set topic_alignment to "
            '"no_topic_statement" and topic_evidence to null.\n'
        )
    return (
        "You are a conservative literature-relevance classifier. Assess the "
        "paper below against each configured research domain using ONLY its "
        "title and abstract. Do NOT use outside knowledge.\n\n"
        "RULES:\n"
        "- Emit one entry in domain_matches for EVERY configured domain, in the "
        "given order, echoing the domain string exactly.\n"
        "- match_level rubric: central = the paper's PRIMARY subject is the "
        "domain; substantial = a MAJOR component but not the sole focus; "
        "peripheral = merely mentioned / used as a tool / related-work framing; "
        "absent = neither the domain nor a clear synonym appears.\n"
        "- When uncertain between two levels, choose the LOWER one.\n"
        "- For any level other than absent you MUST provide evidence_quote: an "
        "EXACT, verbatim substring (<= 200 chars) copied character-for-character "
        "from the title or abstract, and evidence_location set to 'title' or "
        "'abstract'. Do NOT paraphrase, summarize, or invent text. If you cannot "
        "quote supporting text verbatim, use match_level = absent.\n"
        "- topic_alignment: direct = addresses the specific topic question; "
        "partial = addresses part of it or a closely adjacent version; none = "
        "in-domain but not the specific question; no_topic_statement = no topic "
        "statement was provided. For direct/partial you MUST provide "
        "topic_evidence as a verbatim quote from the title or abstract.\n"
        "- Do NOT output any numeric score. Output enums and quotes only.\n\n"
        f"CONFIGURED DOMAINS (in order):\n{domain_lines}\n\n"
        f"{topic_block}\n"
        f"PAPER TITLE:\n{candidate.title}\n\n"
        f"PAPER ABSTRACT:\n{abstract}\n"
    )


def score_relevance(
    llm,
    candidates: list[PaperCandidate],
    domains: list[str],
    topic_statement: str | None,
) -> list[ScoredCandidate]:
    """Stage 5: score every candidate via the anti-hallucination pipeline (S5).

    For each candidate the LLM emits a ``RelevanceAssessment`` (enums + quotes,
    no number) via greedy ``structured_run``; code validates the quotes
    (:func:`validate_assessment`), computes the score
    (:func:`compute_relevance_score`), and records the non-absent
    ``matched_domains``. Candidates are processed sequentially in input order so
    the output list is deterministic. The score field on the assessment never
    comes from the model.
    """
    scored: list[ScoredCandidate] = []
    for candidate in candidates:
        prompt = _build_prompt(candidate, domains, topic_statement)
        raw = llm.structured_run(prompt, _ASSESSMENT_SCHEMA)
        assessment = RelevanceAssessment.model_validate(raw)
        validated = validate_assessment(
            assessment, candidate.title, candidate.abstract, domains
        )
        raw_score = compute_raw_relevance_score(validated)
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                assessment=validated,
                relevance_score=round(raw_score, 2),
                raw_relevance_score=raw_score,
                matched_domains=_matched_domains(validated),
            )
        )
    return scored

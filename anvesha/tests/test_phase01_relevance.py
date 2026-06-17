"""Tests for Stage 5 -- Relevance Scoring (spec S6.1, the anti-hallucination core).

Offline only: a fake ``LLMClient`` returns scripted ``RelevanceAssessment``
dicts from ``structured_run`` (the contract Stage 5 uses). No network, no MCP.

Coverage:
- verbatim quote passes through; fabricated/paraphrased quote -> forced absent
  -> not counted in the score;
- the S6.1.5 equivalence table (every row);
- missing-domain backfill to absent, in config order;
- topic_evidence validation forcing topic_alignment none;
- conservative rounding and the [0,1] cap;
- determinism of score_relevance ordering and the no-number invariant.
"""

from __future__ import annotations

from typing import Sequence

from anvesha.core.adapters.base import LLMClient, LLMResponse
from anvesha.phases.phase01_filter.schemas.candidate import (
    DomainMatch,
    PaperCandidate,
    RelevanceAssessment,
)
from anvesha.phases.phase01_filter.stages.relevance import (
    compute_relevance_score,
    score_relevance,
    validate_assessment,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class ScriptedLLM(LLMClient):
    """Returns pre-scripted structured dicts from ``structured_run``, in order.

    ``run`` is unused by Stage 5 but must be concrete (abstract on the base).
    Records every prompt so tests can assert determinism / inputs if needed.
    """

    name = "scripted"

    def __init__(self, scripted: list[dict]) -> None:
        self._scripted = list(scripted)
        self.calls = 0
        self.prompts: list[str] = []

    def run(
        self,
        prompt: str,
        mcps: Sequence = (),
        input_files: Sequence = (),
    ) -> LLMResponse:  # pragma: no cover - not exercised by Stage 5
        raise AssertionError("Stage 5 must use structured_run, not run")

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence = (),
        input_files: Sequence = (),
    ) -> dict:
        self.prompts.append(prompt)
        result = self._scripted[self.calls]
        self.calls += 1
        return result


def _candidate(
    title: str = "A Retrieval-Augmented Generation System for LLM Agents",
    abstract: str | None = "We propose a retrieval-augmented agent that plans.",
    **kw,
) -> PaperCandidate:
    base = dict(candidate_id="c1", title=title, abstract=abstract, source="paper_search")
    base.update(kw)
    return PaperCandidate(**base)


TITLE = "A Retrieval-Augmented Generation System for LLM Agents"
ABSTRACT = "We propose a retrieval-augmented agent that plans over documents."


# --------------------------------------------------------------------------- #
# validate_assessment -- evidence anchoring (S6.1.3)
# --------------------------------------------------------------------------- #
def test_verbatim_quote_passes() -> None:
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="LLM Agents",
                match_level="central",
                evidence_quote="LLM Agents",
                evidence_location="title",
            )
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.domain_matches[0].match_level == "central"
    assert out.domain_matches[0].evidence_quote == "LLM Agents"


def test_fabricated_quote_forced_absent_and_quote_dropped() -> None:
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="LLM Agents",
                match_level="central",
                evidence_quote="reinforcement learning from human feedback",
                evidence_location="abstract",
            )
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.domain_matches[0].match_level == "absent"
    assert out.domain_matches[0].evidence_quote is None
    assert out.domain_matches[0].evidence_location is None
    # A paper whose every match is forced absent scores 0.0.
    assert compute_relevance_score(out) == 0.00


def test_paraphrased_quote_forced_absent() -> None:
    # Real concept, but not a verbatim substring -> rejected (conservative).
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="LLM Agents",
                match_level="substantial",
                evidence_quote="an agent that retrieves and plans",
                evidence_location="abstract",
            )
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.domain_matches[0].match_level == "absent"


def test_quote_wrong_location_rejected() -> None:
    # Quote is in the title, but the model claimed it was in the abstract.
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="LLM Agents",
                match_level="central",
                evidence_quote="Retrieval-Augmented Generation",
                evidence_location="abstract",
            )
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.domain_matches[0].match_level == "absent"


def test_quote_matches_after_whitespace_collapse_and_nfc() -> None:
    title = "Deep   Learning\nfor Vision"
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="Deep Learning",
                match_level="central",
                # Different internal whitespace than the source; must still match.
                evidence_quote="Deep Learning for Vision",
                evidence_location="title",
            )
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, title, None, ["Deep Learning"])
    assert out.domain_matches[0].match_level == "central"


def test_null_and_empty_quote_forced_absent() -> None:
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(domain="LLM Agents", match_level="central", evidence_quote=None),
            DomainMatch(
                domain="RAG",
                match_level="peripheral",
                evidence_quote="   ",
                evidence_location="title",
            ),
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents", "RAG"])
    assert all(m.match_level == "absent" for m in out.domain_matches)


# --------------------------------------------------------------------------- #
# validate_assessment -- domain coverage / config order
# --------------------------------------------------------------------------- #
def test_missing_domain_backfilled_absent_in_config_order() -> None:
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="LLM Agents",
                match_level="central",
                evidence_quote="LLM Agents",
                evidence_location="title",
            )
        ],
        topic_alignment="no_topic_statement",
    )
    domains = ["Retrieval-Augmented Generation", "LLM Agents", "Vision"]
    out = validate_assessment(a, TITLE, ABSTRACT, domains)
    assert [m.domain for m in out.domain_matches] == domains
    assert out.domain_matches[0].match_level == "absent"  # RAG omitted -> absent
    assert out.domain_matches[1].match_level == "central"  # LLM Agents kept
    assert out.domain_matches[2].match_level == "absent"  # Vision omitted -> absent


def test_unknown_domain_emitted_by_model_is_ignored() -> None:
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="Quantum Computing",  # not configured
                match_level="central",
                evidence_quote="LLM Agents",
                evidence_location="title",
            ),
            DomainMatch(
                domain="LLM Agents",
                match_level="substantial",
                evidence_quote="LLM Agents",
                evidence_location="title",
            ),
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert [m.domain for m in out.domain_matches] == ["LLM Agents"]
    assert out.domain_matches[0].match_level == "substantial"


def test_duplicate_domain_first_occurrence_wins() -> None:
    a = RelevanceAssessment(
        domain_matches=[
            DomainMatch(
                domain="LLM Agents",
                match_level="central",
                evidence_quote="LLM Agents",
                evidence_location="title",
            ),
            DomainMatch(domain="LLM Agents", match_level="absent"),
        ],
        topic_alignment="no_topic_statement",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert len(out.domain_matches) == 1
    assert out.domain_matches[0].match_level == "central"


# --------------------------------------------------------------------------- #
# validate_assessment -- topic evidence (S6.1.3)
# --------------------------------------------------------------------------- #
def test_bad_topic_evidence_forces_none() -> None:
    a = RelevanceAssessment(
        domain_matches=[DomainMatch(domain="LLM Agents", match_level="absent")],
        topic_alignment="direct",
        topic_evidence="this is not in the paper",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.topic_alignment == "none"
    assert out.topic_evidence is None


def test_valid_topic_evidence_kept() -> None:
    a = RelevanceAssessment(
        domain_matches=[DomainMatch(domain="LLM Agents", match_level="absent")],
        topic_alignment="partial",
        topic_evidence="retrieval-augmented agent",
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.topic_alignment == "partial"
    assert out.topic_evidence == "retrieval-augmented agent"


def test_none_topic_alignment_drops_stray_evidence() -> None:
    a = RelevanceAssessment(
        domain_matches=[DomainMatch(domain="LLM Agents", match_level="absent")],
        topic_alignment="none",
        topic_evidence="LLM Agents",  # valid text but not required -> dropped
    )
    out = validate_assessment(a, TITLE, ABSTRACT, ["LLM Agents"])
    assert out.topic_alignment == "none"
    assert out.topic_evidence is None


# --------------------------------------------------------------------------- #
# compute_relevance_score -- the S6.1.5 equivalence table (exact)
# --------------------------------------------------------------------------- #
def _assess(levels: dict[str, str], topic: str = "no_topic_statement") -> RelevanceAssessment:
    """Build an already-validated assessment from {domain: level} (no quotes needed)."""
    return RelevanceAssessment(
        domain_matches=[
            DomainMatch(domain=d, match_level=lvl) for d, lvl in levels.items()
        ],
        topic_alignment=topic,
    )


def test_band_central_is_1_00() -> None:
    assert compute_relevance_score(_assess({"D": "central"})) == 1.00


def test_band_two_strong_capped_at_1_00() -> None:
    # best 1.00 + multi_bonus 0.05 = 1.05 -> capped to 1.00.
    a = _assess({"D1": "central", "D2": "substantial"})
    assert compute_relevance_score(a) == 1.00


def test_band_substantial_is_0_70() -> None:
    assert compute_relevance_score(_assess({"D": "substantial"})) == 0.70


def test_band_central_topic_none_is_0_80() -> None:
    a = _assess({"D": "central"}, topic="none")
    assert compute_relevance_score(a) == 0.80


def test_band_peripheral_is_0_35() -> None:
    assert compute_relevance_score(_assess({"D": "peripheral"})) == 0.35


def test_band_all_absent_is_0_00() -> None:
    a = _assess({"D1": "absent", "D2": "absent"})
    assert compute_relevance_score(a) == 0.00


def test_multi_bonus_requires_two_strong() -> None:
    # one substantial + one peripheral -> n_strong == 1 -> no bonus.
    a = _assess({"D1": "substantial", "D2": "peripheral"})
    assert compute_relevance_score(a) == 0.70


def test_two_substantial_get_bonus() -> None:
    # best 0.70 + 0.05 bonus = 0.75.
    a = _assess({"D1": "substantial", "D2": "substantial"})
    assert compute_relevance_score(a) == 0.75


def test_topic_factor_partial() -> None:
    # central 1.00 + no bonus, * 0.90 = 0.90.
    a = _assess({"D": "central"}, topic="partial")
    assert compute_relevance_score(a) == 0.90


def test_topic_factor_direct_no_penalty() -> None:
    a = _assess({"D": "substantial"}, topic="direct")
    assert compute_relevance_score(a) == 0.70


def test_conservative_rounding_peripheral_topic_none() -> None:
    # 0.35 * 0.80 = 0.28 -> round to 0.28.
    a = _assess({"D": "peripheral"}, topic="none")
    assert compute_relevance_score(a) == 0.28


def test_empty_domain_matches_scores_zero() -> None:
    a = RelevanceAssessment(domain_matches=[], topic_alignment="no_topic_statement")
    assert compute_relevance_score(a) == 0.00


# --------------------------------------------------------------------------- #
# score_relevance -- full Stage 5 pipeline with the scripted LLM
# --------------------------------------------------------------------------- #
def test_score_relevance_verbatim_path() -> None:
    llm = ScriptedLLM(
        [
            {
                "domain_matches": [
                    {
                        "domain": "LLM Agents",
                        "match_level": "central",
                        "evidence_quote": "LLM Agents",
                        "evidence_location": "title",
                    },
                    {
                        "domain": "RAG",
                        "match_level": "substantial",
                        "evidence_quote": "Retrieval-Augmented Generation",
                        "evidence_location": "title",
                    },
                ],
                "topic_alignment": "no_topic_statement",
            }
        ]
    )
    cand = _candidate(title=TITLE, abstract=ABSTRACT)
    out = score_relevance(llm, [cand], ["LLM Agents", "RAG"], None)
    assert len(out) == 1
    sc = out[0]
    # central + substantial -> best 1.00 + 0.05 bonus capped = 1.00.
    assert sc.relevance_score == 1.00
    assert sc.matched_domains == ["LLM Agents", "RAG"]
    assert sc.candidate is cand


def test_score_relevance_fabricated_quote_not_counted() -> None:
    llm = ScriptedLLM(
        [
            {
                "domain_matches": [
                    {
                        "domain": "LLM Agents",
                        "match_level": "central",
                        "evidence_quote": "LLM Agents",
                        "evidence_location": "title",
                    },
                    {
                        # Fabricated quote -> forced absent -> not in matched_domains.
                        "domain": "RAG",
                        "match_level": "central",
                        "evidence_quote": "graph neural network",
                        "evidence_location": "abstract",
                    },
                ],
                "topic_alignment": "no_topic_statement",
            }
        ]
    )
    cand = _candidate(title=TITLE, abstract=ABSTRACT)
    out = score_relevance(llm, [cand], ["LLM Agents", "RAG"], None)
    sc = out[0]
    assert sc.matched_domains == ["LLM Agents"]  # RAG dropped
    # Only one strong domain survives -> best 1.00, no multi bonus -> 1.00.
    assert sc.relevance_score == 1.00
    rag = next(m for m in sc.assessment.domain_matches if m.domain == "RAG")
    assert rag.match_level == "absent" and rag.evidence_quote is None


def test_score_relevance_all_absent_scores_zero() -> None:
    llm = ScriptedLLM(
        [
            {
                "domain_matches": [
                    {"domain": "LLM Agents", "match_level": "absent"},
                ],
                "topic_alignment": "no_topic_statement",
            }
        ]
    )
    cand = _candidate(title="An Unrelated Paper About Bridges", abstract="Concrete.")
    out = score_relevance(llm, [cand], ["LLM Agents"], None)
    assert out[0].relevance_score == 0.00
    assert out[0].matched_domains == []


def test_score_relevance_never_uses_model_number() -> None:
    # Even if the model smuggles a high score field, code recomputes from enums.
    llm = ScriptedLLM(
        [
            {
                "relevance_score": 0.99,  # ignored
                "domain_matches": [
                    {"domain": "LLM Agents", "match_level": "peripheral",
                     "evidence_quote": "agent", "evidence_location": "abstract"},
                ],
                "topic_alignment": "no_topic_statement",
            }
        ]
    )
    cand = _candidate(title=TITLE, abstract=ABSTRACT)
    out = score_relevance(llm, [cand], ["LLM Agents"], None)
    assert out[0].relevance_score == 0.35  # peripheral, not 0.99


def test_score_relevance_topic_statement_validated() -> None:
    llm = ScriptedLLM(
        [
            {
                "domain_matches": [
                    {"domain": "LLM Agents", "match_level": "central",
                     "evidence_quote": "LLM Agents", "evidence_location": "title"},
                ],
                "topic_alignment": "direct",
                "topic_evidence": "totally fabricated topic sentence",
            }
        ]
    )
    cand = _candidate(title=TITLE, abstract=ABSTRACT)
    out = score_relevance(llm, [cand], ["LLM Agents"], "Do agents plan over docs?")
    # Bad topic evidence -> none -> topic_factor 0.80 -> 1.00 * 0.80 = 0.80.
    assert out[0].assessment.topic_alignment == "none"
    assert out[0].relevance_score == 0.80


def test_score_relevance_deterministic_and_ordered() -> None:
    scripts = [
        {
            "domain_matches": [
                {"domain": "LLM Agents", "match_level": "central",
                 "evidence_quote": "LLM Agents", "evidence_location": "title"}
            ],
            "topic_alignment": "no_topic_statement",
        },
        {
            "domain_matches": [{"domain": "LLM Agents", "match_level": "absent"}],
            "topic_alignment": "no_topic_statement",
        },
    ]
    c1 = _candidate(title=TITLE, abstract=ABSTRACT)
    c2 = _candidate(candidate_id="c2", title="Bridges", abstract="Concrete.")
    out_a = score_relevance(ScriptedLLM(scripts), [c1, c2], ["LLM Agents"], None)
    out_b = score_relevance(ScriptedLLM(scripts), [c1, c2], ["LLM Agents"], None)
    assert [s.relevance_score for s in out_a] == [1.00, 0.00]
    assert [s.relevance_score for s in out_a] == [s.relevance_score for s in out_b]
    # Output order mirrors input order (no reordering in Stage 5).
    assert [s.candidate.candidate_id for s in out_a] == ["c1", "c2"]

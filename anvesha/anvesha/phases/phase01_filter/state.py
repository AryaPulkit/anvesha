"""Phase 1 run state -- intermediate stage outputs + resolved config (spec S5).

``PhaseState`` is the single mutable object the :class:`FilterOrchestrator`
threads through Stages 0-11. It carries the resolved :class:`FilterConfig` /
:class:`Options` / topic statement / output-directory name decided at Stage 0,
the :class:`FilterCounts` funnel updated at each stage boundary, and the
intermediate collections produced by each stage (queries, candidates, scored
candidates, retained papers). Keeping all of this in one place makes the stage
sequence explicit and lets a future checkpoint hook snapshot/restore a run from
a single object.

This module performs no I/O, constructs no clients, and imports only sibling
schema modules (charter S12). It is a plain dataclass, not a pydantic model:
``PaperCandidate``/``ScoredCandidate``/``RetainedPaper`` are already validated
pydantic models, so the container needs no further validation and a dataclass
keeps mutation cheap and explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from anvesha.phases.phase01_filter.schemas.candidate import (
    PaperCandidate,
    RetainedPaper,
    ScoredCandidate,
)
from anvesha.phases.phase01_filter.schemas.config import FilterConfig, Options
from anvesha.phases.phase01_filter.schemas.counts import FilterCounts


@dataclass
class PhaseState:
    """Mutable per-run state for the Phase 1 pipeline (spec S5).

    The resolved-config fields (``config``, ``options``, ``topic_statement``,
    ``output_dir_name``, ``generated_at``) are fixed at Stage 0 and never change.
    The intermediate collections and ``counts`` are populated/updated as the
    orchestrator advances through the stages.
    """

    # --- resolved configuration (fixed at Stage 0) -----------------------
    config: FilterConfig
    options: Options
    topic_statement: str | None
    #: Versioned output directory name, workspace-relative (S4.1), e.g.
    #: ``"filtered_literature_v1_llm_agents_2023_2025"``.
    output_dir_name: str
    #: Single ISO-8601 UTC timestamp for this run, derived once at run start so
    #: the manifest is internally consistent (spec: caller owns the clock).
    generated_at: str

    # --- funnel counts (updated at each stage boundary) ------------------
    counts: FilterCounts = field(default_factory=FilterCounts)

    # --- intermediate stage outputs --------------------------------------
    #: Stage 1 search queries.
    queries: list[str] = field(default_factory=list)
    #: Stage 2 raw candidates (the ``identified`` set).
    raw_candidates: list[PaperCandidate] = field(default_factory=list)
    #: Stage 3 deduplicated candidates (the ``after_deduplication`` set).
    deduped_candidates: list[PaperCandidate] = field(default_factory=list)
    #: Stage 4 survivors (the ``after_hard_filter`` set).
    filtered_candidates: list[PaperCandidate] = field(default_factory=list)
    #: Per-reason hard-filter removal counts (S5 Stage 4), for the run log.
    hard_filter_removals: dict[str, int] = field(default_factory=dict)
    #: Stage 5 scored candidates (the ``screened`` set).
    scored_candidates: list[ScoredCandidate] = field(default_factory=list)
    #: Stage 6 retained papers (the ``retained`` set), ascending rank order.
    retained_papers: list[RetainedPaper] = field(default_factory=list)

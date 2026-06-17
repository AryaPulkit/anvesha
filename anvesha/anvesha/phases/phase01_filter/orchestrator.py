"""Phase 1 stage orchestration (spec S5 Stages 1-10, S10 error handling).

``FilterOrchestrator`` drives the deterministic, linear stage sequence and owns
the PRISMA funnel (``FilterCounts``). It does NOT do filesystem I/O or build the
resolved config: Stage 0 (config) and Stage 11 (version log) live in
``pipeline.py`` (the :class:`Phase` entry point), which constructs a
:class:`PhaseState` and hands it here. The orchestrator runs the data-flow
stages (1-9), updates the funnel at every boundary, and produces the assembled
``01_filtered_literature.md`` text (Stage 10) via ``stages.assemble`` -- the
caller persists that text.

Boundary rules (charter S12): the orchestrator only consumes the injected
``LLMClient`` and :class:`ToolRouter`; it constructs no clients and touches the
workspace only through the injected ``ResearchProject`` passed to the PDF stage
(which itself uses ``workspace.phase_io``). It imports nothing from other phase
packages.

Error handling (spec S10):
  * EH-3 (no discovery source) propagates as a fatal ``AnveshaError`` from
    Stage 2 -- the orchestrator does not catch it.
  * EH-2 (one source degraded), EH-4 (missing metadata), EH-5 (PDF failure),
    EH-6 (summary failure), EH-10 (malformed code URL) are handled inside the
    individual stages and are non-fatal here.
  * EH-7 (zero retained) is handled here: the orchestrator emits the guided
    empty-result document instead of crashing.

Determinism (NFR-1): every stage is deterministic given fixed LLM/tool
responses; the orchestrator adds no clock, randomness, or dict-iteration-order
dependence. ``generated_at`` is taken from the state (fixed once at run start).
"""

from __future__ import annotations

import logging
import unicodedata

from anvesha.core.adapters.base import LLMClient
from anvesha.phases.phase01_filter.schemas.candidate import (
    PaperCandidate,
    RetainedPaper,
)
from anvesha.phases.phase01_filter.state import PhaseState
from anvesha.phases.phase01_filter.tools import ToolRouter
from anvesha.phases.phase01_filter.venues.alias_map import VenueAliasMap
# Import each stage callable by its full module path. We import the callables
# (not the stage modules) deliberately: the stages package re-exports some
# callables under names that collide with submodule names (e.g. the
# ``hard_filter`` function vs the ``hard_filter`` module), so referencing
# functions directly avoids any package-attribute shadowing.
from anvesha.phases.phase01_filter.stages.assemble import (
    build_output_document,
    empty_result_document,
)
from anvesha.phases.phase01_filter.stages.dedup import deduplicate
from anvesha.phases.phase01_filter.stages.hard_filter import hard_filter
from anvesha.phases.phase01_filter.stages.pdf_fetch import fetch_pdf
from anvesha.phases.phase01_filter.stages.query_builder import build_queries
from anvesha.phases.phase01_filter.stages.rank import rank_and_select
from anvesha.phases.phase01_filter.stages.relevance import score_relevance
from anvesha.phases.phase01_filter.stages.repo_detect import detect_repository
from anvesha.phases.phase01_filter.stages.search import search_candidates
from anvesha.phases.phase01_filter.stages.summarize import summarize_paper
from anvesha.workspace.project import ResearchProject

logger = logging.getLogger(__name__)


class FilterOrchestrator:
    """Runs Stages 1-9 + assembles Stage 10; maintains the funnel (spec S5).

    Constructed with the injected dependencies and a Stage-0-resolved
    :class:`PhaseState`. Call :meth:`run` to advance through every stage and get
    back the final document text; the funnel lives on ``state.counts`` after the
    call so the caller can record/inspect it.
    """

    def __init__(
        self,
        llm: LLMClient,
        router: ToolRouter,
        project: ResearchProject,
        state: PhaseState,
        alias_map: VenueAliasMap | None = None,
    ) -> None:
        self.llm = llm
        self.router = router
        self.project = project
        self.state = state
        # The alias map is built-in data with no I/O; construct a default when
        # the caller does not supply one (deterministic, fixed built-in order).
        self.alias_map = alias_map if alias_map is not None else VenueAliasMap()

    # -- checkpoint hooks (S5 Checkpointing) ------------------------------
    # Per-stage checkpointing is specified (S5) but deferred for Phase 1: these
    # hooks are intentional no-ops so the stage loop already has the seams a
    # future checkpoint/resume implementation will fill in. They are called
    # after each stage completes; today they only log progress.
    def _checkpoint(self, stage: str) -> None:
        """No-op checkpoint hook (S5). Records stage completion in the log only."""
        logger.debug("stage complete: %s", stage)

    # -- main entry -------------------------------------------------------
    def run(self) -> str:
        """Execute Stages 1-10 and return the ``01_filtered_literature.md`` text.

        Updates ``self.state`` (intermediate collections + ``counts``) in place.
        Raises only on a fatal condition surfaced by a stage (notably EH-3 from
        Stage 2); every per-paper / per-source degradation is non-fatal.
        """
        state = self.state
        config = state.config
        options = state.options

        # Stage 1 -- Query Builder (LLM).
        state.queries = build_queries(
            self.llm,
            config.domains,
            state.topic_statement,
            options.max_queries,
        )
        logger.info("stage 1: built %d queries", len(state.queries))
        self._checkpoint("query_builder")

        # Stage 2 -- Multi-Source Search (MCP). Raises AnveshaError (EH-3) if no
        # discovery source is available; that is intentionally fatal.
        state.raw_candidates = search_candidates(
            self.router, state.queries, options.max_results_per_source
        )
        state.counts.identified = len(state.raw_candidates)
        logger.info("stage 2: identified %d raw candidates", state.counts.identified)
        self._checkpoint("search")

        # Stage 3 -- Normalize & Deduplicate (deterministic).
        state.deduped_candidates = deduplicate(
            state.raw_candidates, options.dedup_threshold
        )
        state.counts.after_deduplication = len(state.deduped_candidates)
        logger.info(
            "stage 3: %d after deduplication", state.counts.after_deduplication
        )
        self._checkpoint("dedup")

        # Stage 4 -- Hard Filter Gate (rule-based).
        kept, removals = hard_filter(
            state.deduped_candidates, config, options, self.alias_map
        )
        state.filtered_candidates = kept
        state.hard_filter_removals = removals
        state.counts.after_hard_filter = len(kept)
        logger.info(
            "stage 4: %d after hard filter (removals=%s)",
            state.counts.after_hard_filter,
            removals,
        )
        self._checkpoint("hard_filter")

        # Stage 5 -- Relevance Scoring (LLM).
        state.scored_candidates = score_relevance(
            self.llm,
            state.filtered_candidates,
            config.domains,
            state.topic_statement,
        )
        state.counts.screened = len(state.scored_candidates)
        logger.info("stage 5: screened %d candidates", state.counts.screened)
        self._checkpoint("relevance")

        # Stage 6 -- Rank & Select (deterministic).
        state.retained_papers = rank_and_select(state.scored_candidates, options)
        state.counts.retained = len(state.retained_papers)
        logger.info("stage 6: retained %d papers", state.counts.retained)
        self._checkpoint("rank")

        # EH-7: zero retained -> guided empty-result document, do not crash.
        if not state.retained_papers:
            logger.warning(
                "stage 6: zero papers retained; emitting guided empty-result "
                "document (EH-7)"
            )
            return self._empty_result_document()

        # Pair each retained paper with the source candidate it was built from,
        # so Stages 7 (repo detection) and 8 (PDF) can read its metadata.
        pairing = self._pair_retained_with_candidates(
            state.retained_papers, state.scored_candidates
        )

        # Stage 7 -- Repository Detection (metadata-only, deterministic).
        for paper in state.retained_papers:
            candidate = pairing[paper.paper_id]
            detect_repository(paper, candidate)
        state.counts.with_code = sum(1 for p in state.retained_papers if p.git_exists)
        logger.info("stage 7: %d papers with code links", state.counts.with_code)
        self._checkpoint("repo_detect")

        # Stage 8 -- PDF Acquisition (MCP). Non-fatal per paper (EH-5).
        for paper in state.retained_papers:
            pdf_relpath = f"{state.output_dir_name}/pdfs/{paper.paper_id}.pdf"
            fetch_pdf(self.router, paper, self.project, pdf_relpath, options)
        state.counts.pdfs_downloaded = sum(
            1 for p in state.retained_papers if p.pdf_status == "ok"
        )
        state.counts.pdfs_failed = sum(
            1 for p in state.retained_papers if p.pdf_status == "failed"
        )
        logger.info(
            "stage 8: %d PDFs downloaded, %d failed",
            state.counts.pdfs_downloaded,
            state.counts.pdfs_failed,
        )
        self._checkpoint("pdf_fetch")

        # Stage 9 -- Summary Generation (LLM). Non-fatal per paper (EH-6).
        for paper in state.retained_papers:
            candidate = pairing[paper.paper_id]
            summarize_paper(self.llm, paper, candidate.abstract, options)
        self._checkpoint("summarize")

        # Stage 10 -- Output Assembly (deterministic). The caller persists this.
        document = build_output_document(
            counts=state.counts,
            filters_dict=config.model_dump(),
            papers=state.retained_papers,
            output_dir=state.output_dir_name,
            generated_at=state.generated_at,
        )
        self._checkpoint("assemble")
        return document

    # -- helpers ----------------------------------------------------------
    def _empty_result_document(self) -> str:
        """Render the EH-7 guided empty-result document (S10).

        The relax recommendation targets the filter that most plausibly caused
        the empty result, inferred from the funnel: if the conference gate (when
        active) or year gate eliminated everything, recommend the hard filters;
        otherwise the relevance threshold is the likely culprit.
        """
        state = self.state
        return empty_result_document(
            counts=state.counts,
            filters_dict=state.config.model_dump(),
            output_dir=state.output_dir_name,
            generated_at=state.generated_at,
            relax_hint=self._relax_hint(),
        )

    def _relax_hint(self) -> str:
        """Pick which filter to recommend relaxing (S10/EH-7, deterministic).

        If hard filtering removed everything (nothing reached screening),
        recommend the conference/year filters; otherwise (candidates were
        scored but all fell below the cut) recommend the relevance threshold.
        """
        counts = self.state.counts
        if counts.after_hard_filter == 0 and counts.after_deduplication > 0:
            # Everything died at the hard-filter gate. Point at whichever gate is
            # active; conferences first since it is the strictest common cause.
            if self.state.config.conferences:
                return "conferences"
            return "years"
        if counts.identified == 0:
            # Nothing was even discovered; widening the topical query / domains
            # is the only lever, but the spec's named levers are filters -- the
            # most actionable is broadening the year range.
            return "years"
        return "relevance_threshold"

    @staticmethod
    def _pair_retained_with_candidates(
        retained: list[RetainedPaper],
        scored: list,
    ) -> dict[str, PaperCandidate]:
        """Map each ``RetainedPaper.paper_id`` to its source ``PaperCandidate``.

        Stage 6 builds each ``RetainedPaper`` from a ``ScoredCandidate`` but does
        not keep a back-reference, so we re-derive the pairing deterministically
        from a stable identity key (DOI / arXiv id / normalized title + source).
        Titles are unique after dedup (S6.3 note), so this key is collision-free
        for the retained set; a defensive fallback handles any unexpected
        collision by skipping already-claimed candidates.
        """
        by_key: dict[tuple, PaperCandidate] = {}
        for sc in scored:
            key = FilterOrchestrator._identity_key(sc.candidate)
            by_key.setdefault(key, sc.candidate)

        pairing: dict[str, PaperCandidate] = {}
        for paper in retained:
            key = FilterOrchestrator._identity_key_from_paper(paper)
            candidate = by_key.get(key)
            if candidate is None:
                # Should not happen for well-formed input; fall back to a title
                # match so repo/PDF/summary still see real metadata rather than
                # crashing the run.
                candidate = next(
                    (
                        sc.candidate
                        for sc in scored
                        if FilterOrchestrator._norm_title(sc.candidate.title)
                        == FilterOrchestrator._norm_title(paper.title)
                    ),
                    None,
                )
            if candidate is None:
                logger.warning(
                    "could not pair retained paper %s with a source candidate; "
                    "using an empty placeholder", paper.paper_id
                )
                candidate = PaperCandidate(
                    candidate_id=paper.paper_id,
                    title=paper.title,
                    source=paper.source,
                )
            pairing[paper.paper_id] = candidate
        return pairing

    @staticmethod
    def _identity_key(candidate: PaperCandidate) -> tuple:
        """Stable identity for a candidate (DOI/arXiv/title + source)."""
        return (
            (candidate.doi or "").strip().lower(),
            (candidate.arxiv_id or "").strip().lower(),
            FilterOrchestrator._norm_title(candidate.title),
            candidate.source,
        )

    @staticmethod
    def _identity_key_from_paper(paper: RetainedPaper) -> tuple:
        """Same identity tuple computed from the retained record's mirrored fields."""
        return (
            (paper.doi or "").strip().lower(),
            (paper.arxiv_id or "").strip().lower(),
            FilterOrchestrator._norm_title(paper.title),
            paper.source,
        )

    @staticmethod
    def _norm_title(title: str) -> str:
        """NFC + lowercase + whitespace-collapse, matching the rank tie-break norm."""
        text = unicodedata.normalize("NFC", title).lower()
        return " ".join(text.split())

"""Phase 1 processing stages (spec S5 Stages 1-10).

Each stage is a small, focused module exposing the public callable(s) the
orchestrator drives. This package re-exports those callables for convenience;
sibling modules should still import by full module path per the project import
convention (e.g. ``from anvesha.phases.phase01_filter.stages.rank import
rank_and_select``).
"""

from anvesha.phases.phase01_filter.stages.assemble import (
    build_output_document,
    empty_result_document,
    validate_document,
)
from anvesha.phases.phase01_filter.stages.dedup import (
    deduplicate,
    normalize_title,
    title_similarity,
)
from anvesha.phases.phase01_filter.stages.hard_filter import hard_filter
from anvesha.phases.phase01_filter.stages.pdf_fetch import fetch_pdf
from anvesha.phases.phase01_filter.stages.query_builder import build_queries
from anvesha.phases.phase01_filter.stages.rank import rank_and_select
from anvesha.phases.phase01_filter.stages.relevance import (
    compute_relevance_score,
    score_relevance,
    validate_assessment,
)
from anvesha.phases.phase01_filter.stages.repo_detect import detect_repository
from anvesha.phases.phase01_filter.stages.search import (
    normalize_result,
    search_candidates,
)
from anvesha.phases.phase01_filter.stages.summarize import summarize_paper

__all__ = [
    "build_output_document",
    "empty_result_document",
    "validate_document",
    "deduplicate",
    "normalize_title",
    "title_similarity",
    "hard_filter",
    "fetch_pdf",
    "build_queries",
    "rank_and_select",
    "compute_relevance_score",
    "score_relevance",
    "validate_assessment",
    "detect_repository",
    "normalize_result",
    "search_candidates",
    "summarize_paper",
]

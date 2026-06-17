"""Phase 1 data-layer schemas (spec S3, S4.3, S6.1, Appendix B).

Re-exports the public schema types and record serializers so other Phase 1
modules can import them from one place.
"""

from anvesha.phases.phase01_filter.schemas.candidate import (
    DomainMatch,
    MatchLevel,
    PaperCandidate,
    PdfStatus,
    RelevanceAssessment,
    RetainedPaper,
    ScoredCandidate,
    Source,
)
from anvesha.phases.phase01_filter.schemas.config import (
    FilterConfig,
    Options,
    YearRange,
    build_filter_config,
    build_options,
)
from anvesha.phases.phase01_filter.schemas.counts import FilterCounts
from anvesha.phases.phase01_filter.schemas.record import (
    COUNTS_FIELD_ORDER,
    GENERATOR,
    MANIFEST_FIELD_ORDER,
    PAPER_FIELD_ORDER,
    SCHEMA_VERSION,
    manifest_counts,
    render_document,
    render_manifest_block,
    render_paper_block,
)

__all__ = [
    "DomainMatch",
    "MatchLevel",
    "PaperCandidate",
    "PdfStatus",
    "RelevanceAssessment",
    "RetainedPaper",
    "ScoredCandidate",
    "Source",
    "FilterConfig",
    "Options",
    "YearRange",
    "build_filter_config",
    "build_options",
    "FilterCounts",
    "COUNTS_FIELD_ORDER",
    "GENERATOR",
    "MANIFEST_FIELD_ORDER",
    "PAPER_FIELD_ORDER",
    "SCHEMA_VERSION",
    "manifest_counts",
    "render_document",
    "render_manifest_block",
    "render_paper_block",
]

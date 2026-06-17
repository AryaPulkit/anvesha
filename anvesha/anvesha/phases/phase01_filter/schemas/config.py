"""Input schema for Phase 1 (Filter Literature), spec S3.

``FilterConfig`` is the required filter block (S3.1); ``Options`` are the
optional operational parameters (S3.2). Both are pydantic v2 models so the
orchestrator gets eager validation with precise messages (EH-1). The
``build_*`` helpers are tolerant constructors that accept the loose ``dict``
shapes coming out of the loaded pipeline config.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class YearRange(BaseModel):
    """Inclusive publication-year bound (S3.1). ``start <= end`` is enforced."""

    start: int
    end: int

    @model_validator(mode="after")
    def _check_order(self) -> "YearRange":
        if self.start > self.end:
            # EH-1: invalid config is fatal; raise a ValueError so pydantic
            # surfaces it as a ValidationError the orchestrator can map to
            # ConfigError.
            raise ValueError(
                f"years.start ({self.start}) must be <= years.end ({self.end})"
            )
        return self


class FilterConfig(BaseModel):
    """Required filter block (S3.1).

    ``conferences`` is a hard filter; an empty list disables venue filtering.
    ``domains`` is the soft topical signal and is required. ``years`` is a
    hard inclusive range.
    """

    conferences: list[str] = []
    domains: list[str]
    years: YearRange


class Options(BaseModel):
    """Optional operational parameters (S3.2, plus ``overwrite`` per S4.1/EH-8).

    Every field has a default, so the required filter block alone is enough to
    run. ``relevance_threshold`` may be ``None`` to keep all scored papers.
    """

    max_papers: int = 40
    relevance_threshold: float | None = 0.50
    dedup_threshold: float = 0.90
    max_results_per_source: int = 100
    max_queries: int = 6
    include_preprints: bool = True
    pdf_download: bool = True
    pdf_max_retries: int = 3
    summary_max_retries: int = 2
    output_root: str = "."
    output_version: int = 1
    dirname_include_domain: bool = True
    dirname_include_years: bool = True
    language: str = "en"
    overwrite: bool = False


def build_filter_config(filters: dict[str, Any]) -> FilterConfig:
    """Build a :class:`FilterConfig` from a loose ``filters`` dict.

    Tolerant of the two ``years`` shapes seen in configs: a ``{"start": ...,
    "end": ...}`` mapping (S3.1) and an already-built :class:`YearRange`.
    Raises a pydantic ``ValidationError`` (which the orchestrator maps to
    ``ConfigError``, EH-1) on bad input.
    """
    return FilterConfig.model_validate(filters)


def build_options(options: dict[str, Any] | None) -> Options:
    """Build an :class:`Options` from a loose ``options`` dict (or ``None``).

    Unknown keys are ignored only insofar as pydantic ignores extras by
    default; missing keys fall back to the S3.2 defaults.
    """
    return Options.model_validate(options or {})

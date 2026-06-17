"""Serializers for the emitted artifact ``01_filtered_literature.md`` (spec S4.2-4.4).

The document is a run-manifest YAML frontmatter block followed by one block per
retained paper (YAML frontmatter in the exact S4.3 field order, then a
``## LLM Summary`` heading and the summary body). Emission is deterministic and
byte-stable (NFR-1): a small hand-rolled YAML emitter gives us exact control
over field order, scalar quoting and ``null`` literals while still producing
output that parses as valid YAML under the S4.4 grammar.

``generated_at`` is supplied by the caller -- this module never reads the clock,
so callers own determinism (NFR-1/NFR-3).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from anvesha.phases.phase01_filter.schemas.candidate import RetainedPaper
from anvesha.phases.phase01_filter.schemas.counts import FilterCounts

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"
GENERATOR = "anvesha.phase01_filter"

# Canonical paper-record field order (S4.3). Authoritative emit order.
PAPER_FIELD_ORDER: tuple[str, ...] = (
    "paper_id",
    "title",
    "authors",
    "conference",
    "year",
    "domain",
    "paper_url",
    "pdf_path",
    "pdf_status",
    "code_url",
    "git_exists",
    "relevance_score",
    "rank",
    "doi",
    "arxiv_id",
    "source",
)

# Canonical run-manifest field order (S4.3).
MANIFEST_FIELD_ORDER: tuple[str, ...] = (
    "document_type",
    "schema_version",
    "generated_at",
    "generator",
    "filters",
    "counts",
    "papers_total",
    "output_dir",
)

# Canonical ``counts`` sub-field order (S4.3).
COUNTS_FIELD_ORDER: tuple[str, ...] = (
    "identified",
    "after_deduplication",
    "after_hard_filter",
    "screened",
    "retained",
    "pdfs_downloaded",
    "pdfs_failed",
    "with_code",
)

# A scalar may be emitted unquoted only if it is a "plain" YAML string that
# round-trips to itself. Anything that could be mis-parsed (numbers, bools,
# null, special indicators, leading/trailing space, control chars) is quoted.
_PLAIN_SAFE = re.compile(r"^[^\s\"'#&*!|>%@`,\[\]{}:][^\n\t]*$")
# Tokens that are NOT safe as plain scalars because YAML would type them.
_RESERVED_PLAIN = {
    "null",
    "Null",
    "NULL",
    "~",
    "true",
    "True",
    "TRUE",
    "false",
    "False",
    "FALSE",
    "yes",
    "Yes",
    "YES",
    "no",
    "No",
    "NO",
    "on",
    "On",
    "ON",
    "off",
    "Off",
    "OFF",
    "",
}
_NUMERIC = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _quote(value: str) -> str:
    """Emit ``value`` as a double-quoted YAML scalar with minimal escaping."""
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _scalar(value: Any) -> str:
    """Render a leaf value as a single-line YAML scalar.

    ``None`` -> ``null``; bools -> ``true``/``false``; ints kept as-is; strings
    emitted plain when safe, otherwise double-quoted. Used for every leaf so
    arbitrary titles/URLs/abstracts never break the YAML grammar (S4.4).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if (
        text in _RESERVED_PLAIN
        or _NUMERIC.match(text)
        or " : " in text
        or text.endswith(":")
        or ": " in text
        or text != text.strip()
        or not _PLAIN_SAFE.match(text)
    ):
        return _quote(text)
    return text


def _emit_field(key: str, value: Any, lines: list[str]) -> None:
    """Append YAML lines for ``key: value`` (lists block-style, scalars inline)."""
    if isinstance(value, list):
        if not value:
            lines.append(f"{key}: []")
            return
        lines.append(f"{key}:")
        for item in value:
            lines.append(f"  - {_scalar(item)}")
        return
    lines.append(f"{key}: {_scalar(value)}")


def manifest_counts(counts: FilterCounts) -> dict[str, int]:
    """Return ``counts`` as an ordered dict in the S4.3 sub-field order."""
    dumped = counts.model_dump()
    return {field: int(dumped[field]) for field in COUNTS_FIELD_ORDER}


def _emit_filters(filters: dict[str, Any], lines: list[str]) -> None:
    """Emit the nested ``filters`` mapping (conferences, domains, years) at S4.3 order."""
    lines.append("filters:")
    # conferences (list)
    conferences = list(filters.get("conferences", []) or [])
    if conferences:
        lines.append("  conferences:")
        for item in conferences:
            lines.append(f"    - {_scalar(item)}")
    else:
        lines.append("  conferences: []")
    # domains (list)
    domains = list(filters.get("domains", []) or [])
    if domains:
        lines.append("  domains:")
        for item in domains:
            lines.append(f"    - {_scalar(item)}")
    else:
        lines.append("  domains: []")
    # years (mapping start/end)
    years = filters.get("years", {}) or {}
    if hasattr(years, "model_dump"):
        years = years.model_dump()
    lines.append("  years:")
    lines.append(f"    start: {_scalar(years.get('start'))}")
    lines.append(f"    end: {_scalar(years.get('end'))}")


def _emit_counts(counts: dict[str, int], lines: list[str]) -> None:
    """Emit the nested ``counts`` mapping in S4.3 sub-field order."""
    lines.append("counts:")
    for field in COUNTS_FIELD_ORDER:
        lines.append(f"  {field}: {int(counts[field])}")


def render_manifest_block(manifest: dict[str, Any]) -> str:
    """Render the run-manifest YAML frontmatter block (without trailing summary).

    ``manifest`` carries: ``generated_at`` (str), ``filters`` (dict or
    FilterConfig-ish), ``counts`` (FilterCounts or dict), ``output_dir`` (str),
    and ``papers_total`` (int). ``document_type``/``schema_version``/
    ``generator`` are fixed literals.
    """
    counts = manifest["counts"]
    if isinstance(counts, FilterCounts):
        counts_map = manifest_counts(counts)
    else:
        counts_map = {field: int(counts[field]) for field in COUNTS_FIELD_ORDER}

    filters = manifest["filters"]
    if hasattr(filters, "model_dump"):
        filters = filters.model_dump()

    lines: list[str] = ["---"]
    for field in MANIFEST_FIELD_ORDER:
        if field == "document_type":
            lines.append("document_type: filtered_literature_index")
        elif field == "schema_version":
            lines.append(f"schema_version: {_quote(SCHEMA_VERSION)}")
        elif field == "generated_at":
            lines.append(f"generated_at: {_scalar(manifest['generated_at'])}")
        elif field == "generator":
            lines.append(f"generator: {GENERATOR}")
        elif field == "filters":
            _emit_filters(filters, lines)
        elif field == "counts":
            _emit_counts(counts_map, lines)
        elif field == "papers_total":
            lines.append(f"papers_total: {int(manifest['papers_total'])}")
        elif field == "output_dir":
            lines.append(f"output_dir: {_scalar(manifest['output_dir'])}")
    lines.append("---")
    return "\n".join(lines)


def _sanitize_summary(summary: str) -> str:
    """Neutralize any line that is exactly ``---`` (the S4.4 record fence).

    Stage 9 summaries are free LLM text; a line that strips to ``---`` would be
    read as a record boundary by a spec-compliant consumer (S4.4:
    ``SUMMARY_TEXT := free text until the next line that is exactly "---"``),
    corrupting the document. Such lines are replaced with em-dashes - visually
    a rule, never the literal fence. Deterministic.
    """
    return "\n".join(
        "———" if line.strip() == "---" else line
        for line in summary.split("\n")
    )


def render_paper_block(paper: RetainedPaper) -> str:
    """Render one paper record: YAML frontmatter (S4.3 order) + summary section."""
    lines: list[str] = ["---"]
    for field in PAPER_FIELD_ORDER:
        if field == "relevance_score":
            lines.append(f"relevance_score: {paper.relevance_score:.2f}")
            continue
        value = getattr(paper, field)
        _emit_field(field, value, lines)
    lines.append("---")
    block = "\n".join(lines)
    summary = _sanitize_summary(paper.summary) if paper.summary else ""
    return f"{block}\n\n## LLM Summary\n\n{summary}"


def render_document(manifest: dict[str, Any], papers: list[RetainedPaper]) -> str:
    """Render the complete ``01_filtered_literature.md`` text (S4.2-4.4).

    Run manifest first, then one paper block per retained paper in the given
    order (the caller passes them in ascending rank). The result is fully
    deterministic for fixed inputs (NFR-1). A trailing newline terminates the
    file.
    """
    blocks: list[str] = [render_manifest_block(manifest)]
    for paper in papers:
        blocks.append(render_paper_block(paper))
    return "\n\n".join(blocks) + "\n"

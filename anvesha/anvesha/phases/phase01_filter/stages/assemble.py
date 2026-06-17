"""Stage 10 -- Output Assembly (spec S4, S10/EH-7).

This module is a thin, deterministic wrapper over ``record.render_document``
(the byte-stable serializer built by the data layer). It owns:

* ``build_output_document`` -- assemble the run manifest dict and render the
  full ``01_filtered_literature.md`` for the normal (>=1 retained paper) case.
* ``empty_result_document`` -- the EH-7 guided empty-result document: a run
  manifest with ``papers_total = 0`` followed by a ``## Notes`` section that
  reports the funnel and recommends which filter to relax.
* ``validate_document`` -- check the S4.5 output-validity exit criteria against
  the emitted text and return the list of violations ([] when valid).

Determinism (NFR-1): ``generated_at`` is always supplied by the caller; this
module never reads the clock. Field order and scalar emission come entirely
from ``record.py``, so identical inputs yield byte-identical output.
"""

from __future__ import annotations

import logging
from typing import Any

import yaml

from anvesha.phases.phase01_filter.schemas.candidate import RetainedPaper
from anvesha.phases.phase01_filter.schemas.counts import FilterCounts
from anvesha.phases.phase01_filter.schemas import record

logger = logging.getLogger(__name__)

# Counts that form the monotonic funnel (S4.5), in non-increasing order.
_FUNNEL_FIELDS: tuple[str, ...] = (
    "identified",
    "after_deduplication",
    "after_hard_filter",
    "screened",
    "retained",
)

# S4.3 paper-record fields that must be present on every record (S4.5).
_REQUIRED_PAPER_FIELDS: tuple[str, ...] = record.PAPER_FIELD_ORDER


def _build_manifest(
    counts: FilterCounts,
    filters_dict: dict[str, Any],
    output_dir: str,
    generated_at: str,
    papers_total: int,
) -> dict[str, Any]:
    """Assemble the manifest dict consumed by ``record.render_manifest_block``.

    ``papers_total`` is passed explicitly so the empty-result path can force 0
    independent of how many records (none) follow.
    """
    return {
        "generated_at": generated_at,
        "filters": filters_dict,
        "counts": counts,
        "papers_total": int(papers_total),
        "output_dir": output_dir,
    }


def build_output_document(
    counts: FilterCounts,
    filters_dict: dict[str, Any],
    papers: list[RetainedPaper],
    output_dir: str,
    generated_at: str,
) -> str:
    """Render the full ``01_filtered_literature.md`` for the normal case (S4.2-4.4).

    ``papers`` must already be in ascending ``rank`` order (Stage 6 output).
    Returns the complete file text; ``papers_total`` in the manifest is set to
    ``len(papers)``.
    """
    manifest = _build_manifest(
        counts=counts,
        filters_dict=filters_dict,
        output_dir=output_dir,
        generated_at=generated_at,
        papers_total=len(papers),
    )
    return record.render_document(manifest, papers)


def empty_result_document(
    counts: FilterCounts,
    filters_dict: dict[str, Any],
    output_dir: str,
    generated_at: str,
    relax_hint: str,
) -> str:
    """Render the EH-7 guided empty-result document (S10).

    Emits the run manifest with ``papers_total = 0`` (no paper records) followed
    by a ``## Notes`` section that reports the funnel and recommends which filter
    to relax. ``relax_hint`` is the caller-chosen recommendation (typically one
    of ``conferences``, ``years`` or ``relevance_threshold``); it is surfaced
    verbatim. The document never crashes a downstream YAML parse: the manifest is
    a valid frontmatter block and the Notes section is plain markdown after it.
    """
    manifest = _build_manifest(
        counts=counts,
        filters_dict=filters_dict,
        output_dir=output_dir,
        generated_at=generated_at,
        papers_total=0,
    )
    manifest_block = record.render_manifest_block(manifest)
    notes = _render_notes(counts, relax_hint)
    return f"{manifest_block}\n\n{notes}\n"


def _render_notes(counts: FilterCounts, relax_hint: str) -> str:
    """Render the ``## Notes`` section for the empty-result document (S10/EH-7).

    Reports the funnel (identified -> retained) line by line and the relax
    recommendation. Deterministic: funnel fields in fixed S4.5 order.
    """
    dumped = counts.model_dump()
    lines: list[str] = ["## Notes", ""]
    lines.append(
        "No papers were retained: every candidate was eliminated during hard "
        "filtering or fell below the relevance threshold (EH-7)."
    )
    lines.append("")
    lines.append("Funnel:")
    lines.append(f"- identified: {int(dumped['identified'])}")
    lines.append(f"- after_deduplication: {int(dumped['after_deduplication'])}")
    lines.append(f"- after_hard_filter: {int(dumped['after_hard_filter'])}")
    lines.append(f"- screened: {int(dumped['screened'])}")
    lines.append(f"- retained: {int(dumped['retained'])}")
    lines.append("")
    lines.append(f"Recommendation: relax {relax_hint}.")
    return "\n".join(lines)


def _split_yaml_blocks(content: str) -> list[str]:
    """Return the raw text of each ``---``-delimited YAML block, in order.

    Parses per the S4.4 grammar: a block opens on a line that is exactly ``---``
    and closes on the next line that is exactly ``---``. Text between a closing
    fence and the next opening fence (summaries, the Notes section) is ignored
    here -- validation only inspects the YAML blocks.
    """
    blocks: list[str] = []
    current: list[str] | None = None
    for line in content.splitlines():
        if line == "---":
            if current is None:
                current = []
            else:
                blocks.append("\n".join(current))
                current = None
            continue
        if current is not None:
            current.append(line)
    return blocks


def _parse_block(raw: str) -> dict[str, Any] | None:
    """Parse one YAML block to a mapping, or ``None`` if it is not a mapping."""
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def validate_document(content: str) -> list[str]:
    """Return the S4.5 exit-criteria violations for ``content`` ([] if valid).

    Checks, in order:

    1. A run manifest is present and well-formed (first YAML block, has
       ``document_type``, no ``paper_id``).
    2. ``papers_total`` equals the number of paper records that follow.
    3. ``counts`` are monotonic:
       ``identified >= after_deduplication >= after_hard_filter >= screened >= retained``.
    4. Every paper record contains all S4.3 fields.
    5. ``rank`` values form the contiguous ascending sequence ``1..retained``.
    6. ``git_exists == (code_url != null)`` for every record (S8 invariant).

    Does not touch the filesystem (the S4.5 ``pdf_path`` existence check belongs
    to the orchestrator, which knows the output directory). Returns a flat,
    deterministically ordered list of human-readable violation strings.
    """
    errors: list[str] = []

    blocks = _split_yaml_blocks(content)
    if not blocks:
        return ["no YAML frontmatter blocks found; run manifest missing"]

    manifest = _parse_block(blocks[0])
    if manifest is None:
        return ["run manifest block is not a valid YAML mapping"]
    if "document_type" not in manifest or "paper_id" in manifest:
        errors.append(
            "first YAML block is not a run manifest "
            "(expected document_type and no paper_id)"
        )
    if manifest.get("document_type") != "filtered_literature_index":
        errors.append(
            "run manifest document_type is not 'filtered_literature_index'"
        )

    # Paper records are every YAML block after the manifest.
    record_blocks = blocks[1:]
    records: list[dict[str, Any]] = []
    for index, raw in enumerate(record_blocks):
        parsed = _parse_block(raw)
        if parsed is None:
            errors.append(f"paper record {index + 1} is not a valid YAML mapping")
            continue
        records.append(parsed)

    # 2. papers_total matches the record count.
    papers_total = manifest.get("papers_total")
    if not isinstance(papers_total, int) or isinstance(papers_total, bool):
        errors.append("manifest papers_total is missing or not an integer")
    elif papers_total != len(record_blocks):
        errors.append(
            f"papers_total ({papers_total}) != number of paper records "
            f"({len(record_blocks)})"
        )

    # 3. counts monotonic.
    errors.extend(_validate_counts(manifest))

    # 4. every record has all S4.3 fields.
    for index, rec in enumerate(records):
        missing = [f for f in _REQUIRED_PAPER_FIELDS if f not in rec]
        if missing:
            errors.append(
                f"paper record {index + 1} missing required field(s): "
                f"{', '.join(missing)}"
            )

    # 5. rank contiguous 1..N ascending.
    errors.extend(_validate_ranks(records))

    # 6. git_exists == (code_url != null).
    errors.extend(_validate_git_invariant(records))

    return errors


def _validate_counts(manifest: dict[str, Any]) -> list[str]:
    """Check the funnel monotonicity criterion (S4.5)."""
    errors: list[str] = []
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        return ["manifest counts mapping is missing or malformed"]
    values: list[int] = []
    for field in _FUNNEL_FIELDS:
        value = counts.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"counts.{field} is missing or not an integer")
            return errors
        values.append(value)
    for left, right, lname, rname in zip(
        values, values[1:], _FUNNEL_FIELDS, _FUNNEL_FIELDS[1:]
    ):
        if left < right:
            errors.append(
                f"counts not monotonic: {lname} ({left}) < {rname} ({right})"
            )
    return errors


def _validate_ranks(records: list[dict[str, Any]]) -> list[str]:
    """Check that ranks are the contiguous ascending sequence 1..N (S4.5)."""
    errors: list[str] = []
    ranks: list[int] = []
    for index, rec in enumerate(records):
        value = rec.get("rank")
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"paper record {index + 1} has a non-integer rank")
            return errors
        ranks.append(value)
    expected = list(range(1, len(records) + 1))
    if ranks != expected:
        errors.append(
            f"rank values are not the contiguous ascending sequence "
            f"{expected!r}; got {ranks!r}"
        )
    return errors


def _validate_git_invariant(records: list[dict[str, Any]]) -> list[str]:
    """Check ``git_exists == (code_url != null)`` for every record (S8 invariant)."""
    errors: list[str] = []
    for index, rec in enumerate(records):
        if "git_exists" not in rec or "code_url" not in rec:
            # Missing-field violation is already reported by the S4.3 check.
            continue
        git_exists = rec.get("git_exists")
        code_url = rec.get("code_url")
        if bool(git_exists) != (code_url is not None):
            errors.append(
                f"paper record {index + 1} violates git_exists invariant: "
                f"git_exists={git_exists!r} but code_url={code_url!r}"
            )
    return errors

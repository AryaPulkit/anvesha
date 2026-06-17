"""Tests for Stage 10 output-assembly helpers (spec S4, S10/EH-7, S4.5).

Offline: no network, no MCP, no LLM. Builds RetainedPaper / FilterCounts inputs
directly and exercises build_output_document, empty_result_document and the
validate_document exit-criteria checks.
"""

from __future__ import annotations

import yaml

from anvesha.phases.phase01_filter.schemas.candidate import RetainedPaper
from anvesha.phases.phase01_filter.schemas.counts import FilterCounts
from anvesha.phases.phase01_filter.stages import assemble


GENERATED_AT = "2026-01-15T09:42:11Z"
OUTPUT_DIR = "filtered_literature_v1_llm_agents_2023_2025"
FILTERS = {
    "conferences": ["NeurIPS", "ICML", "CVPR"],
    "domains": ["LLM Agents", "Retrieval-Augmented Generation"],
    "years": {"start": 2023, "end": 2025},
}


def _counts(**overrides: int) -> FilterCounts:
    base = dict(
        identified=412,
        after_deduplication=318,
        after_hard_filter=96,
        screened=96,
        retained=2,
        pdfs_downloaded=1,
        pdfs_failed=1,
        with_code=1,
    )
    base.update(overrides)
    return FilterCounts(**base)


def _paper(rank: int, **overrides: object) -> RetainedPaper:
    base: dict[str, object] = dict(
        paper_id=f"paper_{rank:03d}",
        rank=rank,
        title=f"Paper Number {rank}",
        authors=["A. Researcher", "B. Scientist"],
        conference="NeurIPS",
        year=2024,
        domain=["LLM Agents"],
        paper_url="https://example.org/abs/0001",
        pdf_path=f"pdfs/paper_{rank:03d}.pdf",
        pdf_status="ok",
        code_url="https://github.com/example/repo",
        git_exists=True,
        relevance_score=0.9,
        doi="10.0000/x",
        arxiv_id="2401.00001",
        source="merged",
        summary="A short implementation-focused summary.",
        summary_status="ok",
    )
    base.update(overrides)
    return RetainedPaper(**base)


def _two_papers() -> list[RetainedPaper]:
    return [
        _paper(1),
        _paper(
            2,
            pdf_path=None,
            pdf_status="failed",
            code_url=None,
            git_exists=False,
            relevance_score=0.88,
            doi=None,
            source="paper_search",
        ),
    ]


# --- build_output_document -------------------------------------------------


def test_build_output_document_is_valid_and_passes_exit_criteria() -> None:
    doc = assemble.build_output_document(
        counts=_counts(),
        filters_dict=FILTERS,
        papers=_two_papers(),
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )
    assert assemble.validate_document(doc) == []
    # Manifest content sanity.
    blocks = doc.split("---\n")
    assert "document_type: filtered_literature_index" in doc
    assert "papers_total: 2" in doc
    assert doc.count("## LLM Summary") == 2


def test_build_output_document_is_deterministic() -> None:
    args = dict(
        counts=_counts(),
        filters_dict=FILTERS,
        papers=_two_papers(),
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )
    assert assemble.build_output_document(**args) == assemble.build_output_document(
        **args
    )


def test_build_output_document_papers_total_tracks_record_count() -> None:
    doc = assemble.build_output_document(
        counts=_counts(retained=1, after_hard_filter=96, screened=96),
        filters_dict=FILTERS,
        papers=[_paper(1)],
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )
    manifest = yaml.safe_load(doc.split("---\n")[1])
    assert manifest["papers_total"] == 1
    assert assemble.validate_document(doc) == []


# --- empty_result_document (EH-7) ------------------------------------------


def test_empty_result_document_has_notes_zero_total_and_valid_manifest() -> None:
    counts = _counts(
        after_hard_filter=0, screened=0, retained=0, pdfs_downloaded=0,
        pdfs_failed=0, with_code=0,
    )
    doc = assemble.empty_result_document(
        counts=counts,
        filters_dict=FILTERS,
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
        relax_hint="conferences",
    )
    # Notes section present with the relax recommendation.
    assert "## Notes" in doc
    assert "relax conferences" in doc
    assert "Funnel:" in doc
    # papers_total == 0 and no paper records.
    assert "papers_total: 0" in doc
    assert "paper_id" not in doc
    # The manifest still parses as a valid YAML mapping.
    manifest = yaml.safe_load(doc.split("---\n")[1])
    assert manifest["document_type"] == "filtered_literature_index"
    assert manifest["papers_total"] == 0
    # And the document passes the exit criteria (0 records, papers_total 0).
    assert assemble.validate_document(doc) == []


def test_empty_result_document_is_deterministic() -> None:
    args = dict(
        counts=_counts(after_hard_filter=0, screened=0, retained=0),
        filters_dict=FILTERS,
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
        relax_hint="years",
    )
    assert assemble.empty_result_document(**args) == assemble.empty_result_document(
        **args
    )
    assert "relax years" in assemble.empty_result_document(**args)


def test_empty_result_document_reports_full_funnel() -> None:
    doc = assemble.empty_result_document(
        counts=_counts(
            identified=412, after_deduplication=318, after_hard_filter=0,
            screened=0, retained=0,
        ),
        filters_dict=FILTERS,
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
        relax_hint="relevance_threshold",
    )
    assert "identified: 412" in doc
    assert "after_deduplication: 318" in doc
    assert "after_hard_filter: 0" in doc
    assert "retained: 0" in doc


# --- validate_document: each violation detected ----------------------------


def _valid_doc() -> str:
    return assemble.build_output_document(
        counts=_counts(),
        filters_dict=FILTERS,
        papers=_two_papers(),
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )


def test_validate_empty_content_reports_missing_manifest() -> None:
    errors = assemble.validate_document("")
    assert errors
    assert any("run manifest" in e or "frontmatter" in e for e in errors)


def test_validate_detects_papers_total_mismatch() -> None:
    doc = _valid_doc().replace("papers_total: 2", "papers_total: 5")
    errors = assemble.validate_document(doc)
    assert any("papers_total" in e for e in errors)


def test_validate_detects_non_monotonic_counts() -> None:
    # Make after_deduplication exceed identified.
    doc = _valid_doc().replace("identified: 412", "identified: 100")
    errors = assemble.validate_document(doc)
    assert any("monotonic" in e for e in errors)


def test_validate_detects_missing_required_field() -> None:
    # Drop the `source:` line from the first record only.
    doc = _valid_doc().replace("source: merged\n", "", 1)
    errors = assemble.validate_document(doc)
    assert any("missing required field" in e and "source" in e for e in errors)


def test_validate_detects_non_contiguous_rank() -> None:
    # Re-rank the second record to 3 (gap), so 1..2 expected but [1, 3].
    doc = _valid_doc().replace("rank: 2", "rank: 3")
    errors = assemble.validate_document(doc)
    assert any("contiguous ascending" in e for e in errors)


def test_validate_detects_descending_rank() -> None:
    # Swap so ranks read [2, 1] in document order.
    paper_a = _paper(2)
    paper_b = _paper(1, code_url=None, git_exists=False)
    doc = assemble.build_output_document(
        counts=_counts(),
        filters_dict=FILTERS,
        papers=[paper_a, paper_b],
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )
    errors = assemble.validate_document(doc)
    assert any("contiguous ascending" in e for e in errors)


def test_validate_detects_git_invariant_violation() -> None:
    # code_url present but git_exists false.
    doc = _valid_doc().replace("git_exists: true", "git_exists: false", 1)
    errors = assemble.validate_document(doc)
    assert any("git_exists invariant" in e for e in errors)


def test_validate_detects_git_invariant_violation_null_url_true_flag() -> None:
    # The second record has code_url null + git_exists false (valid). Flip the
    # flag to true to violate.
    doc = _valid_doc().replace("git_exists: false", "git_exists: true", 1)
    errors = assemble.validate_document(doc)
    assert any("git_exists invariant" in e for e in errors)


def test_validate_single_paper_doc_passes() -> None:
    doc = assemble.build_output_document(
        counts=_counts(retained=1, after_hard_filter=96, screened=96),
        filters_dict=FILTERS,
        papers=[_paper(1)],
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )
    assert assemble.validate_document(doc) == []


def test_validate_counts_equal_is_monotonic() -> None:
    # Equality is allowed (>=), e.g. after_hard_filter == screened.
    doc = assemble.build_output_document(
        counts=_counts(
            identified=10, after_deduplication=10, after_hard_filter=2,
            screened=2, retained=2,
        ),
        filters_dict=FILTERS,
        papers=_two_papers(),
        output_dir=OUTPUT_DIR,
        generated_at=GENERATED_AT,
    )
    assert assemble.validate_document(doc) == []

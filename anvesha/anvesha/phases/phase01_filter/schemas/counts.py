"""PRISMA funnel counts (spec S4.3 ``counts``, Appendix B FilterCounts).

The orchestrator maintains these across stages; they are emitted as the
``counts`` mapping in the run manifest. Field order here is the S4.3 sub-field
order so a plain ``model_dump`` already produces the emit order, though
``record.manifest_counts`` is the canonical serializer.
"""

from __future__ import annotations

from pydantic import BaseModel


class FilterCounts(BaseModel):
    """The funnel: identified -> ... -> retained, plus PDF/code tallies (S4.3)."""

    identified: int = 0
    after_deduplication: int = 0
    after_hard_filter: int = 0
    screened: int = 0
    retained: int = 0
    pdfs_downloaded: int = 0
    pdfs_failed: int = 0
    with_code: int = 0

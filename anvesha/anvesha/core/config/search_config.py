"""Adaptive web-search limits for Judge agents
(research_gap_pipeline_implementation.md S7.7)."""

from __future__ import annotations

from pydantic import BaseModel


class SearchConfig(BaseModel):
    max_search_rounds: int = 3
    max_queries_per_round: int = 2
    max_total_queries_per_judge_call: int = 5
    skip_search_at_iteration: int = 3

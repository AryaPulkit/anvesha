"""Iteration limits and scoring thresholds for review loops
(research_gap_pipeline_implementation.md S10)."""

from __future__ import annotations

from pydantic import BaseModel


class RubricConfig(BaseModel):
    max_tactical_iterations: int = 3
    max_strategic_iterations: int = 2
    max_total_iterations: int = 5
    top_n_gaps_for_solution_loop: int = 5
    gap_approval_score_threshold: float = 7.0
    gap_continue_score_threshold: float = 5.0
    solution_approval_score_threshold: float = 7.0
    solution_continue_score_threshold: float = 5.0

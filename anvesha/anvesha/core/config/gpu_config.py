"""GPU node configuration (research_gap_pipeline_implementation.md S13.1)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ModelNodeAssignment(BaseModel):
    """Manual model-role -> CUDA node assignment (S13.3)."""

    generator_nodes: list[int] = Field(default_factory=list)
    critic_nodes: list[int] = Field(default_factory=list)
    analyzer_nodes: list[int] = Field(default_factory=list)
    judge_nodes: list[int] = Field(default_factory=list)
    merger_nodes: list[int] = Field(default_factory=list)


class GpuConfig(BaseModel):
    mode: Literal["auto", "manual", "single"] = "auto"
    execution_mode: Literal["sequential", "parallel"] = "sequential"
    #: Min and max CUDA device IDs to consider (inclusive).
    available_node_range: tuple[int, int] = (0, 7)
    vram_headroom_gb: float = 8.0
    manual_assignments: ModelNodeAssignment | None = None
    single_node_id: int = 0

    @model_validator(mode="after")
    def _validate(self) -> "GpuConfig":
        lo, hi = self.available_node_range
        if lo < 0 or lo > hi:
            raise ValueError(
                f"available_node_range must be (min, max) with 0 <= min <= max, "
                f"got {self.available_node_range}"
            )
        if self.mode == "manual" and self.manual_assignments is None:
            raise ValueError('gpu.mode "manual" requires gpu.manual_assignments')
        return self

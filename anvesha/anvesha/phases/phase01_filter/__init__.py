"""Phase 1 -- Filter Literature (spec v2.0; charter S7, Appendix C).

Public entry point: :class:`FilterLiteraturePipeline`. Importing this package
registers the pipeline in :data:`anvesha.phases.PHASE_REGISTRY` under key ``1``
so the CLI / orchestrator can locate and run it (registration side-effect).

Boundary compliance (charter S12): this package imports nothing from other phase
packages; all workspace I/O flows through the workspace module; the LLM and MCP
clients are injected.
"""

from __future__ import annotations

from anvesha.phases import PHASE_REGISTRY
from anvesha.phases.phase01_filter.pipeline import FilterLiteraturePipeline

# Register the Phase 1 implementation (charter S7 registry). Idempotent: a
# re-import simply re-binds the same class.
PHASE_REGISTRY[1] = FilterLiteraturePipeline

__all__ = ["FilterLiteraturePipeline"]

"""Phase modules - one package per phase (ANVESHA.md S3).

Boundary rule (S12.1): phase modules never import from other phase modules;
phase outputs travel via workspace Markdown files.

As each phase is implemented, its module registers its ``Phase`` subclass in
``PHASE_REGISTRY`` keyed by phase id, so the CLI can locate and run it. The
registry is empty until phases are built; the CLI reports unimplemented
phases gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anvesha.core.phase import Phase

PHASE_REGISTRY: dict[int, type["Phase"]] = {}

# Register implemented phases by importing them for their side-effect. Each
# implemented phase package binds itself into PHASE_REGISTRY on import. This is
# placed AFTER PHASE_REGISTRY is defined so the registering import sees it.
# Unimplemented phases are simply absent from the registry (the CLI reports
# them gracefully); only implemented phases are imported here. A plain import
# is used so a genuine bug in an implemented phase surfaces rather than being
# masked.
from anvesha.phases import phase01_filter  # noqa: E402,F401  (registration side-effect)

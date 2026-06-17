"""GPU node manager: probe, allocate, and startup health check (S13.2, S13.6)."""

from __future__ import annotations

import logging

from anvesha.core.adapters.base import LLMClient
from anvesha.core.config import GpuConfig
from anvesha.core.exceptions import GpuNodeUnavailableError, InsufficientVramError
from anvesha.core.gpu.allocator import allocate
from anvesha.core.gpu.probe import GpuDevice, probe_devices

logger = logging.getLogger(__name__)


class GpuManager:
    """Owns the probed device list and the role -> node assignments.

    When ``devices`` is None the manager probes nvidia-smi on construction;
    tests inject a synthetic device list instead.
    """

    def __init__(self, config: GpuConfig, devices: list[GpuDevice] | None = None) -> None:
        self.config = config
        self.devices: list[GpuDevice] = (
            devices if devices is not None else probe_devices(config.available_node_range)
        )
        self.assignments: dict[str, list[int]] = {}

    def allocate(self, role_vram: dict[str, float]) -> dict[str, list[int]]:
        """Compute and store role -> node assignments (see allocator.allocate)."""
        self.assignments = allocate(role_vram, self.config, self.devices)
        return self.assignments

    def health_check(
        self,
        adapters: dict[str, LLMClient],
        run_test_inference: bool = False,
    ) -> None:
        """Startup health check per S13.6.

        1. Every assigned node must exist among the probed devices
           (:class:`GpuNodeUnavailableError` with node id and reason).
        2. The free VRAM of each role's node group must cover that role's
           requirement - checked largest requirement first
           (:class:`InsufficientVramError`).
        3. Optionally run a one-token test inference per assigned adapter;
           failures are wrapped in :class:`GpuNodeUnavailableError`.
        """
        by_id = {d.device_id: d for d in self.devices}

        for role, nodes in self.assignments.items():
            for node in nodes:
                if node not in by_id:
                    raise GpuNodeUnavailableError(
                        f"node {node} assigned to role {role!r} was not found "
                        f"among probed devices {sorted(by_id)}"
                    )

        checked = [
            (role, adapter.reported_vram_gb or 0.0)
            for role, adapter in adapters.items()
            if self.assignments.get(role)
        ]
        for role, vram in sorted(checked, key=lambda item: -item[1]):
            nodes = self.assignments[role]
            free = sum(by_id[n].free_vram_gb for n in nodes)
            if vram > free:
                raise InsufficientVramError(
                    f"role {role!r} needs {vram:.1f}GB but its node group "
                    f"{nodes} has only {free:.1f}GB free"
                )

        if run_test_inference:
            for role, adapter in adapters.items():
                if not self.assignments.get(role):
                    continue
                try:
                    adapter.run("ping")
                except Exception as exc:
                    raise GpuNodeUnavailableError(
                        f"test inference failed for role {role!r} "
                        f"(adapter {adapter.name!r}) on nodes "
                        f"{self.assignments[role]}: {exc}"
                    ) from exc

        logger.info(
            "GPU readiness: %d device(s), assignments %s, test inference %s",
            len(self.devices),
            self.assignments,
            "passed" if run_test_inference else "skipped",
        )

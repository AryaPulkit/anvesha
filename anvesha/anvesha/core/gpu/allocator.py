"""Role -> CUDA node assignment (S13.2 auto-detection, S13.3 manual, S13.4 single).

``allocate`` is a pure function: it never probes hardware and never loads
models. Roles whose VRAM requirement is ``None`` or ``0`` are remote/CPU
backends and receive an empty node list.
"""

from __future__ import annotations

import logging

from anvesha.core.config import GpuConfig
from anvesha.core.exceptions import ConfigError, GpuNodeUnavailableError, InsufficientVramError
from anvesha.core.gpu.probe import GpuDevice

logger = logging.getLogger(__name__)


def allocate(
    role_vram: dict[str, float],
    config: GpuConfig,
    devices: list[GpuDevice],
) -> dict[str, list[int]]:
    """Assign each model role to a list of CUDA node ids per S13.2.

    ``role_vram`` maps role name (e.g. ``"generator"``) to the adapter's
    ``reported_vram_gb`` (``None``/``0`` for remote backends).
    Raises :class:`InsufficientVramError` when no assignment fits and
    :class:`GpuNodeUnavailableError` when a manual assignment references a
    node that was not probed.
    """
    assignments: dict[str, list[int]] = {
        role: [] for role, vram in role_vram.items() if not vram
    }
    gpu_roles = {role: vram for role, vram in role_vram.items() if vram}

    if config.mode == "single":
        for role in gpu_roles:
            assignments[role] = [config.single_node_id]
    elif config.mode == "manual":
        _allocate_manual(gpu_roles, config, devices, assignments)
    else:  # "auto"
        usable = [d for d in devices if d.free_vram_gb > config.vram_headroom_gb]
        if config.execution_mode == "sequential":
            _allocate_sequential(gpu_roles, config, usable, assignments)
        else:
            _allocate_parallel(gpu_roles, usable, assignments)

    logger.info(
        "GPU assignment (%s/%s): %s", config.mode, config.execution_mode, assignments
    )
    return assignments


def _allocate_manual(
    gpu_roles: dict[str, float],
    config: GpuConfig,
    devices: list[GpuDevice],
    assignments: dict[str, list[int]],
) -> None:
    """S13.3: trust the configured mapping, but verify the nodes exist."""
    manual = config.manual_assignments
    assert manual is not None  # enforced by GpuConfig validation
    known = {d.device_id for d in devices}
    for role in gpu_roles:
        nodes = getattr(manual, f"{role}_nodes", None)
        if nodes is None:
            raise ConfigError(f"manual_assignments has no node list for role {role!r}")
        missing = sorted(n for n in nodes if n not in known)
        if missing:
            raise GpuNodeUnavailableError(
                f"manual assignment for role {role!r} references node(s) {missing} "
                f"not present among probed devices {sorted(known)}"
            )
        assignments[role] = list(nodes)


def _allocate_sequential(
    gpu_roles: dict[str, float],
    config: GpuConfig,
    usable: list[GpuDevice],
    assignments: dict[str, list[int]],
) -> None:
    """S13.2 step 4: all roles time-share the group with the most free VRAM."""
    if not gpu_roles:
        return
    group = sorted(usable, key=lambda d: d.device_id)
    group_free = sum(d.free_vram_gb for d in group)
    largest_role, largest_vram = max(gpu_roles.items(), key=lambda item: item[1])
    if largest_vram > group_free:
        raise InsufficientVramError(
            f"sequential mode: largest role {largest_role!r} needs "
            f"{largest_vram:.1f}GB but the usable node group "
            f"{[d.device_id for d in group]} has only {group_free:.1f}GB free "
            f"(headroom {config.vram_headroom_gb:.1f}GB)"
        )
    node_ids = [d.device_id for d in group]
    for role in gpu_roles:
        assignments[role] = list(node_ids)


def _allocate_parallel(
    gpu_roles: dict[str, float],
    usable: list[GpuDevice],
    assignments: dict[str, list[int]],
) -> None:
    """S13.2 step 5: largest role first, smallest node group that fits it."""
    # device_id is the tie-break so equal-VRAM nodes pick deterministically by
    # id (matching the sequential path), not by caller-supplied input order.
    available = sorted(usable, key=lambda d: (d.free_vram_gb, d.device_id))
    for role, vram in sorted(gpu_roles.items(), key=lambda item: -item[1]):
        # Best fit: the smallest single node that satisfies the requirement.
        single = next((d for d in available if d.free_vram_gb >= vram), None)
        if single is not None:
            chosen = [single]
        else:
            # Span multiple nodes, biggest first, until the requirement is met.
            chosen = []
            accumulated = 0.0
            for device in sorted(available, key=lambda d: (-d.free_vram_gb, d.device_id)):
                chosen.append(device)
                accumulated += device.free_vram_gb
                if accumulated >= vram:
                    break
            if accumulated < vram:
                raise InsufficientVramError(
                    f"parallel mode: role {role!r} needs {vram:.1f}GB but only "
                    f"{accumulated:.1f}GB free remains across "
                    f"{len(available)} unassigned node(s); consider "
                    f'execution_mode "sequential"'
                )
        assignments[role] = sorted(d.device_id for d in chosen)
        for device in chosen:
            available.remove(device)

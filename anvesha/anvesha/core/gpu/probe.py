"""CUDA device enumeration via nvidia-smi (S13.2 step 1).

Probing never raises: a missing or failing ``nvidia-smi`` yields an empty
device list so remote-only configurations work on GPU-less machines.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

_QUERY_CMD: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.free",
    "--format=csv,noheader,nounits",
)

_MIB_PER_GIB = 1024.0


@dataclass
class GpuDevice:
    """One CUDA device as reported by nvidia-smi (VRAM in GiB)."""

    device_id: int
    name: str
    total_vram_gb: float
    free_vram_gb: float


def _default_runner(cmd: Sequence[str]) -> str:
    """Run ``cmd`` and return its stdout. Raises on missing binary / failure."""
    result = subprocess.run(
        list(cmd), capture_output=True, text=True, check=True, timeout=30
    )
    return result.stdout


def probe_devices(
    node_range: tuple[int, int] = (0, 7),
    runner: Callable[[Sequence[str]], str] | None = None,
) -> list[GpuDevice]:
    """Enumerate CUDA devices whose index falls within ``node_range`` (inclusive).

    ``runner`` is an injectable callable returning the command stdout (tests
    pass a fake; the default wraps ``subprocess.run``). If nvidia-smi is
    missing or errors, logs a warning and returns ``[]`` - never raises.
    Memory values are converted from MiB to GiB.
    """
    run = runner if runner is not None else _default_runner
    try:
        stdout = run(_QUERY_CMD)
    except Exception as exc:  # FileNotFoundError, CalledProcessError, timeout, ...
        logger.warning("GPU probe failed (%s); assuming no local GPUs", exc)
        return []

    lo, hi = node_range
    devices: list[GpuDevice] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        try:
            device_id = int(parts[0])
            name = parts[1]
            total_vram_gb = float(parts[2]) / _MIB_PER_GIB
            free_vram_gb = float(parts[3]) / _MIB_PER_GIB
        except (IndexError, ValueError):
            logger.warning("skipping malformed nvidia-smi line: %r", line)
            continue
        if lo <= device_id <= hi:
            devices.append(GpuDevice(device_id, name, total_vram_gb, free_vram_gb))
    logger.debug("probed %d device(s) in range %s", len(devices), node_range)
    return devices

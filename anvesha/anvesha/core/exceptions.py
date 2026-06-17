"""Exception hierarchy shared across the Anvesha SDK.

Failure types map to the recovery rules in
research_gap_pipeline_implementation.md S14.5: timeout, parse_error and
gpu_oom are the triggers the FallbackAdapter and the retry machinery key on.
"""


class AnveshaError(Exception):
    """Base class for every Anvesha error."""


class ConfigError(AnveshaError):
    """Invalid, missing, or inconsistent configuration."""


class WorkspaceError(AnveshaError):
    """Research project workspace is missing, invalid, or unwritable."""


class PhaseValidationError(AnveshaError):
    """Phase inputs are missing or phase output failed its exit criteria."""


class ToolNotPermittedError(AnveshaError):
    """An MCP tool outside the phase's whitelist was requested (ANVESHA.md S12.8)."""


class CheckpointError(AnveshaError):
    """Checkpoint state could not be read or reconstructed."""


class AdapterError(AnveshaError):
    """Base class for LLM backend failures."""


class AdapterTimeoutError(AdapterError):
    """Backend did not respond in time (fallback trigger: "timeout")."""


class AdapterParseError(AdapterError):
    """Backend output could not be parsed (fallback trigger: "parse_error")."""


class AdapterConnectionError(AdapterError):
    """Backend is unreachable."""


class LowConfidenceError(AdapterError):
    """Backend reported a low-confidence result (fallback trigger: "low_confidence")."""


class GpuError(AnveshaError):
    """Base class for GPU management failures."""


class GpuOomError(GpuError):
    """GPU ran out of memory (fallback trigger: "gpu_oom")."""


class InsufficientVramError(GpuError):
    """No node assignment satisfies the models' VRAM requirements (S13.2)."""


class GpuNodeUnavailableError(GpuError):
    """An assigned CUDA node failed the startup health check (S13.6)."""

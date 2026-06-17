"""Adapter factory: build an :class:`LLMClient` from resolved configuration.

Backend selection follows the ANVESHA.md S7.2 precedence chain (CLI override
> per-phase override > project default), implemented in
:meth:`LlmConfig.resolve_backend_name`. The factory then constructs the
adapter named by that backend's ``type``, validating its fields through the
adapter's own pydantic config class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import ValidationError

from anvesha.core.adapters.base import LLMClient
from anvesha.core.adapters.cloud import (
    AnthropicAdapter,
    AnthropicAdapterConfig,
    CloudOpenAIConfig,
    GoogleAdapter,
    GoogleAdapterConfig,
    OpenAICloudAdapter,
)
from anvesha.core.adapters.fallback import FallbackAdapter, FallbackAdapterConfig
from anvesha.core.adapters.local import (
    HuggingFaceAdapter,
    HuggingFaceAdapterConfig,
    LlamaCppAdapter,
    LlamaCppAdapterConfig,
    NemotronAdapter,
    NemotronAdapterConfig,
    OllamaAdapter,
    OllamaAdapterConfig,
    VLLMAdapter,
    VLLMAdapterConfig,
)
from anvesha.core.exceptions import ConfigError

if TYPE_CHECKING:
    from anvesha.core.config import BackendConfig, PipelineConfig

#: backend type -> (adapter class, pydantic config class). The "fallback"
#: type is handled separately because it composes other backends.
_SIMPLE_BACKENDS: dict[str, tuple[type[LLMClient], type]] = {
    "vllm": (VLLMAdapter, VLLMAdapterConfig),
    "nemotron": (NemotronAdapter, NemotronAdapterConfig),
    "ollama": (OllamaAdapter, OllamaAdapterConfig),
    "llamacpp": (LlamaCppAdapter, LlamaCppAdapterConfig),
    "huggingface": (HuggingFaceAdapter, HuggingFaceAdapterConfig),
    "anthropic": (AnthropicAdapter, AnthropicAdapterConfig),
    "openai": (OpenAICloudAdapter, CloudOpenAIConfig),
    "google": (GoogleAdapter, GoogleAdapterConfig),
}


def _build_simple(name: str, entry: "BackendConfig") -> LLMClient:
    adapter_cls, config_cls = _SIMPLE_BACKENDS[entry.type]
    try:
        config = config_cls(**entry.extra_fields())
    except ValidationError as exc:
        raise ConfigError(f"invalid config for backend {name!r}: {exc}") from exc
    return adapter_cls(config)


def _build_fallback(
    name: str,
    entry: "BackendConfig",
    backends: dict[str, "BackendConfig"],
    building: frozenset[str],
) -> LLMClient:
    fields = entry.extra_fields()
    primary_name = fields.pop("primary", None)
    fallback_name = fields.pop("fallback", None)
    if not primary_name or not fallback_name:
        raise ConfigError(
            f"fallback backend {name!r} must name both a 'primary' and a "
            "'fallback' backend"
        )
    if primary_name == name or fallback_name == name:
        raise ConfigError(f"fallback backend {name!r} cannot reference itself")
    primary = _build(primary_name, backends, building)
    fallback = _build(fallback_name, backends, building)
    try:
        config = FallbackAdapterConfig(primary=primary, fallback=fallback, **fields)
    except ValidationError as exc:
        raise ConfigError(f"invalid config for backend {name!r}: {exc}") from exc
    return FallbackAdapter(config)


def _build(
    name: str,
    backends: dict[str, "BackendConfig"],
    building: frozenset[str] = frozenset(),
) -> LLMClient:
    if name in building:
        chain = " -> ".join([*building, name])
        raise ConfigError(f"fallback backend cycle detected: {chain}")
    if name not in backends:
        raise ConfigError(
            f"unknown LLM backend {name!r}: not declared under llm.backends"
        )
    entry = backends[name]
    if entry.type == "fallback":
        return _build_fallback(name, entry, backends, building | {name})
    if entry.type not in _SIMPLE_BACKENDS:
        raise ConfigError(
            f"backend {name!r} has unknown type {entry.type!r} "
            f"(known: {', '.join(sorted(_SIMPLE_BACKENDS))}, fallback)"
        )
    return _build_simple(name, entry)


def create_llm_client(
    config: "PipelineConfig",
    phase_id: int,
    backend_override: str | None = None,
) -> LLMClient:
    """Build the LLM client for ``phase_id`` per the S7.2 precedence chain."""
    name = config.llm.resolve_backend_name(phase_id, backend_override)
    return _build(name, config.llm.backends)

"""Local LLM backend adapters (ANVESHA.md S10.1/S10.3, impl spec S15.1-15.3)."""

from anvesha.core.adapters.local.huggingface import (
    HuggingFaceAdapter,
    HuggingFaceAdapterConfig,
)
from anvesha.core.adapters.local.llamacpp import LlamaCppAdapter, LlamaCppAdapterConfig
from anvesha.core.adapters.local.nemotron import NemotronAdapter, NemotronAdapterConfig
from anvesha.core.adapters.local.ollama import OllamaAdapter, OllamaAdapterConfig
from anvesha.core.adapters.local.vllm import VLLMAdapter, VLLMAdapterConfig

__all__ = [
    "HuggingFaceAdapter",
    "HuggingFaceAdapterConfig",
    "LlamaCppAdapter",
    "LlamaCppAdapterConfig",
    "NemotronAdapter",
    "NemotronAdapterConfig",
    "OllamaAdapter",
    "OllamaAdapterConfig",
    "VLLMAdapter",
    "VLLMAdapterConfig",
]

"""Nemotron 3 Super adapter - S15.2.

Both supported backends (vLLM and TensorRT-LLM serve) expose the OpenAI
chat-completions API, so the transport is entirely inherited; the only
Nemotron-specific behaviour is the reasoning-level system prompt header:
``"<reasoning_level>{level}</reasoning_level>"``.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import AliasChoices, Field

from anvesha.core.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig

logger = logging.getLogger(__name__)


class NemotronAdapterConfig(OpenAICompatConfig):
    """Configuration for Nemotron 3 Super (S15.2)."""

    base_url: str | None = "http://localhost:8000/v1"
    model_name: str = Field(
        default="nvidia/nemotron-3-super-120b-a12b",
        validation_alias=AliasChoices("model_name", "model"),
    )
    #: Serving stack; both speak the OpenAI-compatible API.
    backend: Literal["tensorrt_llm", "vllm"] = "vllm"
    #: "high" recommended for research pipelines without latency pressure.
    reasoning_level: Literal["low", "medium", "high"] = "high"
    context_window: int = 1_000_000
    timeout_seconds: int = 240
    reported_vram_gb: float | None = 64.0


class NemotronAdapter(OpenAICompatAdapter):
    """LLMClient over a Nemotron 3 Super server (vLLM or TensorRT-LLM)."""

    config_cls = NemotronAdapterConfig
    config: NemotronAdapterConfig

    def system_prompt_header(self) -> str | None:
        return f"<reasoning_level>{self.config.reasoning_level}</reasoning_level>"

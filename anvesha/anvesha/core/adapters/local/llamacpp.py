"""llama.cpp server-mode adapter (ANVESHA.md S10.3).

``llama-server`` exposes an OpenAI-compatible endpoint and serves a single
model regardless of the ``model`` request field, so ``model_name`` defaults
to the placeholder ``"local"``. Everything else is inherited.
"""

from __future__ import annotations

import logging

from pydantic import AliasChoices, Field

from anvesha.core.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig

logger = logging.getLogger(__name__)


class LlamaCppAdapterConfig(OpenAICompatConfig):
    """Configuration for a llama.cpp server (llama-server)."""

    base_url: str | None = "http://localhost:8080/v1"
    model_name: str = Field(
        default="local",
        validation_alias=AliasChoices("model_name", "model"),
    )


class LlamaCppAdapter(OpenAICompatAdapter):
    """LLMClient over a llama.cpp server."""

    config_cls = LlamaCppAdapterConfig
    config: LlamaCppAdapterConfig

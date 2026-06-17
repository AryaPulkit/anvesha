"""Ollama adapter - S15.3.

Ollama exposes an OpenAI-compatible endpoint at ``/v1``. The models served
through it are typically smaller, so by default ``structured_run`` strips
non-essential JSON-Schema keywords (titles, descriptions, format
constraints, ...) before the prompt-based schema instruction - smaller
models follow lean schemas more reliably.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from anvesha.core.adapters.base import MCPClient
from anvesha.core.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig

logger = logging.getLogger(__name__)

#: JSON-Schema keywords kept by schema simplification.
_KEEP_KEYS = frozenset({"type", "properties", "required", "items", "enum"})


def simplify_schema(schema: dict) -> dict:
    """Strip a JSON Schema down to type/properties/required/items/enum."""
    simplified: dict = {}
    for key, value in schema.items():
        if key not in _KEEP_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            simplified[key] = {
                name: simplify_schema(sub) if isinstance(sub, dict) else sub
                for name, sub in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            simplified[key] = simplify_schema(value)
        else:
            simplified[key] = value
    return simplified


class OllamaAdapterConfig(OpenAICompatConfig):
    """Configuration for Ollama's OpenAI-compatible endpoint (S15.3)."""

    base_url: str | None = "http://localhost:11434/v1"
    api_key: str | None = "ollama"
    timeout_seconds: int = 90
    #: reported_vram_gb stays user-declared - no detection via Ollama API (S13.5).
    simplify_schemas: bool = True


class OllamaAdapter(OpenAICompatAdapter):
    """LLMClient over an Ollama server."""

    config_cls = OllamaAdapterConfig
    config: OllamaAdapterConfig

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> dict:
        if self.config.simplify_schemas:
            json_schema = simplify_schema(json_schema)
        return super().structured_run(
            prompt, json_schema, mcps=mcps, input_files=input_files
        )

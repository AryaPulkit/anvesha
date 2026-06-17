"""Google Gemini cloud adapter (ANVESHA.md S10.2).

v0 scope: plain text completion only - this adapter does NOT support MCP
tool calls and raises :class:`AdapterError` if ``mcps`` are passed. Use the
Anthropic or OpenAI cloud adapters as the fallback for tool-using phases.

The ``google.generativeai`` SDK is an optional dependency - it is imported
lazily (install via ``pip install "anvesha[cloud]"``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient
from anvesha.core.exceptions import AdapterError, AnveshaError

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    'the "google-generativeai" SDK is not installed - '
    'run pip install "anvesha[cloud]"'
)


class GoogleAdapterConfig(BaseModel):
    """Configuration for :class:`GoogleAdapter`.

    ``model_name`` also accepts the alias ``model``. ``api_key=None`` falls
    back to the ``GOOGLE_API_KEY`` environment variable at client build time.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"))
    api_key: str | None = None
    timeout_seconds: int = 120


class GoogleAdapter(LLMClient):
    """LLMClient over the Google Gemini API (plain completion only, v0)."""

    def __init__(self, config: GoogleAdapterConfig, client: Any | None = None):
        self.config = config
        self.name = config.model_name
        self.reported_vram_gb = None  # cloud backend - no local VRAM
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import google.generativeai as genai
            except ImportError as exc:
                raise AdapterError(f"{self.name}: {_INSTALL_HINT}") from exc
            genai.configure(api_key=self._resolve_api_key())
            self._client = genai.GenerativeModel(self.config.model_name)
        return self._client

    def _resolve_api_key(self) -> str:
        key = self.config.api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise AdapterError(
                f"{self.name}: no Google API key - set api_key in the "
                "backend config or the GOOGLE_API_KEY environment variable"
            )
        return key

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        if mcps:
            raise AdapterError(
                f"{self.name}: GoogleAdapter does not support MCP tool calls "
                "(v0 scope is plain completion only) - use the Anthropic or "
                "OpenAI cloud adapter for tool-using phases"
            )
        full_prompt = self.inject_input_files(prompt, input_files)
        try:
            response = self.client.generate_content(
                full_prompt,
                request_options={"timeout": self.config.timeout_seconds},
            )
        except AnveshaError:
            raise
        except Exception as exc:
            raise AdapterError(f"{self.name}: Google API error: {exc}") from exc

        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            content=response.text or "",
            tool_calls_made=0,
            tokens_in=(getattr(usage, "prompt_token_count", 0) or 0) if usage else 0,
            tokens_out=(getattr(usage, "candidates_token_count", 0) or 0)
            if usage
            else 0,
        )

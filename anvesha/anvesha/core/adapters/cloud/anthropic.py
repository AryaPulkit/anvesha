"""Anthropic Claude cloud adapter (ANVESHA.md S10.2).

Implements the LLMClient contract over the Anthropic Messages API,
including the tool-use loop for MCP tools. The ``anthropic`` SDK is an
optional dependency - it is imported lazily so the rest of the package
works without it (install via ``pip install "anvesha[cloud]"``).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, NoReturn, Sequence

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient
from anvesha.core.exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)

#: Safety cap on tool-use turns within a single run() call.
MAX_TOOL_TURNS = 25

_INSTALL_HINT = (
    'the "anthropic" SDK is not installed - run pip install "anvesha[cloud]"'
)


class AnthropicAdapterConfig(BaseModel):
    """Configuration for :class:`AnthropicAdapter`.

    ``model_name`` also accepts the alias ``model`` (the key used in the
    ANVESHA.md S8 config examples). ``api_key=None`` falls back to the
    ``ANTHROPIC_API_KEY`` environment variable at client build time.
    """

    model_config = ConfigDict(populate_by_name=True)

    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"))
    api_key: str | None = None
    max_tokens: int = 8192
    timeout_seconds: int = 120


def _sanitize_tool_name(name: str) -> str:
    """Map namespaced MCP names ("papersflow.search") to the Anthropic tool
    name charset ("papersflow__search")."""
    return name.replace(".", "__")


class AnthropicAdapter(LLMClient):
    """LLMClient over the Anthropic Messages API with tool-use support."""

    def __init__(self, config: AnthropicAdapterConfig, client: Any | None = None):
        self.config = config
        self.name = config.model_name
        self.reported_vram_gb = None  # cloud backend - no local VRAM
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise AdapterError(f"{self.name}: {_INSTALL_HINT}") from exc
            self._client = anthropic.Anthropic(
                api_key=self._resolve_api_key(),
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _resolve_api_key(self) -> str:
        key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise AdapterError(
                f"{self.name}: no Anthropic API key - set api_key in the "
                "backend config or the ANTHROPIC_API_KEY environment variable"
            )
        return key

    def _raise_mapped(self, exc: Exception) -> NoReturn:
        """Translate anthropic SDK errors to Anvesha adapter errors.

        The SDK is imported lazily here too: with an injected fake client
        (tests) the SDK may be absent, in which case the original exception
        propagates unchanged.
        """
        try:
            import anthropic
        except ImportError:
            raise exc
        # APITimeoutError subclasses APIConnectionError - check it first.
        if isinstance(exc, anthropic.APITimeoutError):
            raise AdapterTimeoutError(
                f"{self.name}: request timed out after "
                f"{self.config.timeout_seconds}s"
            ) from exc
        if isinstance(exc, anthropic.APIConnectionError):
            raise AdapterConnectionError(
                f"{self.name}: cannot reach the Anthropic API: {exc}"
            ) from exc
        if isinstance(exc, anthropic.AnthropicError):
            raise AdapterError(f"{self.name}: Anthropic API error: {exc}") from exc
        raise exc

    def _build_tools(
        self, mcps: Sequence[MCPClient]
    ) -> tuple[list[dict], dict[str, tuple[MCPClient, str]]]:
        tools: list[dict] = []
        dispatch: dict[str, tuple[MCPClient, str]] = {}
        for mcp in mcps:
            for tool in mcp.list_tools():
                wire_name = _sanitize_tool_name(tool.name)
                tools.append(
                    {
                        "name": wire_name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                )
                dispatch[wire_name] = (mcp, tool.name)
        return tools, dispatch

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        full_prompt = self.inject_input_files(prompt, input_files)
        tools, dispatch = self._build_tools(mcps)

        messages: list[dict[str, Any]] = [{"role": "user", "content": full_prompt}]
        tool_calls_made = tokens_in = tokens_out = 0
        for _ in range(MAX_TOOL_TURNS):
            request: dict[str, Any] = {
                "model": self.config.model_name,
                "max_tokens": self.config.max_tokens,
                "messages": messages,
            }
            if tools:
                request["tools"] = tools

            try:
                response = self.client.messages.create(**request)
            except Exception as exc:  # mapped (or re-raised) below
                self._raise_mapped(exc)

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens_in += getattr(usage, "input_tokens", 0) or 0
                tokens_out += getattr(usage, "output_tokens", 0) or 0

            if response.stop_reason != "tool_use":
                text = "".join(
                    block.text
                    for block in response.content
                    if getattr(block, "type", None) == "text"
                )
                return LLMResponse(
                    content=text,
                    tool_calls_made=tool_calls_made,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

            # Echo the assistant turn back verbatim, then answer every
            # tool_use block with a tool_result block in one user turn.
            messages.append({"role": "assistant", "content": response.content})
            results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                wire_name = block.name
                if wire_name not in dispatch:
                    result = f"ERROR: unknown tool {wire_name!r}"
                    logger.warning(
                        "%s requested unknown tool %r", self.name, wire_name
                    )
                else:
                    mcp, real_name = dispatch[wire_name]
                    result = mcp.call_tool(real_name, dict(block.input or {}))
                tool_calls_made += 1
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )
            messages.append({"role": "user", "content": results})

        raise AdapterError(
            f"{self.name}: exceeded {MAX_TOOL_TURNS} tool-use turns without "
            "a final response"
        )

"""Shared implementation for every backend that speaks the OpenAI
chat-completions API: vLLM, Ollama, llama.cpp server, TensorRT-LLM serve,
and the OpenAI cloud API itself.

The tool-call loop here is the reference implementation of the LLMClient
contract (ANVESHA.md S7.2): inject input files, expose MCP tools in the
backend's native format, dispatch tool calls until the model produces a
final text response.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import openai
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient
from anvesha.core.exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterTimeoutError,
)

logger = logging.getLogger(__name__)

#: Safety cap on tool-call turns within a single run() call.
MAX_TOOL_TURNS = 25


class OpenAICompatConfig(BaseModel):
    """Base configuration for OpenAI-compatible backends.

    ``model_name`` also accepts the alias ``model`` (the key used in the
    ANVESHA.md S8.2 example config).
    """

    model_config = ConfigDict(populate_by_name=True)

    base_url: str | None = None
    model_name: str = Field(validation_alias=AliasChoices("model_name", "model"))
    api_key: str | None = "EMPTY"
    timeout_seconds: int = 120
    reported_vram_gb: float | None = None
    #: Greedy decoding by default - determinism (filter spec NFR-1).
    temperature: float = 0.0
    max_tokens: int | None = None


def _sanitize_tool_name(name: str) -> str:
    """Map namespaced MCP names ("papersflow.search") to the OpenAI function
    name charset ("papersflow__search")."""
    return name.replace(".", "__")


class OpenAICompatAdapter(LLMClient):
    """LLMClient over any OpenAI-compatible chat-completions endpoint."""

    config_cls: type[OpenAICompatConfig] = OpenAICompatConfig

    def __init__(self, config: OpenAICompatConfig, client: Any | None = None):
        self.config = config
        self.name = config.model_name
        self.reported_vram_gb = config.reported_vram_gb
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.config.base_url,
                api_key=self._resolve_api_key(),
                timeout=self.config.timeout_seconds,
            )
        return self._client

    def _resolve_api_key(self) -> str | None:
        """Hook: cloud subclasses resolve keys from the environment."""
        return self.config.api_key

    def system_prompt_header(self) -> str | None:
        """Hook: backend-specific system header (e.g. Nemotron reasoning level)."""
        return None

    def _extra_request_kwargs(self) -> dict[str, Any]:
        """Hook: extra chat.completions.create kwargs (e.g. vLLM extra_body)."""
        return {}

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
                        "type": "function",
                        "function": {
                            "name": wire_name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
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

        messages: list[dict[str, Any]] = []
        header = self.system_prompt_header()
        if header:
            messages.append({"role": "system", "content": header})
        messages.append({"role": "user", "content": full_prompt})

        tool_calls_made = tokens_in = tokens_out = 0
        for _ in range(MAX_TOOL_TURNS):
            request: dict[str, Any] = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": self.config.temperature,
            }
            if self.config.max_tokens is not None:
                request["max_tokens"] = self.config.max_tokens
            if tools:
                request["tools"] = tools
            request.update(self._extra_request_kwargs())

            try:
                response = self.client.chat.completions.create(**request)
            except openai.APITimeoutError as exc:
                raise AdapterTimeoutError(
                    f"{self.name}: request timed out after "
                    f"{self.config.timeout_seconds}s"
                ) from exc
            except openai.APIConnectionError as exc:
                raise AdapterConnectionError(
                    f"{self.name}: cannot reach backend at "
                    f"{self.config.base_url}: {exc}"
                ) from exc
            except openai.OpenAIError as exc:
                raise AdapterError(f"{self.name}: backend error: {exc}") from exc

            usage = getattr(response, "usage", None)
            if usage is not None:
                tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                tokens_out += getattr(usage, "completion_tokens", 0) or 0

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            if not tool_calls:
                return LLMResponse(
                    content=message.content or "",
                    tool_calls_made=tool_calls_made,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            )
            for tc in tool_calls:
                wire_name = tc.function.name
                if wire_name not in dispatch:
                    result = f"ERROR: unknown tool {wire_name!r}"
                    logger.warning("%s requested unknown tool %r", self.name, wire_name)
                else:
                    mcp, real_name = dispatch[wire_name]
                    try:
                        arguments = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result = mcp.call_tool(real_name, arguments)
                tool_calls_made += 1
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

        raise AdapterError(
            f"{self.name}: exceeded {MAX_TOOL_TURNS} tool-call turns without "
            "a final response"
        )

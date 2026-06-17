"""Adapter contract shared by every LLM backend.

Every backend - local (vLLM, Nemotron, Ollama, llama.cpp, HuggingFace) and
cloud (Anthropic, OpenAI, Google), plus the FallbackAdapter - implements
``LLMClient``. Phases receive an already-constructed ``LLMClient`` and never
know which backend is in use (ANVESHA.md S7.2).
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from anvesha.core.exceptions import AdapterParseError


@dataclass
class LLMResponse:
    """Result of one ``LLMClient.run`` call (ANVESHA.md S7.2)."""

    content: str
    tool_calls_made: int = 0
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class MCPToolDef:
    """One tool exposed by an MCP server, in backend-neutral form.

    ``name`` is namespaced ``"<server>.<tool>"`` (e.g. ``"papersflow.search"``)
    so per-phase whitelist patterns like ``"papersflow.*"`` can match it.
    ``input_schema`` is the JSON Schema of the tool's arguments.
    """

    name: str
    description: str
    input_schema: dict


@runtime_checkable
class MCPClient(Protocol):
    """Minimal MCP client surface the adapters need for the tool-call loop."""

    def list_tools(self) -> list[MCPToolDef]:
        ...

    def call_tool(self, name: str, arguments: dict) -> str:
        ...


def parse_json_response(text: str) -> dict:
    """Extract a single JSON object from a model response.

    Tries, in order: the whole stripped text, a ```json fenced block, and the
    outermost ``{...}`` span. Raises :class:`AdapterParseError` if none parse
    to a JSON object.
    """
    stripped = text.strip()
    attempts = [stripped]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        attempts.append(fence.group(1).strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        attempts.append(stripped[start : end + 1])
    for candidate in attempts:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise AdapterParseError(f"response is not a JSON object: {stripped[:200]!r}")


class LLMClient(ABC):
    """The minimal interface every LLM backend implements (ANVESHA.md S7.2).

    Implementations of :meth:`run` must:

    1. Inject the contents of ``input_files`` into the prompt context
       (:meth:`inject_input_files`).
    2. Convert the MCP clients into the backend's native tool format.
    3. Run a tool-call loop: send prompt -> dispatch any tool calls via the
       MCP clients -> append results -> repeat until a final text response.
    4. Return an :class:`LLMResponse`.
    """

    #: Human-readable backend identifier (usually the model name).
    name: str = "llm"
    #: VRAM the GPU allocator should budget for this backend (S13.5).
    #: ``None`` means remote/CPU - the GPU scheduler ignores it.
    reported_vram_gb: float | None = None

    @abstractmethod
    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        ...

    def structured_run(
        self,
        prompt: str,
        json_schema: dict,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> dict:
        """Run and parse a JSON-object response conforming to ``json_schema``.

        Default implementation is prompt-based schema instruction; backends
        with native structured output (e.g. vLLM ``guided_json``) override it.
        Raises :class:`AdapterParseError` if the response is not valid JSON.
        """
        schema_prompt = (
            f"{prompt}\n\n"
            "Respond with ONLY a single JSON object that conforms to this "
            "JSON Schema (no prose, no markdown fences):\n"
            f"{json.dumps(json_schema, indent=2)}"
        )
        response = self.run(schema_prompt, mcps=mcps, input_files=input_files)
        return parse_json_response(response.content)

    def load(self) -> None:
        """Make the model resident (GPU scheduler hook, S12.2). No-op by default."""

    def unload(self) -> None:
        """Release the model's resources (GPU scheduler hook). No-op by default."""

    @staticmethod
    def inject_input_files(prompt: str, input_files: Sequence[Path]) -> str:
        """Prepend the contents of ``input_files`` to the prompt context."""
        if not input_files:
            return prompt
        blocks = []
        for path in input_files:
            path = Path(path)
            blocks.append(
                f'<input_file path="{path}">\n'
                f'{path.read_text(encoding="utf-8")}\n'
                f"</input_file>"
            )
        return "\n\n".join(blocks) + "\n\n" + prompt

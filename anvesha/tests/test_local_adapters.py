"""Offline tests for the local LLM adapters (impl spec S15.1-15.3).

All served backends are exercised through a fake OpenAI-compatible client
that records requests and returns scripted responses - no network, no GPU.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from anvesha.core.adapters.base import MCPToolDef
from anvesha.core.adapters.local import (
    HuggingFaceAdapter,
    HuggingFaceAdapterConfig,
    LlamaCppAdapterConfig,
    NemotronAdapter,
    NemotronAdapterConfig,
    OllamaAdapter,
    OllamaAdapterConfig,
    VLLMAdapter,
    VLLMAdapterConfig,
)
from anvesha.core.adapters.local.ollama import simplify_schema
from anvesha.core.exceptions import AdapterError, AdapterTimeoutError

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


def make_response(
    content: str | None = None,
    tool_calls: list | None = None,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
        usage=SimpleNamespace(prompt_tokens=tokens_in, completion_tokens=tokens_out),
    )


def make_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


class FakeOpenAIClient:
    """Records chat.completions.create requests, returns scripted items.

    A scripted item that is an Exception instance is raised instead.
    """

    def __init__(self, script: list[Any]):
        self.requests: list[dict] = []
        self._script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        return [
            MCPToolDef(
                name="papersflow.search",
                description="Search papers.",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return "TOOL_RESULT"


def timeout_error() -> openai.APITimeoutError:
    return openai.APITimeoutError(request=httpx.Request("POST", "http://x"))


def bad_request_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "guided_json not supported",
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None,
    )


# ---------------------------------------------------------------------------
# plain completion / token accounting
# ---------------------------------------------------------------------------


def test_plain_completion_and_token_accounting() -> None:
    fake = FakeOpenAIClient([make_response("hello", tokens_in=11, tokens_out=7)])
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)

    result = adapter.run("hi")

    assert result.content == "hello"
    assert result.tool_calls_made == 0
    assert result.tokens_in == 11
    assert result.tokens_out == 7
    assert fake.requests[0]["model"] == "m"
    assert "tools" not in fake.requests[0]


def test_config_defaults() -> None:
    assert VLLMAdapterConfig(model_name="m").base_url == "http://localhost:8000/v1"
    assert VLLMAdapterConfig(model_name="m").harmony_format is False
    assert VLLMAdapterConfig(model_name="m").reasoning_effort == "high"

    nemotron = NemotronAdapterConfig()
    assert nemotron.model_name == "nvidia/nemotron-3-super-120b-a12b"
    assert nemotron.backend == "vllm"
    assert nemotron.reasoning_level == "high"
    assert nemotron.context_window == 1_000_000
    assert nemotron.timeout_seconds == 240
    assert nemotron.reported_vram_gb == 64.0

    ollama = OllamaAdapterConfig(model_name="m")
    assert ollama.base_url == "http://localhost:11434/v1"
    assert ollama.api_key == "ollama"
    assert ollama.timeout_seconds == 90
    assert ollama.simplify_schemas is True

    llamacpp = LlamaCppAdapterConfig()
    assert llamacpp.base_url == "http://localhost:8080/v1"
    assert llamacpp.model_name == "local"


# ---------------------------------------------------------------------------
# tool-call loop
# ---------------------------------------------------------------------------


def test_tool_loop_dispatches_namespaced_mcp_call() -> None:
    tool_turn = make_response(
        tool_calls=[make_tool_call("tc1", "papersflow__search", '{"q": "gaps"}')],
        tokens_in=10,
        tokens_out=5,
    )
    final_turn = make_response("done", tokens_in=20, tokens_out=3)
    fake = FakeOpenAIClient([tool_turn, final_turn])
    mcp = FakeMCP()
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)

    result = adapter.run("find gaps", mcps=[mcp])

    # MCP name "papersflow.search" exposed on the wire as "papersflow__search"
    wire_tools = fake.requests[0]["tools"]
    assert wire_tools[0]["function"]["name"] == "papersflow__search"
    # ... and dispatched back under its real namespaced name.
    assert mcp.calls == [("papersflow.search", {"q": "gaps"})]
    assert result.content == "done"
    assert result.tool_calls_made == 1
    assert result.tokens_in == 30
    assert result.tokens_out == 8
    # Second request carries the assistant tool-call turn and the tool result.
    roles = [m["role"] for m in fake.requests[1]["messages"]]
    assert roles == ["user", "assistant", "tool"]
    assert fake.requests[1]["messages"][2]["content"] == "TOOL_RESULT"


# ---------------------------------------------------------------------------
# vLLM harmony header (S15.1)
# ---------------------------------------------------------------------------


def test_vllm_harmony_header_present() -> None:
    fake = FakeOpenAIClient([make_response("ok")])
    adapter = VLLMAdapter(
        VLLMAdapterConfig(model_name="gpt-oss-120b", harmony_format=True,
                          reasoning_effort="medium"),
        client=fake,
    )
    adapter.run("hi")
    messages = fake.requests[0]["messages"]
    assert messages[0] == {"role": "system", "content": "Reasoning: medium"}


def test_vllm_no_harmony_header_when_disabled() -> None:
    fake = FakeOpenAIClient([make_response("ok")])
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)
    adapter.run("hi")
    assert all(m["role"] != "system" for m in fake.requests[0]["messages"])


# ---------------------------------------------------------------------------
# Nemotron reasoning level header (S15.2)
# ---------------------------------------------------------------------------


def test_nemotron_reasoning_level_header() -> None:
    fake = FakeOpenAIClient([make_response("ok")])
    adapter = NemotronAdapter(NemotronAdapterConfig(reasoning_level="low"), client=fake)
    adapter.run("hi")
    messages = fake.requests[0]["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "<reasoning_level>low</reasoning_level>",
    }


# ---------------------------------------------------------------------------
# vLLM guided_json structured_run + fallback (S15.1)
# ---------------------------------------------------------------------------

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}},
    "required": ["a"],
}


def test_vllm_structured_run_uses_guided_json() -> None:
    fake = FakeOpenAIClient([make_response('{"a": 1}')])
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)

    result = adapter.structured_run("give me a", SCHEMA)

    assert result == {"a": 1}
    assert len(fake.requests) == 1
    assert fake.requests[0]["extra_body"] == {"guided_json": SCHEMA}
    # Single plain call: no schema text spliced into the prompt.
    assert "JSON Schema" not in fake.requests[0]["messages"][-1]["content"]


def test_vllm_structured_run_falls_back_on_bad_request() -> None:
    fake = FakeOpenAIClient([bad_request_error(), make_response('{"a": 2}')])
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)

    result = adapter.structured_run("give me a", SCHEMA)

    assert result == {"a": 2}
    assert len(fake.requests) == 2
    # Fallback is the prompt-based base implementation: no guided_json,
    # schema instruction embedded in the prompt instead.
    assert "extra_body" not in fake.requests[1]
    assert "JSON Schema" in fake.requests[1]["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# Ollama schema simplification (S15.3)
# ---------------------------------------------------------------------------


def test_simplify_schema_strips_non_essential_keywords() -> None:
    schema = {
        "title": "Thing",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string", "description": "the name", "minLength": 1},
            "kind": {"type": "string", "enum": ["a", "b"], "title": "Kind"},
            "tags": {
                "type": "array",
                "items": {"type": "string", "format": "uri"},
                "maxItems": 5,
            },
        },
        "required": ["name"],
    }
    assert simplify_schema(schema) == {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "kind": {"type": "string", "enum": ["a", "b"]},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
    }


def test_ollama_structured_run_simplifies_schema_in_prompt() -> None:
    fake = FakeOpenAIClient([make_response('{"name": "x"}')])
    adapter = OllamaAdapter(OllamaAdapterConfig(model_name="m"), client=fake)
    schema = {
        "type": "object",
        "title": "Thing",
        "description": "a thing",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    result = adapter.structured_run("make a thing", schema)

    assert result == {"name": "x"}
    prompt = fake.requests[0]["messages"][-1]["content"]
    assert '"title"' not in prompt and '"description"' not in prompt
    assert '"required"' in prompt and '"properties"' in prompt


def test_ollama_simplification_can_be_disabled() -> None:
    fake = FakeOpenAIClient([make_response('{"name": "x"}')])
    adapter = OllamaAdapter(
        OllamaAdapterConfig(model_name="m", simplify_schemas=False), client=fake
    )
    adapter.structured_run("make a thing", {"type": "object", "title": "Thing"})
    assert '"title"' in fake.requests[0]["messages"][-1]["content"]


# ---------------------------------------------------------------------------
# input-file injection
# ---------------------------------------------------------------------------


def test_input_file_injection(tmp_path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("important context", encoding="utf-8")
    fake = FakeOpenAIClient([make_response("ok")])
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)

    adapter.run("summarize", input_files=[notes])

    prompt = fake.requests[0]["messages"][-1]["content"]
    assert prompt.startswith(f'<input_file path="{notes}">')
    assert "important context" in prompt
    assert prompt.rstrip().endswith("summarize")


# ---------------------------------------------------------------------------
# timeout mapping
# ---------------------------------------------------------------------------


def test_timeout_maps_to_adapter_timeout_error() -> None:
    fake = FakeOpenAIClient([timeout_error()])
    adapter = NemotronAdapter(NemotronAdapterConfig(), client=fake)
    with pytest.raises(AdapterTimeoutError, match="240"):
        adapter.run("hi")


def test_structured_run_guided_timeout_maps_too() -> None:
    fake = FakeOpenAIClient([timeout_error()])
    adapter = VLLMAdapter(VLLMAdapterConfig(model_name="m"), client=fake)
    with pytest.raises(AdapterTimeoutError):
        adapter.structured_run("hi", SCHEMA)


# ---------------------------------------------------------------------------
# HuggingFace adapter (in-process)
# ---------------------------------------------------------------------------


def test_huggingface_lazy_import_error(monkeypatch) -> None:
    # Blocking the module in sys.modules forces ImportError even when
    # transformers happens to be installed.
    monkeypatch.setitem(sys.modules, "transformers", None)
    adapter = HuggingFaceAdapter(HuggingFaceAdapterConfig(model_name="m"))
    with pytest.raises(AdapterError, match=r'pip install "anvesha\[hf\]"'):
        adapter.load()
    # run() auto-loads, so it surfaces the same error.
    with pytest.raises(AdapterError, match=r'pip install "anvesha\[hf\]"'):
        adapter.run("hi")


def test_huggingface_rejects_mcps() -> None:
    adapter = HuggingFaceAdapter(HuggingFaceAdapterConfig(model_name="m"))
    with pytest.raises(AdapterError, match="MCP tool calls"):
        adapter.run("hi", mcps=[FakeMCP()])


def test_huggingface_defaults_and_unload_without_torch(monkeypatch) -> None:
    config = HuggingFaceAdapterConfig(model_name="m")
    assert config.device == "auto"
    assert config.max_new_tokens == 4096
    adapter = HuggingFaceAdapter(config)
    assert adapter.name == "m"
    assert adapter.reported_vram_gb is None
    monkeypatch.setitem(sys.modules, "torch", None)
    adapter.unload()  # must not raise when torch is absent

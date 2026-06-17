"""Offline tests for the cloud adapters and the FallbackAdapter.

No SDKs, no network: cloud adapters get injected fake clients, missing-SDK
paths are simulated via sys.modules, and the FallbackAdapter is exercised
with tiny LLMClient stubs that raise scripted errors.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Sequence

import pytest

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.adapters.cloud.anthropic import AnthropicAdapter, AnthropicAdapterConfig
from anvesha.core.adapters.cloud.google import GoogleAdapter, GoogleAdapterConfig
from anvesha.core.adapters.cloud.openai import CloudOpenAIConfig, OpenAICloudAdapter
from anvesha.core.adapters.fallback import FallbackAdapter, FallbackAdapterConfig
from anvesha.core.exceptions import (
    AdapterConnectionError,
    AdapterError,
    AdapterParseError,
    AdapterTimeoutError,
    GpuOomError,
    LowConfidenceError,
)

# ---------------------------------------------------------------------------
# Stubs


class StubClient(LLMClient):
    """LLMClient stub that raises a scripted error or returns fixed content."""

    reported_vram_gb = None

    def __init__(
        self,
        name: str = "stub",
        content: str = "ok",
        error: Exception | None = None,
    ):
        self.name = name
        self.content = content
        self.error = error
        self.calls: list[str] = []

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return LLMResponse(content=self.content)


class FakeMCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        return [
            MCPToolDef(
                name="papersflow.search",
                description="search papers",
                input_schema={"type": "object"},
            )
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return "TOOL RESULT"


def make_fallback(primary: StubClient, fallback: StubClient, **cfg) -> FallbackAdapter:
    return FallbackAdapter(
        FallbackAdapterConfig(primary=primary, fallback=fallback, **cfg)
    )


# ---------------------------------------------------------------------------
# FallbackAdapter: triggers


@pytest.mark.parametrize(
    "error",
    [
        AdapterTimeoutError("primary timed out"),
        AdapterParseError("primary garbage"),
        GpuOomError("cuda oom"),
    ],
)
def test_default_triggers_route_to_fallback(error: Exception) -> None:
    primary = StubClient(name="local", error=error)
    fallback = StubClient(name="cloud", content="from cloud")
    adapter = make_fallback(primary, fallback)

    response = adapter.run("hello")

    assert response.content == "from cloud"
    assert primary.calls == ["hello"]
    # Same input sent unchanged (S15.4).
    assert fallback.calls == ["hello"]


def test_low_confidence_trigger_when_enabled() -> None:
    primary = StubClient(error=LowConfidenceError("unsure"))
    fallback = StubClient(content="confident")
    adapter = make_fallback(primary, fallback, fallback_on=["low_confidence"])

    assert adapter.run("q").content == "confident"


def test_low_confidence_not_in_default_triggers_propagates() -> None:
    primary = StubClient(error=LowConfidenceError("unsure"))
    fallback = StubClient()
    adapter = make_fallback(primary, fallback)

    with pytest.raises(LowConfidenceError):
        adapter.run("q")
    assert fallback.calls == []


def test_non_trigger_error_propagates() -> None:
    primary = StubClient(error=AdapterConnectionError("unreachable"))
    fallback = StubClient()
    adapter = make_fallback(primary, fallback)

    with pytest.raises(AdapterConnectionError):
        adapter.run("q")
    assert fallback.calls == []


def test_fallback_failure_reraises_original_primary_error() -> None:
    original = AdapterTimeoutError("the original timeout")
    primary = StubClient(error=original)
    fallback = StubClient(error=AdapterConnectionError("cloud down"))
    adapter = make_fallback(primary, fallback)

    with pytest.raises(AdapterTimeoutError) as excinfo:
        adapter.run("q")
    assert excinfo.value is original


# ---------------------------------------------------------------------------
# FallbackAdapter: explicit routing


def test_force_fallback_routes_straight_to_fallback() -> None:
    primary = StubClient(name="local")
    fallback = StubClient(name="cloud", content="explicit cloud")
    adapter = make_fallback(
        primary, fallback, fallback_on=["timeout", "explicit"]
    )

    response = adapter.run("q", force_fallback=True)

    assert response.content == "explicit cloud"
    assert primary.calls == []


def test_force_fallback_ignored_without_explicit_trigger() -> None:
    primary = StubClient(content="from local")
    fallback = StubClient()
    adapter = make_fallback(primary, fallback)  # no "explicit"

    response = adapter.run("q", force_fallback=True)

    assert response.content == "from local"
    assert fallback.calls == []


# ---------------------------------------------------------------------------
# FallbackAdapter: structured_run, logging, delegation


def test_structured_run_falls_back_on_parse_error() -> None:
    primary = StubClient(name="local", content="this is not json")
    fallback = StubClient(name="cloud", content='{"answer": 42}')
    adapter = make_fallback(primary, fallback)

    result = adapter.structured_run("q", json_schema={"type": "object"})

    assert result == {"answer": 42}


def test_fallback_event_logged(caplog: pytest.LogCaptureFixture) -> None:
    primary = StubClient(name="local", error=AdapterTimeoutError("slow"))
    fallback = StubClient(name="cloud")
    adapter = make_fallback(primary, fallback)

    with caplog.at_level(logging.WARNING, logger="anvesha.core.adapters.fallback"):
        adapter.run("q")

    messages = [rec.getMessage() for rec in caplog.records]
    assert any(
        "fallback event" in m and "local" in m and "cloud" in m and "timeout" in m
        for m in messages
    )


def test_fallback_logging_disabled(caplog: pytest.LogCaptureFixture) -> None:
    primary = StubClient(error=AdapterTimeoutError("slow"))
    fallback = StubClient()
    adapter = make_fallback(primary, fallback, log_fallback_events=False)

    with caplog.at_level(logging.WARNING, logger="anvesha.core.adapters.fallback"):
        adapter.run("q")

    assert caplog.records == []


def test_name_and_vram_delegate_to_primary() -> None:
    primary = StubClient(name="local-model")
    primary.reported_vram_gb = 80.0
    adapter = make_fallback(primary, StubClient(name="cloud"))

    assert adapter.name == "local-model"
    assert adapter.reported_vram_gb == 80.0


# ---------------------------------------------------------------------------
# OpenAICloudAdapter: API key resolution


def test_openai_cloud_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    adapter = OpenAICloudAdapter(CloudOpenAIConfig(model="gpt-4o"))
    assert adapter._resolve_api_key() == "sk-env"


def test_openai_cloud_key_from_config_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    adapter = OpenAICloudAdapter(
        CloudOpenAIConfig(model="gpt-4o", api_key="sk-config")
    )
    assert adapter._resolve_api_key() == "sk-config"


def test_openai_cloud_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    adapter = OpenAICloudAdapter(CloudOpenAIConfig(model="gpt-4o"))
    with pytest.raises(AdapterError, match="OPENAI_API_KEY"):
        adapter._resolve_api_key()


# ---------------------------------------------------------------------------
# AnthropicAdapter: tool-use loop with a fake client


class FakeAnthropicMessages:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeAnthropicClient:
    def __init__(self, responses: list) -> None:
        self.messages = FakeAnthropicMessages(responses)


def _anthropic_adapter(responses: list) -> tuple[AnthropicAdapter, FakeAnthropicClient]:
    fake = FakeAnthropicClient(responses)
    adapter = AnthropicAdapter(
        AnthropicAdapterConfig(model="claude-sonnet-4-5", api_key="sk-ant"),
        client=fake,
    )
    return adapter, fake


def test_anthropic_tool_use_loop() -> None:
    tool_turn = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="text", text="let me search"),
            SimpleNamespace(
                type="tool_use",
                id="tu_1",
                name="papersflow__search",
                input={"query": "ssl gaps"},
            ),
        ],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )
    final_turn = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="final answer")],
        usage=SimpleNamespace(input_tokens=20, output_tokens=7),
    )
    adapter, fake = _anthropic_adapter([tool_turn, final_turn])
    mcp = FakeMCP()

    response = adapter.run("find gaps", mcps=[mcp])

    assert response.content == "final answer"
    assert response.tool_calls_made == 1
    assert response.tokens_in == 30
    assert response.tokens_out == 12
    # Tool dispatched through the owning MCP client under its real name.
    assert mcp.calls == [("papersflow.search", {"query": "ssl gaps"})]
    # Tools were exposed with sanitized names.
    first_request = fake.messages.requests[0]
    assert first_request["tools"] == [
        {
            "name": "papersflow__search",
            "description": "search papers",
            "input_schema": {"type": "object"},
        }
    ]
    # Second request carries the assistant echo and a tool_result reply.
    second_messages = fake.messages.requests[1]["messages"]
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[2] == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "tu_1",
                "content": "TOOL RESULT",
            }
        ],
    }


def test_anthropic_plain_completion_no_tools() -> None:
    final_turn = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    adapter, fake = _anthropic_adapter([final_turn])

    response = adapter.run("hi")

    assert response.content == "hello"
    assert response.tool_calls_made == 0
    assert "tools" not in fake.messages.requests[0]


def test_anthropic_error_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate an installed anthropic SDK so the lazy isinstance mapping runs.
    fake_sdk = ModuleType("anthropic")

    class AnthropicError(Exception):
        pass

    class APIConnectionError(AnthropicError):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    fake_sdk.AnthropicError = AnthropicError
    fake_sdk.APIConnectionError = APIConnectionError
    fake_sdk.APITimeoutError = APITimeoutError
    monkeypatch.setitem(sys.modules, "anthropic", fake_sdk)

    for sdk_error, anvesha_error in [
        (APITimeoutError("slow"), AdapterTimeoutError),
        (APIConnectionError("down"), AdapterConnectionError),
        (AnthropicError("bad request"), AdapterError),
    ]:
        adapter, _ = _anthropic_adapter([sdk_error])
        with pytest.raises(anvesha_error):
            adapter.run("q")


def test_anthropic_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    # sys.modules[name] = None makes "import anthropic" raise ImportError.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    adapter = AnthropicAdapter(
        AnthropicAdapterConfig(model="claude-sonnet-4-5", api_key="sk-ant")
    )
    with pytest.raises(AdapterError, match=r"anvesha\[cloud\]"):
        adapter.run("q")


# ---------------------------------------------------------------------------
# GoogleAdapter


class FakeGenerativeModel:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[tuple] = []

    def generate_content(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def test_google_plain_completion() -> None:
    fake = FakeGenerativeModel(
        SimpleNamespace(
            text="gemini says hi",
            usage_metadata=SimpleNamespace(
                prompt_token_count=4, candidates_token_count=6
            ),
        )
    )
    adapter = GoogleAdapter(
        GoogleAdapterConfig(model="gemini-2.0-flash", api_key="g-key"), client=fake
    )

    response = adapter.run("hi")

    assert response.content == "gemini says hi"
    assert response.tokens_in == 4
    assert response.tokens_out == 6
    assert fake.calls[0][0] == "hi"


def test_google_rejects_mcps() -> None:
    adapter = GoogleAdapter(
        GoogleAdapterConfig(model="gemini-2.0-flash", api_key="g-key"),
        client=FakeGenerativeModel(SimpleNamespace(text="x", usage_metadata=None)),
    )
    with pytest.raises(AdapterError, match="does not support MCP"):
        adapter.run("q", mcps=[FakeMCP()])


def test_google_maps_errors_to_adapter_error() -> None:
    adapter = GoogleAdapter(
        GoogleAdapterConfig(model="gemini-2.0-flash", api_key="g-key"),
        client=FakeGenerativeModel(RuntimeError("quota exceeded")),
    )
    with pytest.raises(AdapterError, match="quota exceeded"):
        adapter.run("q")


def test_google_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "google", None)
    monkeypatch.setitem(sys.modules, "google.generativeai", None)
    adapter = GoogleAdapter(
        GoogleAdapterConfig(model="gemini-2.0-flash", api_key="g-key")
    )
    with pytest.raises(AdapterError, match=r"anvesha\[cloud\]"):
        adapter.run("q")

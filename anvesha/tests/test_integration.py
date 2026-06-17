"""End-to-end integration: workspace + Phase + factory wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from anvesha.core.adapters import create_llm_client
from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.adapters.cloud import AnthropicAdapter
from anvesha.core.adapters.fallback import FallbackAdapter
from anvesha.core.adapters.local import NemotronAdapter, VLLMAdapter
from anvesha.core.config import load_config
from anvesha.core.exceptions import ConfigError, ToolNotPermittedError
from anvesha.core.phase import Phase, PhaseSpec, ValidationResult
from anvesha.workspace import ResearchProject, VersionLog, init_workspace


class FakeLLM(LLMClient):
    """Records the (input-injected) prompt and the restricted MCP clients it
    received, and returns canned markdown - the real adapters' inject step is
    reproduced so the input-file marker shows up in the captured prompt."""

    name = "fake"

    def __init__(self, content: str = "# Output\n\nok\n"):
        self._content = content
        self.seen_prompt: str | None = None
        self.seen_mcps: Sequence[MCPClient] = ()

    def run(self, prompt, mcps=(), input_files=()):
        self.seen_prompt = self.inject_input_files(prompt, input_files)
        self.seen_mcps = mcps
        return LLMResponse(content=self._content, tokens_in=1, tokens_out=1)


class DummyPhase(Phase):
    def task_description(self) -> str:
        return "Produce the dummy output."

    def validate_output(self, content: str) -> ValidationResult:
        if "Output" in content:
            return ValidationResult(ok=True)
        return ValidationResult(ok=False, errors=["missing heading"])


class FakeMCP:
    """Exposes one whitelisted and one non-whitelisted tool."""

    def list_tools(self) -> list[MCPToolDef]:
        empty = {"type": "object", "properties": {}}
        return [
            MCPToolDef("papersflow.search", "search", empty),
            MCPToolDef("github.read", "read", empty),
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        return f"called {name}"


def _spec_from_config(root: Path, phase_id: int) -> PhaseSpec:
    config = load_config(root)
    entry = config.phase(phase_id)
    return PhaseSpec(
        phase_id=phase_id,
        name=load_config().phase_name(phase_id),
        inputs=entry.inputs,
        optional_inputs=entry.optional_inputs,
        output=entry.output,
        tools_permitted=entry.tools,
    )


def test_phase_end_to_end(tmp_path):
    project = init_workspace(tmp_path, "demo")
    # Phase 3 reads the literature survey (required input).
    survey = project.resolve("phases/02_literature_survey.md")
    survey.parent.mkdir(parents=True, exist_ok=True)
    survey.write_text("SURVEY-CONTENT-MARKER\n", encoding="utf-8")
    # Give phase 3 a tool whitelist via project config.
    project.config_path.write_text(
        "project:\n  name: demo\n"
        "phases:\n  3:\n    inputs:\n      - phases/02_literature_survey.md\n"
        "    output: phases/03_research_gaps.md\n"
        "    tools:\n      - papersflow.*\n",
        encoding="utf-8",
    )

    spec = _spec_from_config(tmp_path, 3)
    assert spec.tools_permitted == ["papersflow.*"]
    llm = FakeLLM()
    phase = DummyPhase(project, VersionLog(project), llm, spec)

    fake_mcp = FakeMCP()
    written = phase.run(mcps=[fake_mcp])

    # output written and version-logged
    assert written == project.resolve("phases/03_research_gaps.md")
    assert written.read_text(encoding="utf-8").startswith("# Output")
    assert VersionLog(project).current(3) == "phases/03_research_gaps.md"

    # the input-file content reached the prompt
    assert "SURVEY-CONTENT-MARKER" in llm.seen_prompt
    assert "PHASE: 3 — Research Gaps" in llm.seen_prompt

    # tool restriction held: only the whitelisted tool is visible, and the
    # non-whitelisted one raises through the restriction wrapper
    restricted = llm.seen_mcps[0]
    assert [t.name for t in restricted.list_tools()] == ["papersflow.search"]
    assert restricted.call_tool("papersflow.search", {}) == "called papersflow.search"
    with pytest.raises(ToolNotPermittedError):
        restricted.call_tool("github.read", {})


def test_phase3_discovery_mode_runs_without_optional_approach(tmp_path):
    """Phase 3 with the default spec must run in Discovery Mode (no
    03_approach.md present) - the optional input is not required."""
    project = init_workspace(tmp_path, "demo")
    survey = project.resolve("phases/02_literature_survey.md")
    survey.write_text("SURVEY\n", encoding="utf-8")
    # No phases/03_approach.md on disk -> Discovery Mode.

    spec = _spec_from_config(tmp_path, 3)
    assert spec.inputs == ["phases/02_literature_survey.md"]
    assert spec.optional_inputs == ["phases/03_approach.md"]

    llm = FakeLLM()
    written = DummyPhase(project, VersionLog(project), llm, spec).run()
    assert written.name == "03_research_gaps.md"
    # only the present required input was injected; the absent optional one
    # is not listed in INPUTS
    assert "phases/02_literature_survey.md" in llm.seen_prompt
    assert "phases/03_approach.md" not in llm.seen_prompt


def test_phase3_refinement_mode_includes_present_approach(tmp_path):
    """When 03_approach.md exists it is picked up (Refinement Mode)."""
    project = init_workspace(tmp_path, "demo")
    project.resolve("phases/02_literature_survey.md").write_text("S\n", encoding="utf-8")
    project.resolve("phases/03_approach.md").write_text("APPROACH-MARKER\n", encoding="utf-8")

    spec = _spec_from_config(tmp_path, 3)
    llm = FakeLLM()
    DummyPhase(project, VersionLog(project), llm, spec).run()
    assert "phases/03_approach.md" in llm.seen_prompt
    assert "APPROACH-MARKER" in llm.seen_prompt


def test_phase_rerun_versions_and_logs(tmp_path):
    project = init_workspace(tmp_path, "demo")
    survey = project.resolve("phases/02_literature_survey.md")
    survey.write_text("x\n", encoding="utf-8")
    spec = PhaseSpec(
        phase_id=3,
        name="Research Gaps",
        inputs=["phases/02_literature_survey.md"],
        output="phases/03_research_gaps.md",
    )
    vlog = VersionLog(project)
    DummyPhase(project, vlog, FakeLLM(), spec).run()
    second = DummyPhase(project, vlog, FakeLLM(), spec).run()
    assert second.name == "03_research_gaps_v2.md"
    assert vlog.current(3) == "phases/03_research_gaps_v2.md"


def test_phase_validation_failure_writes_nothing(tmp_path):
    from anvesha.core.exceptions import PhaseValidationError

    project = init_workspace(tmp_path, "demo")
    survey = project.resolve("phases/02_literature_survey.md")
    survey.write_text("x\n", encoding="utf-8")
    spec = PhaseSpec(
        phase_id=3,
        name="Research Gaps",
        inputs=["phases/02_literature_survey.md"],
        output="phases/03_research_gaps.md",
    )
    # FakeLLM returns content without "Output" -> validate_output fails.
    llm = FakeLLM(content="nope\n")
    with pytest.raises(PhaseValidationError):
        DummyPhase(project, VersionLog(project), llm, spec).run()
    assert not project.resolve("phases/03_research_gaps.md").exists()
    assert VersionLog(project).current(3) is None


def test_factory_builds_vllm_with_model_alias():
    cfg = load_config(
        cli_overrides={
            "llm": {
                "default_backend": "v",
                "backends": {
                    "v": {
                        "type": "vllm",
                        "model": "openai/gpt-oss-120b",
                        "harmony_format": True,
                    }
                },
            }
        }
    )
    client = create_llm_client(cfg, 3)
    assert isinstance(client, VLLMAdapter)
    assert client.config.model_name == "openai/gpt-oss-120b"
    assert client.config.harmony_format is True
    # S13.5 reference default so the GPU allocator can budget VRAM.
    assert client.reported_vram_gb == 40.0


def test_factory_builds_fallback_composition():
    cfg = load_config(
        cli_overrides={
            "llm": {
                "default_backend": "robust",
                "backends": {
                    "local": {"type": "nemotron"},
                    "cloud": {"type": "anthropic", "model": "claude-x"},
                    "robust": {
                        "type": "fallback",
                        "primary": "local",
                        "fallback": "cloud",
                        "fallback_on": ["timeout"],
                    },
                },
            }
        }
    )
    client = create_llm_client(cfg, 3)
    assert isinstance(client, FallbackAdapter)
    assert isinstance(client.config.primary, NemotronAdapter)
    assert isinstance(client.config.fallback, AnthropicAdapter)
    assert client.config.fallback_on == ["timeout"]
    # name/vram delegate to the primary
    assert client.name == client.config.primary.name


def test_factory_unknown_type_raises():
    cfg = load_config(
        cli_overrides={
            "llm": {"default_backend": "x", "backends": {"x": {"type": "bogus"}}}
        }
    )
    with pytest.raises(ConfigError):
        create_llm_client(cfg, 3)


def test_factory_fallback_self_reference_raises():
    cfg = load_config(
        cli_overrides={
            "llm": {
                "default_backend": "a",
                "backends": {
                    "a": {"type": "fallback", "primary": "a", "fallback": "a"}
                },
            }
        }
    )
    with pytest.raises(ConfigError):
        create_llm_client(cfg, 3)


def test_factory_bad_field_wrapped_as_config_error():
    cfg = load_config(
        cli_overrides={
            "llm": {
                "default_backend": "v",
                "backends": {
                    "v": {"type": "vllm", "timeout_seconds": "not-an-int"}
                },
            }
        }
    )
    with pytest.raises(ConfigError):
        create_llm_client(cfg, 3)

"""Tests for the Phase base class (ANVESHA.md S7.1) and PipelineHooks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.exceptions import PhaseValidationError, ToolNotPermittedError
from anvesha.core.hooks import PipelineHooks
from anvesha.core.phase import Phase, PhaseSpec, RestrictedMCPClient, ValidationResult
from anvesha.workspace.project import ResearchProject, init_workspace
from anvesha.workspace.version_log import VersionLog

CANNED_OUTPUT = "# Research Gaps\n\nSome canned markdown.\n"


class FakeLLM(LLMClient):
    """Records the call and returns canned markdown."""

    name = "fake"

    def __init__(self, content: str = CANNED_OUTPUT) -> None:
        self.content = content
        self.last_prompt: str | None = None
        self.last_mcps: Sequence[MCPClient] | None = None
        self.last_input_files: Sequence[Path] | None = None

    def run(
        self,
        prompt: str,
        mcps: Sequence[MCPClient] = (),
        input_files: Sequence[Path] = (),
    ) -> LLMResponse:
        self.last_prompt = prompt
        self.last_mcps = list(mcps)
        self.last_input_files = list(input_files)
        return LLMResponse(content=self.content)


class FakeMCP:
    """In-memory MCPClient exposing two namespaced tools."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def list_tools(self) -> list[MCPToolDef]:
        return [
            MCPToolDef("papersflow.search", "search papers", {}),
            MCPToolDef("scholar.lookup", "lookup citations", {}),
        ]

    def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return f"result-of-{name}"


class DummyPhase(Phase):
    validation = ValidationResult(ok=True)

    def task_description(self) -> str:
        return "Do the dummy task exactly as specified."

    def validate_output(self, content: str) -> ValidationResult:
        return self.validation


class FailingPhase(DummyPhase):
    validation = ValidationResult(ok=False, errors=["missing Gap table", "no header"])


@pytest.fixture()
def project(tmp_path: Path) -> ResearchProject:
    proj = init_workspace(tmp_path / "proj", "Demo Project")
    (proj.root / "phases" / "02_literature_survey.md").write_text(
        "# Survey\n", encoding="utf-8"
    )
    return proj


def make_spec(**overrides) -> PhaseSpec:
    base = dict(
        phase_id=3,
        name="Research Gaps",
        inputs=["phases/02_literature_survey.md"],
        optional_inputs=["notes/extra.md"],
        output="phases/03_research_gaps.md",
        tools_permitted=["papersflow.*"],
    )
    base.update(overrides)
    return PhaseSpec(**base)


def make_phase(
    project: ResearchProject,
    spec: PhaseSpec | None = None,
    llm: FakeLLM | None = None,
    cls: type[DummyPhase] = DummyPhase,
) -> DummyPhase:
    return cls(project, VersionLog(project), llm or FakeLLM(), spec or make_spec())


# --- build_prompt -----------------------------------------------------------


def test_build_prompt_matches_charter_template_exactly(project: ResearchProject):
    phase = make_phase(project)
    expected = (
        "PHASE: 3 — Research Gaps\n"
        "\n"
        "OPERATING RULES\n"
        "- Read ONLY the files listed in INPUTS.\n"
        "- Produce EXACTLY ONE output file at the path in OUTPUT.\n"
        "- Follow the output template precisely.\n"
        "- Do not start, plan, or hint at the next phase.\n"
        "- If INPUTS are missing or incomplete, stop and ask.\n"
        "\n"
        "INPUTS\n"
        "- phases/02_literature_survey.md\n"
        "\n"
        "OUTPUT\n"
        "- phases/03_research_gaps.md\n"
        "\n"
        "TOOLS PERMITTED\n"
        "- papersflow.*\n"
        "\n"
        "TASK\n"
        "Do the dummy task exactly as specified."
    )
    assert phase.build_prompt() == expected


def test_build_prompt_includes_optional_input_when_present(project: ResearchProject):
    notes = project.root / "notes" / "extra.md"
    notes.parent.mkdir()
    notes.write_text("notes\n", encoding="utf-8")
    prompt = make_phase(project).build_prompt()
    assert "- phases/02_literature_survey.md\n- notes/extra.md\n" in prompt


# --- validate_inputs --------------------------------------------------------


def test_missing_required_inputs_named_in_error(project: ResearchProject):
    spec = make_spec(inputs=["phases/02_literature_survey.md", "missing_a.md", "missing_b.md"])
    phase = make_phase(project, spec=spec)
    with pytest.raises(PhaseValidationError) as exc_info:
        phase.run()
    assert "missing_a.md" in str(exc_info.value)
    assert "missing_b.md" in str(exc_info.value)
    assert "02_literature_survey" not in str(exc_info.value)


def test_optional_input_absent_still_runs(project: ResearchProject):
    llm = FakeLLM()
    written = make_phase(project, llm=llm).run()
    assert written.is_file()
    assert llm.last_input_files == [project.resolve("phases/02_literature_survey.md")]


def test_optional_input_present_is_passed_to_llm(project: ResearchProject):
    notes = project.root / "notes" / "extra.md"
    notes.parent.mkdir()
    notes.write_text("notes\n", encoding="utf-8")
    llm = FakeLLM()
    make_phase(project, llm=llm).run()
    assert llm.last_input_files == [
        project.resolve("phases/02_literature_survey.md"),
        project.resolve("notes/extra.md"),
    ]


# --- tool restriction (ANVESHA.md S12.8) ------------------------------------


def test_restrict_tools_filters_and_blocks(project: ResearchProject):
    phase = make_phase(project)
    fake_mcp = FakeMCP()
    (restricted,) = phase.restrict_tools([fake_mcp])
    assert isinstance(restricted, RestrictedMCPClient)
    assert [t.name for t in restricted.list_tools()] == ["papersflow.search"]
    assert restricted.call_tool("papersflow.search", {"q": "x"}) == (
        "result-of-papersflow.search"
    )
    with pytest.raises(ToolNotPermittedError):
        restricted.call_tool("scholar.lookup", {})
    assert fake_mcp.calls == [("papersflow.search", {"q": "x"})]


def test_empty_whitelist_blocks_all_tools(project: ResearchProject):
    phase = make_phase(project, spec=make_spec(tools_permitted=[]))
    (restricted,) = phase.restrict_tools([FakeMCP()])
    assert restricted.list_tools() == []
    with pytest.raises(ToolNotPermittedError):
        restricted.call_tool("papersflow.search", {})


def test_run_passes_restricted_clients_to_llm(project: ResearchProject):
    llm = FakeLLM()
    make_phase(project, llm=llm).run(mcps=[FakeMCP()])
    assert llm.last_mcps is not None and len(llm.last_mcps) == 1
    assert isinstance(llm.last_mcps[0], RestrictedMCPClient)


# --- validate_output failure -------------------------------------------------


def test_validation_failure_writes_nothing(project: ResearchProject):
    phase = make_phase(project, cls=FailingPhase)
    with pytest.raises(PhaseValidationError) as exc_info:
        phase.run()
    assert "missing Gap table" in str(exc_info.value)
    assert "no header" in str(exc_info.value)
    assert not project.resolve("phases/03_research_gaps.md").exists()
    assert VersionLog(project).current(3) is None


# --- happy path and versioned re-run -----------------------------------------


def test_happy_path_writes_records_and_returns_path(project: ResearchProject):
    llm = FakeLLM()
    written = make_phase(project, llm=llm).run()
    assert written == project.resolve("phases/03_research_gaps.md")
    assert written.is_absolute()
    assert written.read_text(encoding="utf-8") == CANNED_OUTPUT
    assert VersionLog(project).current(3) == "phases/03_research_gaps.md"
    assert llm.last_prompt is not None and llm.last_prompt.startswith("PHASE: 3")


def test_rerun_writes_v2_and_updates_version_log(project: ResearchProject):
    make_phase(project).run()
    second = make_phase(project).run()
    assert second == project.resolve("phases/03_research_gaps_v2.md")
    assert second.is_file()
    assert VersionLog(project).current(3) == "phases/03_research_gaps_v2.md"


# --- PipelineHooks ------------------------------------------------------------


def test_hooks_fire_in_registration_order_with_payload():
    hooks = PipelineHooks()
    seen: list[tuple[str, str, dict]] = []
    hooks.add_pre_agent(lambda name, payload: seen.append(("pre1", name, payload)))
    hooks.add_pre_agent(lambda name, payload: seen.append(("pre2", name, payload)))
    hooks.add_post_agent(lambda name, payload: seen.append(("post1", name, payload)))
    hooks.add_post_agent(lambda name, payload: seen.append(("post2", name, payload)))

    hooks.fire_pre_agent("gap_finder", {"round": 1})
    hooks.fire_post_agent("gap_finder", {"ok": True})

    assert seen == [
        ("pre1", "gap_finder", {"round": 1}),
        ("pre2", "gap_finder", {"round": 1}),
        ("post1", "gap_finder", {"ok": True}),
        ("post2", "gap_finder", {"ok": True}),
    ]


def test_hook_exceptions_propagate():
    hooks = PipelineHooks()

    def boom(name: str, payload: dict) -> None:
        raise RuntimeError("hook failed")

    hooks.add_pre_agent(boom)
    with pytest.raises(RuntimeError, match="hook failed"):
        hooks.fire_pre_agent("x", {})

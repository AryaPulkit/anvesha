"""The Phase base class - the central abstraction of the pipeline (ANVESHA.md S7.1).

Every phase has the same shape: declare inputs, declare permitted tools,
build a prompt, call the LLM, validate the output, commit to disk. This base
class implements that lifecycle; subclasses customise only the task-specific
parts (``task_description`` and ``validate_output``, per S7.3).

Core stays phase-agnostic (S12.2): nothing here knows about specific phases.
All file I/O flows through ``workspace/phase_io.py`` (S12.3) and tool
whitelists are enforced before the LLM is invoked (S12.8).
"""

from __future__ import annotations

import fnmatch
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field

from anvesha.core.adapters.base import LLMClient, LLMResponse, MCPClient, MCPToolDef
from anvesha.core.exceptions import PhaseValidationError, ToolNotPermittedError
from anvesha.workspace import phase_io
from anvesha.workspace.project import ResearchProject
from anvesha.workspace.version_log import VersionLog

logger = logging.getLogger(__name__)


class PhaseSpec(BaseModel):
    """Declarative config for one phase, loaded from .anvesha/config.yaml (S7.1)."""

    phase_id: int
    name: str
    inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    output: str
    tools_permitted: list[str] = Field(default_factory=list)
    template: str | None = None
    exit_criteria: list[str] = Field(default_factory=list)
    #: Phase-specific tuning, e.g. Phase 1's max_papers/thresholds (S3.2).
    options: dict[str, Any] = Field(default_factory=dict)
    #: Phase-specific filter config, e.g. Phase 1's conferences/domains/years.
    filters: dict[str, Any] = Field(default_factory=dict)


@dataclass
class ValidationResult:
    """Outcome of a phase's structural output checks."""

    ok: bool
    errors: list[str] = field(default_factory=list)


class RestrictedMCPClient:
    """Wraps an :class:`MCPClient`, enforcing a phase's tool whitelist (S12.8).

    ``patterns`` are ``fnmatch`` globs over namespaced tool names (e.g.
    ``"papersflow.*"``). An empty whitelist permits NO tools.
    """

    def __init__(self, inner: MCPClient, patterns: Sequence[str]) -> None:
        self._inner = inner
        self._patterns = list(patterns)

    def _permitted(self, name: str) -> bool:
        return any(fnmatch.fnmatch(name, pattern) for pattern in self._patterns)

    def list_tools(self) -> list[MCPToolDef]:
        return [t for t in self._inner.list_tools() if self._permitted(t.name)]

    def call_tool(self, name: str, arguments: dict) -> str:
        if not self._permitted(name):
            raise ToolNotPermittedError(
                f"Tool {name!r} is not in this phase's whitelist "
                f"(permitted: {self._patterns})"
            )
        return self._inner.call_tool(name, arguments)


class Phase(ABC):
    """Owns the lifecycle of one phase execution (ANVESHA.md S7.1)."""

    def __init__(
        self,
        project: ResearchProject,
        version_log: VersionLog,
        llm: LLMClient,
        spec: PhaseSpec,
    ) -> None:
        self.project = project
        self.version_log = version_log
        self.llm = llm
        self.spec = spec

    @abstractmethod
    def task_description(self) -> str:
        """The phase-specific paragraph that goes into the prompt."""

    @abstractmethod
    def validate_output(self, content: str) -> ValidationResult:
        """Phase-specific structural checks on the produced Markdown."""

    def validate_inputs(self) -> None:
        """Raise :class:`PhaseValidationError` naming ALL missing required inputs."""
        missing = [
            rel for rel in self.spec.inputs if not phase_io.exists(self.project, rel)
        ]
        if missing:
            raise PhaseValidationError(
                f"Phase {self.spec.phase_id} ({self.spec.name}) is missing required "
                f"input files: {', '.join(missing)}"
            )

    def _present_inputs(self) -> list[str]:
        """Required inputs plus those optional inputs that exist."""
        present = list(self.spec.inputs)
        present.extend(
            rel
            for rel in self.spec.optional_inputs
            if phase_io.exists(self.project, rel)
        )
        return present

    def build_prompt(self) -> str:
        """Assemble the strict S7.1 prompt template; not customisable per phase."""
        lines = [
            f"PHASE: {self.spec.phase_id} — {self.spec.name}",
            "",
            "OPERATING RULES",
            "- Read ONLY the files listed in INPUTS.",
            "- Produce EXACTLY ONE output file at the path in OUTPUT.",
            "- Follow the output template precisely.",
            "- Do not start, plan, or hint at the next phase.",
            "- If INPUTS are missing or incomplete, stop and ask.",
            "",
            "INPUTS",
            *(f"- {rel}" for rel in self._present_inputs()),
            "",
            "OUTPUT",
            f"- {self.spec.output}",
            "",
            "TOOLS PERMITTED",
            *(f"- {tool}" for tool in self.spec.tools_permitted),
            "",
            "TASK",
            self.task_description(),
        ]
        return "\n".join(lines)

    def restrict_tools(self, mcps: Sequence[MCPClient]) -> list[RestrictedMCPClient]:
        """Wrap each MCP client so only whitelisted tools are visible/callable."""
        return [
            RestrictedMCPClient(mcp, self.spec.tools_permitted) for mcp in mcps
        ]

    def _resolve_output_path(self) -> Path:
        """Next output path, with automatic _v2, _v3 versioning (ANVESHA.md S11)."""
        return phase_io.next_versioned_path(self.project, self.spec.output)

    def run(self, mcps: Sequence[MCPClient] = ()) -> Path:
        """Orchestrate the full phase lifecycle; returns the written output path."""
        self.validate_inputs()
        prompt = self.build_prompt()
        restricted = self.restrict_tools(mcps)
        input_files = [self.project.resolve(rel) for rel in self._present_inputs()]
        logger.info(
            "Running phase %d (%s) with %d input file(s)",
            self.spec.phase_id,
            self.spec.name,
            len(input_files),
        )
        response: LLMResponse = self.llm.run(
            prompt, mcps=restricted, input_files=input_files
        )
        result = self.validate_output(response.content)
        if not result.ok:
            raise PhaseValidationError(
                f"Phase {self.spec.phase_id} ({self.spec.name}) output failed "
                f"validation: {'; '.join(result.errors)}"
            )
        written = phase_io.write_phase_output(
            self.project, self.spec.output, response.content
        )
        self.version_log.record(
            self.spec.phase_id,
            written.relative_to(self.project.root).as_posix(),
        )
        logger.info("Phase %d output written to %s", self.spec.phase_id, written)
        return written

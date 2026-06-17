"""Project-wide configuration: assembly, defaults, and loading.

``PipelineConfig`` mirrors ``.anvesha/config.yaml`` (ANVESHA.md S8).
Precedence (S8.1): package defaults -> config.yaml -> environment variables
-> CLI flags. Later sources override earlier ones.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, Field, ValidationError

from anvesha.core.config.gpu_config import GpuConfig
from anvesha.core.config.model_config import LlmConfig
from anvesha.core.config.rubric_config import RubricConfig
from anvesha.core.config.search_config import SearchConfig
from anvesha.core.exceptions import ConfigError

#: Default phase inputs/outputs (ANVESHA.md S8.4). Projects override these in
#: `.anvesha/config.yaml`. `phases/03_approach.md` is optional - its absence
#: switches Phase 3 to Discovery Mode.
DEFAULT_PHASE_IO: dict[int, dict[str, Any]] = {
    1: {
        "inputs": ["README.md"],
        "output": "filtered_literature/01_filtered_literature.md",
        "tools": [
            "papersflow.*",
            "paper-search.search_papers",
            "paper-search.download_with_fallback",
            "filesystem.*",
        ],
    },
    2: {
        "inputs": ["filtered_literature/01_filtered_literature.md"],
        "output": "phases/02_literature_survey.md",
        "tools": ["filesystem.*"],
    },
    3: {
        "inputs": ["phases/02_literature_survey.md"],
        # phases/03_approach.md is OPTIONAL: present -> Refinement Mode,
        # absent -> Discovery Mode (ANVESHA.md S3, S8.4).
        "optional_inputs": ["phases/03_approach.md"],
        "output": "phases/03_research_gaps.md",
        "tools": [
            "papersflow.verify_citation",
            "papersflow.expand_citation_graph",
            "web_search.*",
            "filesystem.*",
        ],
    },
    4: {
        "inputs": ["phases/02_literature_survey.md", "phases/03_research_gaps.md"],
        "output": "phases/04_base_paper.md",
    },
    5: {
        "inputs": ["phases/03_research_gaps.md", "phases/04_base_paper.md"],
        "output": "phases/05_ideas.md",
    },
    6: {
        "inputs": ["phases/04_base_paper.md", "phases/05_ideas.md"],
        "output": "phases/06_experiment_plan.md",
    },
    7: {
        "inputs": ["phases/06_experiment_plan.md"],
        "output": "phases/07_implementation_notes.md",
    },
    8: {"inputs": [], "output": "phases/08_results.txt"},
    9: {
        "inputs": ["phases/06_experiment_plan.md", "phases/08_results.txt"],
        "output": "phases/09_evaluation.md",
    },
    10: {"inputs": ["phases/09_evaluation.md"], "output": "phases/10_future_ideation.md"},
    11: {
        "inputs": ["phases/09_evaluation.md", "phases/10_future_ideation.md"],
        "output": "phases/11_loop_decision.md",
    },
}

PHASE_NAMES: dict[int, str] = {
    1: "Filter Literature",
    2: "Literature Survey",
    3: "Research Gaps",
    4: "Base Paper Selection",
    5: "Idea Generation",
    6: "Experiment Planning",
    7: "Implementation",
    8: "Training",
    9: "Evaluation & Analysis",
    10: "Future Ideation",
    11: "Loop Decision Gate",
}


class ProjectConfig(BaseModel):
    name: str = "research-project"
    #: Path to the researcher's separate code repository (never written to).
    code_repo: str | None = None


class PhaseEntry(BaseModel):
    """One entry under ``phases:`` - paths, tool whitelist, and options."""

    inputs: list[str] = Field(default_factory=list)
    #: Inputs that may be absent (e.g. Phase 3's optional 03_approach.md).
    optional_inputs: list[str] = Field(default_factory=list)
    output: str = ""
    tools: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    filters: dict[str, Any] = Field(default_factory=dict)


class CheckpointConfig(BaseModel):
    """ANVESHA.md S8.3 ``checkpoint:`` block + retry settings (S10)."""

    enabled: bool = True
    granularity: Literal["agent", "iteration", "gap"] = "iteration"
    storage: Literal["filesystem", "sqlite"] = "filesystem"
    dir: str = ".anvesha/checkpoints"
    max_agent_retries: int = 3
    retry_backoff_seconds: int = 5


class GpuBoxConfig(BaseModel):
    host: str | None = None
    ssh_key: str | None = None


class TrainingConfig(BaseModel):
    gpu_box: GpuBoxConfig = Field(default_factory=GpuBoxConfig)
    remote_workdir: str | None = None


class PipelineConfig(BaseModel):
    """The fully-resolved project configuration injected into phases."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    phases: dict[int, PhaseEntry] = Field(default_factory=dict)
    gpu: GpuConfig = Field(default_factory=GpuConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    rubric: RubricConfig = Field(default_factory=RubricConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    def phase(self, phase_id: int) -> PhaseEntry:
        if not 1 <= phase_id <= 11:
            raise ConfigError(f"phase id must be 1-11, got {phase_id}")
        if phase_id in self.phases:
            return self.phases[phase_id]
        return PhaseEntry(**DEFAULT_PHASE_IO[phase_id])

    @staticmethod
    def phase_name(phase_id: int) -> str:
        if phase_id not in PHASE_NAMES:
            raise ConfigError(f"phase id must be 1-11, got {phase_id}")
        return PHASE_NAMES[phase_id]


def default_config_dict() -> dict[str, Any]:
    """The built-in package defaults as a plain dict (precedence layer 1)."""
    return {"phases": {pid: dict(io) for pid, io in DEFAULT_PHASE_IO.items()}}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge mappings; scalars and lists in ``override`` win."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_phase_keys(data: dict) -> None:
    """YAML may parse phase ids as str (quoted) or int; normalize to int so
    deep-merging against the int-keyed defaults works."""
    phases = data.get("phases")
    if isinstance(phases, dict):
        data["phases"] = {
            int(key) if isinstance(key, str) and key.isdigit() else key: value
            for key, value in phases.items()
        }


def _set_nested(data: dict, keys: tuple[str, ...], value: Any) -> None:
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


#: Environment-variable overrides (precedence layer 3, ANVESHA.md S8.1).
#: API keys are not listed here - cloud adapters read them directly.
_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    "ANVESHA_DEFAULT_BACKEND": ("llm", "default_backend"),
    "ANVESHA_CHECKPOINT_DIR": ("checkpoint", "dir"),
    "ANVESHA_CODE_REPO": ("project", "code_repo"),
}


def load_config(
    project_root: Path | str | None = None,
    *,
    config_path: Path | str | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Load and resolve configuration with the S8.1 precedence chain.

    With neither ``project_root`` nor ``config_path``, returns pure package
    defaults (plus env/CLI layers). With a path, the config file must exist.
    """
    data = default_config_dict()

    path: Path | None = None
    if config_path is not None:
        path = Path(config_path)
    elif project_root is not None:
        path = Path(project_root) / ".anvesha" / "config.yaml"

    if path is not None:
        if not path.is_file():
            raise ConfigError(
                f"config file not found: {path} (run `anvesha init` to create "
                "a project workspace)"
            )
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ConfigError(f"config root must be a mapping: {path}")
        _normalize_phase_keys(loaded)
        data = _deep_merge(data, loaded)

    env_map = os.environ if env is None else env
    for var, keys in _ENV_OVERRIDES.items():
        if var in env_map:
            _set_nested(data, keys, env_map[var])

    if cli_overrides:
        data = _deep_merge(data, cli_overrides)

    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration: {exc}") from exc

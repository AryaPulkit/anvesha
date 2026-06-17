"""LLM backend declarations and role assignments (ANVESHA.md S8.3 ``llm:`` block)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anvesha.core.exceptions import ConfigError


class BackendConfig(BaseModel):
    """One named entry under ``llm.backends``.

    Backend-specific fields (``base_url``, ``model_name``/``model``, ...) are
    preserved as extras and consumed by the adapter factory.
    """

    model_config = ConfigDict(extra="allow")

    type: str

    def extra_fields(self) -> dict:
        return dict(self.__pydantic_extra__ or {})


class LlmConfig(BaseModel):
    """The ``llm:`` section of ``.anvesha/config.yaml``."""

    default_backend: str | None = None
    backends: dict[str, BackendConfig] = Field(default_factory=dict)
    phase_overrides: dict[int, str] = Field(default_factory=dict)
    #: e.g. {"phase_03": {"generator": "vllm_nemotron", "critic": "vllm_gptoss"}}
    role_assignments: dict[str, dict[str, str]] = Field(default_factory=dict)

    def resolve_backend_name(self, phase_id: int, override: str | None = None) -> str:
        """Backend selection precedence (ANVESHA.md S7.2):
        CLI ``--backend`` flag > ``phase_overrides[N]`` > ``default_backend``.
        """
        name = override or self.phase_overrides.get(phase_id) or self.default_backend
        if not name:
            raise ConfigError(
                f"no LLM backend configured for phase {phase_id}: "
                "set llm.default_backend in .anvesha/config.yaml"
            )
        if name not in self.backends:
            raise ConfigError(
                f"unknown LLM backend {name!r}: not declared under llm.backends"
            )
        return name


class ModelConfig(BaseModel):
    """Resolved role -> backend assignment for a multi-agent phase.

    Decision 1 (research_gap_pipeline_implementation.md S3): the Critic must
    use a different backend from the Generator so their blind spots do not
    correlate. ``analyzer`` defaults to the generator backend (S10).
    """

    generator: str
    critic: str
    judge: str | None = None
    analyzer: str | None = None
    merger: str | None = None
    #: Per-role reasoning-level overrides, e.g. {"analyzer": "low"}.
    reasoning_overrides: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _critic_must_differ(self) -> "ModelConfig":
        if self.critic == self.generator:
            raise ValueError(
                "critic backend must differ from generator backend "
                "(research_gap_pipeline_implementation.md, Decision 1)"
            )
        return self

    @property
    def analyzer_backend(self) -> str:
        return self.analyzer or self.generator

    @property
    def judge_backend(self) -> str:
        return self.judge or self.generator

    @property
    def merger_backend(self) -> str:
        return self.merger or self.analyzer_backend

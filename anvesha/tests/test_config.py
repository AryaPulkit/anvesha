"""Tests for core/config: precedence, defaults, validation (ANVESHA.md S8)."""

from __future__ import annotations

import pytest

from anvesha.core.config import (
    GpuConfig,
    LlmConfig,
    ModelConfig,
    ModelNodeAssignment,
    PipelineConfig,
    load_config,
)
from anvesha.core.exceptions import ConfigError


def _write_config(root, body: str):
    cfg_dir = root / ".anvesha"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.yaml").write_text(body, encoding="utf-8")
    return cfg_dir / "config.yaml"


def test_defaults_only():
    cfg = load_config()
    assert cfg.phase(2).output == "phases/02_literature_survey.md"
    assert cfg.checkpoint.granularity == "iteration"
    assert cfg.gpu.execution_mode == "sequential"


def test_default_phase_paths_table():
    """Spot-check the S8.4 default phase paths across all eleven phases."""
    cfg = load_config()
    assert cfg.phase(1).output == "filtered_literature/01_filtered_literature.md"
    # 02_literature_survey.md is required; 03_approach.md is optional (S3/S8.4).
    assert cfg.phase(3).inputs == ["phases/02_literature_survey.md"]
    assert cfg.phase(3).optional_inputs == ["phases/03_approach.md"]
    assert cfg.phase(8).output == "phases/08_results.txt"
    assert cfg.phase(11).output == "phases/11_loop_decision.md"
    assert PipelineConfig.phase_name(9) == "Evaluation & Analysis"


def test_default_tool_whitelists_present():
    """Package defaults ship per-phase tool whitelists (ANVESHA.md S8.2/S8.3)."""
    cfg = load_config()
    assert cfg.phase(1).tools == [
        "papersflow.*",
        "paper-search.search_papers",
        "paper-search.download_with_fallback",
        "filesystem.*",
    ]
    assert cfg.phase(2).tools == ["filesystem.*"]
    assert "web_search.*" in cfg.phase(3).tools


def test_yaml_overrides_defaults(tmp_path):
    path = _write_config(
        tmp_path,
        "project:\n  name: demo\n"
        "checkpoint:\n  granularity: gap\n",
    )
    cfg = load_config(config_path=path)
    assert cfg.project.name == "demo"
    assert cfg.checkpoint.granularity == "gap"
    # untouched defaults remain
    assert cfg.phase(2).output == "phases/02_literature_survey.md"


def test_precedence_env_over_yaml_and_cli_over_env(tmp_path):
    path = _write_config(
        tmp_path, "llm:\n  default_backend: from_yaml\n"
    )
    cfg = load_config(
        config_path=path,
        env={"ANVESHA_DEFAULT_BACKEND": "from_env"},
    )
    assert cfg.llm.default_backend == "from_env"

    cfg2 = load_config(
        config_path=path,
        env={"ANVESHA_DEFAULT_BACKEND": "from_env"},
        cli_overrides={"llm": {"default_backend": "from_cli"}},
    )
    assert cfg2.llm.default_backend == "from_cli"


def test_phase_key_normalization_quoted_ids(tmp_path):
    """YAML quoted phase ids ("3") must merge against the int-keyed defaults."""
    path = _write_config(
        tmp_path,
        'phases:\n  "3":\n    output: custom/gaps.md\n',
    )
    cfg = load_config(config_path=path)
    assert cfg.phase(3).output == "custom/gaps.md"


def test_invalid_yaml_raises_config_error(tmp_path):
    path = _write_config(tmp_path, "project: [unbalanced\n")
    with pytest.raises(ConfigError):
        load_config(config_path=path)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(config_path=tmp_path / ".anvesha" / "config.yaml")


def test_config_root_must_be_mapping(tmp_path):
    path = _write_config(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ConfigError):
        load_config(config_path=path)


def test_model_config_critic_must_differ():
    ModelConfig(generator="a", critic="b")  # ok
    with pytest.raises(ValueError):
        ModelConfig(generator="x", critic="x")


def test_model_config_role_defaults():
    mc = ModelConfig(generator="gen", critic="crit")
    assert mc.analyzer_backend == "gen"
    assert mc.judge_backend == "gen"
    assert mc.merger_backend == "gen"


def test_resolve_backend_name_precedence():
    llm = LlmConfig(
        default_backend="base",
        backends={
            "base": {"type": "vllm"},
            "perphase": {"type": "vllm"},
            "flag": {"type": "vllm"},
        },
        phase_overrides={3: "perphase"},
    )
    assert llm.resolve_backend_name(2) == "base"
    assert llm.resolve_backend_name(3) == "perphase"
    assert llm.resolve_backend_name(3, "flag") == "flag"


def test_resolve_backend_name_errors():
    llm = LlmConfig(backends={"a": {"type": "vllm"}})
    with pytest.raises(ConfigError):
        llm.resolve_backend_name(1)  # no default set
    llm2 = LlmConfig(default_backend="missing", backends={"a": {"type": "vllm"}})
    with pytest.raises(ConfigError):
        llm2.resolve_backend_name(1)  # unknown backend


def test_gpu_manual_mode_requires_assignments():
    with pytest.raises(ValueError):
        GpuConfig(mode="manual")
    GpuConfig(mode="manual", manual_assignments=ModelNodeAssignment(generator_nodes=[0]))


def test_gpu_node_range_validation():
    with pytest.raises(ValueError):
        GpuConfig(available_node_range=(5, 2))


def test_phase_id_out_of_range():
    cfg = load_config()
    with pytest.raises(ConfigError):
        cfg.phase(12)

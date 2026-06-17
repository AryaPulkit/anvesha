"""Tests for anvesha.core.gpu (probe, allocator, manager, scheduler).

All offline: fake nvidia-smi runners, synthetic devices, fake adapters.
"""

from __future__ import annotations

import pytest

from anvesha.core.config import GpuConfig, ModelNodeAssignment
from anvesha.core.exceptions import (
    ConfigError,
    GpuNodeUnavailableError,
    InsufficientVramError,
)
from anvesha.core.gpu import GpuDevice, GpuManager, GpuScheduler, allocate, probe_devices


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def device(device_id: int, free: float, total: float = 80.0) -> GpuDevice:
    return GpuDevice(device_id=device_id, name="H100", total_vram_gb=total, free_vram_gb=free)


class FakeAdapter:
    """Duck-typed adapter recording load/unload/run calls."""

    def __init__(self, name: str, vram: float | None = 10.0, run_error: Exception | None = None):
        self.name = name
        self.reported_vram_gb = vram
        self.load_calls = 0
        self.unload_calls = 0
        self.run_calls = 0
        self._run_error = run_error

    def load(self) -> None:
        self.load_calls += 1

    def unload(self) -> None:
        self.unload_calls += 1

    def run(self, prompt, mcps=(), input_files=()):
        self.run_calls += 1
        if self._run_error is not None:
            raise self._run_error
        return "ok"


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def test_probe_parses_csv_and_converts_mib_to_gib():
    stdout = (
        "0, NVIDIA H100 80GB HBM3, 81920, 76800\n"
        "1, NVIDIA H100 80GB HBM3, 81920, 40960\n"
    )
    devices = probe_devices(runner=lambda cmd: stdout)
    assert len(devices) == 2
    assert devices[0] == GpuDevice(0, "NVIDIA H100 80GB HBM3", 80.0, 75.0)
    assert devices[1].free_vram_gb == 40.0


def test_probe_skips_malformed_lines_and_blanks():
    stdout = "0, H100, 81920, 76800\n\ngarbage line\n1, H100, not_a_number, 1024\n"
    devices = probe_devices(runner=lambda cmd: stdout)
    assert [d.device_id for d in devices] == [0]


def test_probe_filters_to_node_range():
    stdout = "0, A, 1024, 1024\n3, B, 1024, 1024\n7, C, 1024, 1024\n"
    devices = probe_devices(node_range=(1, 6), runner=lambda cmd: stdout)
    assert [d.device_id for d in devices] == [3]


def test_probe_missing_nvidia_smi_returns_empty():
    def runner(cmd):
        raise FileNotFoundError("nvidia-smi")

    assert probe_devices(runner=runner) == []


def test_probe_command_failure_returns_empty():
    def runner(cmd):
        raise RuntimeError("NVML error")

    assert probe_devices(runner=runner) == []


# ---------------------------------------------------------------------------
# allocator
# ---------------------------------------------------------------------------


def test_sequential_assigns_all_roles_to_usable_group():
    config = GpuConfig(mode="auto", execution_mode="sequential", vram_headroom_gb=8.0)
    devices = [device(0, 75.0), device(1, 75.0), device(2, 4.0)]  # node 2 unusable
    result = allocate({"generator": 64.0, "critic": 40.0}, config, devices)
    assert result == {"generator": [0, 1], "critic": [0, 1]}


def test_sequential_insufficient_vram_message_has_numbers():
    config = GpuConfig(mode="auto", execution_mode="sequential", vram_headroom_gb=8.0)
    devices = [device(0, 30.0), device(1, 20.0)]
    with pytest.raises(InsufficientVramError) as exc_info:
        allocate({"generator": 64.0}, config, devices)
    message = str(exc_info.value)
    assert "64.0" in message and "50.0" in message and "generator" in message


def test_parallel_greedy_largest_first_best_fit():
    config = GpuConfig(mode="auto", execution_mode="parallel", vram_headroom_gb=8.0)
    devices = [device(0, 75.0), device(1, 75.0), device(2, 24.0)]
    result = allocate(
        {"generator": 64.0, "critic": 40.0, "analyzer": 20.0}, config, devices
    )
    # generator (largest) and critic each take one 75GB node; analyzer gets
    # the small node - the best (smallest) fit.
    assert sorted(result["generator"] + result["critic"]) == [0, 1]
    assert result["analyzer"] == [2]


def test_parallel_tie_break_is_deterministic_by_device_id():
    """Equal-VRAM nodes must be chosen by lowest device_id regardless of the
    order the devices are passed in (parity with the sequential path)."""
    config = GpuConfig(mode="auto", execution_mode="parallel", vram_headroom_gb=8.0)
    forward = allocate({"generator": 40.0}, config, [device(0, 75.0), device(5, 75.0)])
    reverse = allocate({"generator": 40.0}, config, [device(5, 75.0), device(0, 75.0)])
    assert forward["generator"] == [0]
    assert reverse["generator"] == [0]


def test_parallel_spans_multiple_nodes_when_no_single_fits():
    config = GpuConfig(mode="auto", execution_mode="parallel", vram_headroom_gb=8.0)
    devices = [device(0, 75.0), device(1, 75.0), device(2, 24.0)]
    result = allocate({"generator": 120.0, "analyzer": 20.0}, config, devices)
    assert result["generator"] == [0, 1]
    assert result["analyzer"] == [2]


def test_parallel_failure_suggests_sequential():
    config = GpuConfig(mode="auto", execution_mode="parallel", vram_headroom_gb=8.0)
    devices = [device(0, 75.0)]
    with pytest.raises(InsufficientVramError) as exc_info:
        allocate({"generator": 64.0, "critic": 40.0}, config, devices)
    assert "sequential" in str(exc_info.value)


def test_manual_mode_returns_configured_mapping():
    config = GpuConfig(
        mode="manual",
        manual_assignments=ModelNodeAssignment(
            generator_nodes=[0, 1], critic_nodes=[0], analyzer_nodes=[1]
        ),
    )
    devices = [device(0, 75.0), device(1, 75.0)]
    result = allocate({"generator": 64.0, "critic": 40.0}, config, devices)
    assert result == {"generator": [0, 1], "critic": [0]}


def test_manual_mode_unknown_node_raises():
    config = GpuConfig(
        mode="manual",
        manual_assignments=ModelNodeAssignment(generator_nodes=[0, 5]),
    )
    devices = [device(0, 75.0)]
    with pytest.raises(GpuNodeUnavailableError) as exc_info:
        allocate({"generator": 64.0}, config, devices)
    assert "5" in str(exc_info.value)


def test_manual_mode_unknown_role_raises_config_error():
    config = GpuConfig(
        mode="manual", manual_assignments=ModelNodeAssignment(generator_nodes=[0])
    )
    with pytest.raises(ConfigError):
        allocate({"mystery_role": 10.0}, config, [device(0, 75.0)])


def test_single_mode_assigns_all_roles_to_single_node():
    config = GpuConfig(mode="single", single_node_id=3)
    result = allocate({"generator": 40.0, "critic": 30.0}, config, [device(3, 75.0)])
    assert result == {"generator": [3], "critic": [3]}


def test_remote_roles_get_empty_node_list():
    config = GpuConfig(mode="auto", execution_mode="sequential")
    devices = [device(0, 75.0)]
    result = allocate({"generator": 40.0, "judge": None, "merger": 0}, config, devices)
    assert result["judge"] == []
    assert result["merger"] == []
    assert result["generator"] == [0]


def test_all_remote_roles_need_no_devices():
    config = GpuConfig(mode="auto", execution_mode="sequential")
    result = allocate({"judge": None, "merger": None}, config, [])
    assert result == {"judge": [], "merger": []}


# ---------------------------------------------------------------------------
# manager
# ---------------------------------------------------------------------------


def test_manager_probes_when_devices_none(monkeypatch):
    probed = [device(0, 75.0)]
    calls = []

    def fake_probe(node_range):
        calls.append(node_range)
        return probed

    monkeypatch.setattr("anvesha.core.gpu.manager.probe_devices", fake_probe)
    manager = GpuManager(GpuConfig(available_node_range=(0, 3)))
    assert manager.devices == probed
    assert calls == [(0, 3)]


def test_manager_allocate_stores_assignments():
    manager = GpuManager(GpuConfig(mode="single", single_node_id=0), devices=[device(0, 75.0)])
    result = manager.allocate({"generator": 40.0})
    assert result == {"generator": [0]}
    assert manager.assignments == {"generator": [0]}


def test_health_check_missing_node():
    manager = GpuManager(GpuConfig(), devices=[device(0, 75.0)])
    manager.assignments = {"generator": [0, 4]}
    with pytest.raises(GpuNodeUnavailableError) as exc_info:
        manager.health_check({"generator": FakeAdapter("gen", 40.0)})
    assert "4" in str(exc_info.value) and "generator" in str(exc_info.value)


def test_health_check_insufficient_free_vram():
    manager = GpuManager(GpuConfig(), devices=[device(0, 30.0)])
    manager.assignments = {"generator": [0]}
    with pytest.raises(InsufficientVramError):
        manager.health_check({"generator": FakeAdapter("gen", 64.0)})


def test_health_check_test_inference_failure_wrapped():
    manager = GpuManager(GpuConfig(), devices=[device(0, 75.0)])
    manager.assignments = {"generator": [0]}
    adapter = FakeAdapter("gen", 40.0, run_error=RuntimeError("CUDA error"))
    with pytest.raises(GpuNodeUnavailableError) as exc_info:
        manager.health_check({"generator": adapter}, run_test_inference=True)
    assert "CUDA error" in str(exc_info.value)


def test_health_check_happy_path_runs_inference_only_for_assigned():
    manager = GpuManager(GpuConfig(), devices=[device(0, 75.0)])
    manager.assignments = {"generator": [0], "judge": []}
    gen = FakeAdapter("gen", 40.0)
    judge = FakeAdapter("judge", None)
    manager.health_check({"generator": gen, "judge": judge}, run_test_inference=True)
    assert gen.run_calls == 1
    assert judge.run_calls == 0  # remote role: no nodes, no test inference


# ---------------------------------------------------------------------------
# scheduler
# ---------------------------------------------------------------------------


def make_scheduler() -> GpuScheduler:
    manager = GpuManager(GpuConfig(), devices=[device(0, 75.0), device(1, 75.0)])
    return GpuScheduler(manager)


def test_scheduler_warm_cache_hit_skips_reload():
    scheduler = make_scheduler()
    adapter = FakeAdapter("nemotron", 64.0)
    scheduler.ensure_loaded(adapter, [0, 1])
    scheduler.ensure_loaded(adapter, [0, 1])
    scheduler.ensure_loaded(adapter, [0])  # subset still warm
    assert adapter.load_calls == 1
    assert adapter.unload_calls == 0


def test_scheduler_swap_unloads_previous_exactly_once():
    scheduler = make_scheduler()
    nemotron = FakeAdapter("nemotron", 64.0)
    gpt_oss = FakeAdapter("gpt-oss", 40.0)
    scheduler.ensure_loaded(nemotron, [0, 1])
    scheduler.ensure_loaded(gpt_oss, [0, 1])  # warm on both nodes -> one unload
    assert nemotron.unload_calls == 1
    assert gpt_oss.load_calls == 1
    # nemotron went fully cold, so requesting it again reloads it.
    scheduler.ensure_loaded(nemotron, [0, 1])
    assert nemotron.load_calls == 2
    assert gpt_oss.unload_calls == 1


def test_scheduler_partial_overlap_evicts_whole_model():
    scheduler = make_scheduler()
    nemotron = FakeAdapter("nemotron", 64.0)
    gpt_oss = FakeAdapter("gpt-oss", 40.0)
    scheduler.ensure_loaded(nemotron, [0, 1])
    scheduler.ensure_loaded(gpt_oss, [0])  # overlaps only node 0
    assert nemotron.unload_calls == 1
    # node 1 is cold now too: loading nemotron there requires a fresh load.
    scheduler.ensure_loaded(nemotron, [1])
    assert nemotron.load_calls == 2
    assert gpt_oss.unload_calls == 0  # node 1 did not hold gpt-oss


def test_scheduler_release_all_unloads_each_adapter_once():
    scheduler = make_scheduler()
    nemotron = FakeAdapter("nemotron", 64.0)
    scheduler.ensure_loaded(nemotron, [0, 1])
    scheduler.release_all()
    assert nemotron.unload_calls == 1
    # registry cleared: next request loads again.
    scheduler.ensure_loaded(nemotron, [0])
    assert nemotron.load_calls == 2


def test_scheduler_release_all_when_empty_is_noop():
    scheduler = make_scheduler()
    scheduler.release_all()  # must not raise

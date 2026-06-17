"""Tests for the checkpoint and resume system (S14). Fully offline."""

from __future__ import annotations

import hashlib
import logging

import pytest

from anvesha.core.checkpoint import (
    CheckpointManager,
    CheckpointState,
    FailureRecord,
    ResumePlanner,
    StateSerializer,
    classify_failure,
    hash_inputs,
    make_failure_record,
    with_retries,
)
from anvesha.core.config import CheckpointConfig
from anvesha.core.exceptions import (
    AdapterParseError,
    AdapterTimeoutError,
    CheckpointError,
    ConfigError,
    GpuOomError,
)

RUN = "run-abc"


# ---------------------------------------------------------------------------
# StateSerializer (parameterized over both storage backends)
# ---------------------------------------------------------------------------


@pytest.fixture(params=["filesystem", "sqlite"])
def serializer(request, tmp_path):
    return StateSerializer(request.param, tmp_path / "ckpt")


def test_save_load_roundtrip(serializer):
    serializer.save(RUN, 1, {"a": 1})
    serializer.save(RUN, 2, {"a": 2})
    assert serializer.load(RUN, 1) == {"a": 1}
    assert serializer.load(RUN) == {"a": 2}  # latest when version is None


def test_list_versions_sorted(serializer):
    for v in (3, 1, 2):
        serializer.save(RUN, v, {"v": v})
    assert serializer.list_versions(RUN) == [1, 2, 3]
    assert serializer.list_versions("other-run") == []


def test_load_missing_raises(serializer):
    with pytest.raises(CheckpointError):
        serializer.load(RUN)
    serializer.save(RUN, 1, {})
    with pytest.raises(CheckpointError):
        serializer.load(RUN, 7)


def test_prune_keeps_last_five(serializer):
    for v in range(1, 9):
        serializer.save(RUN, v, {"v": v})
    serializer.prune(RUN, keep=5)
    assert serializer.list_versions(RUN) == [4, 5, 6, 7, 8]
    assert serializer.load(RUN) == {"v": 8}


def test_runs_are_isolated(serializer):
    serializer.save("run-a", 1, {"run": "a"})
    serializer.save("run-b", 1, {"run": "b"})
    serializer.prune("run-a", keep=5)
    assert serializer.load("run-b", 1) == {"run": "b"}
    assert serializer.list_versions("run-a") == [1]


def test_filesystem_exact_filename(tmp_path):
    ser = StateSerializer("filesystem", tmp_path)
    ser.save(RUN, 3, {"x": 1})
    assert (tmp_path / f"{RUN}_00003.checkpoint.json").is_file()


def test_sqlite_single_db_file(tmp_path):
    ser = StateSerializer("sqlite", tmp_path)
    ser.save(RUN, 1, {})
    ser.save(RUN, 2, {})
    files = [p.name for p in tmp_path.iterdir()]
    assert files == ["anvesha_checkpoints.sqlite"]


def test_unknown_storage_rejected(tmp_path):
    with pytest.raises(ConfigError):
        StateSerializer("redis", tmp_path)


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


def make_manager(tmp_path, **overrides):
    config = CheckpointConfig(**overrides)
    return CheckpointManager(config, tmp_path / "ckpt")


def test_manager_version_increment_and_load_latest(tmp_path):
    mgr = make_manager(tmp_path)
    assert mgr.save(RUN, {"step": 1}, {"last": "a"}, {"lit": "h1"}) == 1
    record = FailureRecord(
        failed_at="t", agent_name="critic", failure_type="timeout", error_message="x"
    )
    assert mgr.save(RUN, {"step": 2}, {"last": "b"}, {"lit": "h1"}, [record]) == 2

    state = mgr.load_latest(RUN)
    assert isinstance(state, CheckpointState)
    assert state.pipeline_run_id == RUN
    assert state.checkpoint_version == 2
    assert state.pipeline_state == {"step": 2}
    assert state.resume_hint == {"last": "b"}
    assert state.input_hashes == {"lit": "h1"}
    assert state.failure_log == [record]
    assert state.granularity == "iteration"
    assert state.last_saved_at  # ISO timestamp present


def test_manager_preserves_created_at(tmp_path):
    mgr = make_manager(tmp_path)
    mgr.save(RUN, {}, {}, {})
    first = mgr.load_latest(RUN)
    mgr.save(RUN, {}, {}, {})
    second = mgr.load_latest(RUN)
    assert second.created_at == first.created_at


def test_manager_retention_five(tmp_path):
    mgr = make_manager(tmp_path)
    for i in range(7):
        mgr.save(RUN, {"i": i}, {}, {})
    assert mgr._serializer.list_versions(RUN) == [3, 4, 5, 6, 7]


def test_manager_disabled_returns_none(tmp_path):
    mgr = make_manager(tmp_path, enabled=False)
    assert mgr.save(RUN, {}, {}, {}) is None
    assert mgr.load_latest(RUN) is None


def test_manager_load_latest_none_when_empty(tmp_path):
    assert make_manager(tmp_path).load_latest(RUN) is None


def test_manager_write_failure_swallowed(tmp_path, monkeypatch, caplog):
    mgr = make_manager(tmp_path)

    def poisoned_save(run_id, version, payload):
        raise OSError("disk full")

    monkeypatch.setattr(mgr._serializer, "save", poisoned_save)
    with caplog.at_level(logging.WARNING, logger="anvesha.core.checkpoint.manager"):
        assert mgr.save(RUN, {}, {}, {}) is None
    assert any("Checkpoint write failed" in r.message for r in caplog.records)


def test_manager_prune_failure_does_not_mask_successful_write(tmp_path, monkeypatch, caplog):
    """A prune error after a durable write must still return the version, not
    None - the checkpoint is on disk and loadable (S14.5)."""
    mgr = make_manager(tmp_path)
    mgr.save(RUN, {"i": 0}, {}, {})  # version 1, written normally

    def poisoned_prune(run_id, keep):
        raise OSError("cannot unlink old checkpoint")

    monkeypatch.setattr(mgr._serializer, "prune", poisoned_prune)
    with caplog.at_level(logging.WARNING, logger="anvesha.core.checkpoint.manager"):
        version = mgr.save(RUN, {"i": 1}, {}, {})
    assert version == 2  # write succeeded; not masked as failure
    assert mgr.load_latest(RUN).checkpoint_version == 2
    assert any("prune failed" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# hash_inputs + ResumePlanner
# ---------------------------------------------------------------------------


def test_hash_inputs_deterministic():
    contents = {"literature_md": "abc", "approach_md": "xyz"}
    hashes = hash_inputs(contents)
    assert hashes == hash_inputs(dict(contents))
    assert hashes["literature_md"] == hashlib.sha256(b"abc").hexdigest()
    assert set(hashes) == {"literature_md", "approach_md"}


def test_planner_no_checkpoint_fresh(tmp_path):
    planner = ResumePlanner(make_manager(tmp_path))
    plan = planner.plan(RUN, {"lit": "h1"})
    assert plan.action == "fresh"
    assert plan.checkpoint is None
    assert plan.inputs_changed is False


def test_planner_hash_mismatch_fresh_with_warning_reason(tmp_path):
    mgr = make_manager(tmp_path)
    mgr.save(RUN, {"s": 1}, {}, {"lit": "h1"})
    plan = ResumePlanner(mgr).plan(RUN, {"lit": "DIFFERENT"})
    assert plan.action == "fresh"
    assert plan.inputs_changed is True
    assert plan.checkpoint is not None  # caller may offer resume-anyway
    assert "warn" in plan.reason.lower()


def test_planner_hash_match_resume(tmp_path):
    mgr = make_manager(tmp_path)
    mgr.save(RUN, {"s": 1}, {"last": "judge"}, {"lit": "h1"})
    mgr.save(RUN, {"s": 2}, {"last": "merge"}, {"lit": "h1"})
    plan = ResumePlanner(mgr).plan(RUN, {"lit": "h1"})
    assert plan.action == "resume"
    assert plan.inputs_changed is False
    assert plan.checkpoint.checkpoint_version == 2
    assert plan.checkpoint.resume_hint == {"last": "merge"}


# ---------------------------------------------------------------------------
# Recovery primitives
# ---------------------------------------------------------------------------


def test_classify_failure_mapping():
    assert classify_failure(AdapterTimeoutError("t")) == "timeout"
    assert classify_failure(AdapterParseError("p")) == "parse_error"
    assert classify_failure(GpuOomError("o")) == "gpu_oom"
    assert classify_failure(ValueError("v")) == "unknown"


def test_make_failure_record():
    rec = make_failure_record(
        "critic", AdapterTimeoutError("slow"), recovery_action="retried", gap_id="g1"
    )
    assert rec.agent_name == "critic"
    assert rec.failure_type == "timeout"
    assert rec.error_message == "slow"
    assert rec.recovery_action == "retried"
    assert rec.gap_id == "g1"
    assert rec.failed_at


def test_with_retries_succeeds_after_failures():
    delays: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise AdapterTimeoutError("busy")
        return "ok"

    assert with_retries(flaky, sleep=delays.append) == "ok"
    assert calls["n"] == 3
    assert delays == [5, 10]  # backoff_seconds * 2**attempt


def test_with_retries_exhaustion_reraises():
    delays: list[float] = []
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise AdapterTimeoutError("busy")

    with pytest.raises(AdapterTimeoutError):
        with_retries(
            always_fails, max_retries=3, backoff_seconds=2, sleep=delays.append
        )
    assert calls["n"] == 4  # initial attempt + 3 retries
    assert delays == [2, 4, 8]


def test_with_retries_non_retryable_raises_immediately():
    delays: list[float] = []
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        with_retries(boom, sleep=delays.append)
    assert calls["n"] == 1
    assert delays == []


def test_with_retries_custom_retry_on():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise GpuOomError("oom")
        return 42

    assert with_retries(flaky, retry_on=(GpuOomError,), sleep=lambda _d: None) == 42
    assert calls["n"] == 2

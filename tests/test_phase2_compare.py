"""Checks for temporal isolation, round audit and matched-time comparisons."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.context import derive_context, verify_event_only_coverage
from cs2_pipeline.schema import PipelineError
from scripts.audit_phase2_data import audit_identities, audit_sample_rounds
from scripts.context_audit_batch import update
from scripts.explore_phase2 import paired_times

ML_MISSING = importlib.util.find_spec("sklearn") is None


def test_progress_retries_windows_sharing_violation(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "progress.json"
    target.write_text("{}", encoding="utf-8")
    original = Path.replace
    attempts = []

    def flaky_replace(path, destination):
        attempts.append(path)
        if len(attempts) == 1:
            raise PermissionError("dashboard is reading")
        return original(path, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("scripts.context_audit_batch.time.sleep", lambda seconds: None)
    update(target, {"completed": 93})
    assert len(attempts) == 2
    assert json.loads(target.read_text(encoding="utf-8"))["completed"] == 93


def sample_round() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    samples = pl.DataFrame(
        {
            "sample_id": ["x"],
            "round_id": [0],
            "round_num": [7],
            "tick": [100],
            "elapsed_since_freeze_end_seconds": [0.0],
        }
    )
    labels = pl.DataFrame(
        {"sample_id": ["x"], "y_ct_win": [1]}, schema_overrides={"y_ct_win": pl.UInt8}
    )
    rounds = pl.DataFrame(
        {
            "round_id": [0],
            "round_num": [7],
            "freeze_end_tick": [100],
            "end_tick": [420],
            "winner_side": ["CT"],
            "is_valid_round": [True],
        }
    )
    return samples, labels, rounds


def test_audit_accepts_freeze_end_and_rejects_end_and_wrong_label() -> None:
    samples, labels, rounds = sample_round()
    assert not any(audit_sample_rounds(samples, labels, rounds, 64).values())
    at_end = samples.with_columns(
        pl.lit(420).alias("tick"), pl.lit(5.0).alias("elapsed_since_freeze_end_seconds")
    )
    with pytest.raises(PipelineError, match="outside_combat_window"):
        audit_sample_rounds(at_end, labels, rounds, 64)
    with pytest.raises(PipelineError, match="wrong_label"):
        audit_sample_rounds(
            samples, labels.with_columns(pl.lit(0, dtype=pl.UInt8).alias("y_ct_win")), rounds, 64
        )


def test_identity_audit_detects_resurrection_and_preserves_64_bit_ids() -> None:
    samples = pl.DataFrame(
        {
            "sample_id": ["a", "b"],
            "match_id": ["m", "m"],
            "demo_sha256": ["d", "d"],
            "round_id": [1, 1],
        }
    )
    rows = []
    for sample, tick in (("a", 100), ("b", 420)):
        for i in range(10):
            rows.append(
                {
                    "sample_id": sample,
                    "tick": tick,
                    "steamid": 76561198000000000 + i,
                    "side": "CT" if i < 5 else "T",
                    "team": "A" if i < 5 else "B",
                    "is_alive": not (sample == "a" and i == 0),
                    "death_observed": False,
                }
            )
    nodes = pl.DataFrame(rows, schema_overrides={"steamid": pl.UInt64})
    with pytest.raises(PipelineError, match="alive_after_prior_dead_state"):
        audit_identities(nodes, samples)
    stable = nodes.with_columns(pl.lit(True).alias("is_alive"))
    assert audit_identities(stable, samples)["nodes"] == 20
    with pytest.raises(PipelineError, match="UInt64"):
        audit_identities(stable.with_columns(pl.col("steamid").cast(pl.Float64)), samples)


def test_event_only_uses_no_future_plant_or_invented_clock() -> None:
    moments = pl.DataFrame({"tick": [10, 20, 30], "round_id": [0, 0, 0]})
    rules = pl.DataFrame(schema={"tick": pl.Int64})
    events = pl.DataFrame(
        {
            "tick": [20, 25],
            "round_id": [0, 0],
            "event": ["finish_plant", "defuse"],
            "bombsite": ["A", None],
        }
    )
    result = derive_context(moments, rules, events)
    assert result["bomb_planted"].to_list() == [0.0, 1.0, 0.0]
    assert result["bomb_site_a"].to_list() == [0.0, 1.0, 0.0]
    assert result["round_elapsed_game_seconds"].null_count() == 3
    assert result["is_paused"].null_count() == 3
    rounds = pl.DataFrame(
        {"round_id": [0], "is_valid_round": [True], "reason_name": ["bomb_defused"]}
    )
    verify_event_only_coverage(rounds, events)
    with pytest.raises(PipelineError, match="缺少已记录的下包事件"):
        verify_event_only_coverage(rounds, events.filter(pl.col("event") != "finish_plant"))


@pytest.mark.skipif(ML_MISSING, reason="需要可选 ml 依赖")
def test_split_requires_distinct_events_matches_and_ascending_dates() -> None:
    from cs2_pipeline.phase2_compare import check_split

    plan = {
        "candidate_fit_events": ["A"],
        "candidate_calibration_events": ["B"],
        "candidate_validation_events": ["C"],
        "full_baseline_fit_events": ["A", "B", "C"],
    }
    historic = pl.DataFrame(
        {
            "event": ["A", "B", "C"],
            "match_date": ["2026-01-01", "2026-02-01", "2026-03-01"],
            "match_id": ["a", "b", "c"],
        }
    )
    future = pl.DataFrame({"event": ["D"], "match_date": ["2026-04-01"], "match_id": ["d"]})
    check_split(historic, future, plan)
    with pytest.raises(PipelineError, match="赛事必须相互隔离"):
        check_split(historic, future.with_columns(pl.lit("A").alias("event")), plan)
    with pytest.raises(PipelineError, match="日期必须严格"):
        check_split(historic, future.with_columns(pl.lit("2026-02-01").alias("match_date")), plan)
    with pytest.raises(PipelineError, match="同一比赛"):
        check_split(historic, future.with_columns(pl.lit("a").alias("match_id")), plan)
    with pytest.raises(PipelineError, match="Bucharest"):
        check_split(
            historic, future.with_columns(pl.lit("pgl-masters-bucharest-2026").alias("event")), plan
        )


@pytest.mark.skipif(ML_MISSING, reason="需要可选 ml 依赖")
def test_simple_model_and_calibration_only_use_whitelisted_inputs() -> None:
    import numpy as np

    from cs2_pipeline.phase2_compare import (
        apply_sigmoid,
        fit_logistic,
        fit_sigmoid,
        predict_logistic,
    )

    frame = pl.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "demo_sha256": ["x"] * 4,
            "round_id": [1, 2, 3, 4],
            "y_ct_win": [0, 0, 1, 1],
            "ct_minus_t_alive_count": [-2.0, -1.0, 1.0, 2.0],
            "elapsed_since_freeze_end_seconds": [5.0, 10.0, 15.0, 20.0],
        }
    )
    columns = ["ct_minus_t_alive_count", "elapsed_since_freeze_end_seconds"]
    model = fit_logistic(frame, columns)
    probabilities = predict_logistic(model, frame)
    changed_labels = frame.with_columns((1 - pl.col("y_ct_win")).alias("y_ct_win"))
    assert np.array_equal(probabilities, predict_logistic(model, changed_labels))
    calibration = fit_sigmoid(np.array([0.1, 0.2, 0.8, 0.9]), frame)
    assert np.all(np.diff(apply_sigmoid(np.array([0.1, 0.2, 0.8, 0.9]), calibration)) > 0)
    with pytest.raises(PipelineError, match="白名单"):
        fit_logistic(frame, ["y_ct_win"])


@pytest.mark.skipif(ML_MISSING, reason="需要可选 ml 依赖")
def test_time_comparison_uses_only_same_round_pairs() -> None:
    frame = pl.DataFrame(
        {
            "demo_sha256": ["a", "a", "a", "b", "b"],
            "round_id": [1, 1, 2, 1, 1],
            "match_id": ["m1", "m1", "m1", "m2", "m2"],
            "y_ct_win": [0, 0, 0, 1, 1],
            "elapsed_since_freeze_end_seconds": [0.0, 60.0, 0.0, 0.0, 60.0],
            "frozen_v1": [0.4, 0.2, 0.3, 0.6, 0.8],
        }
    )
    result = paired_times(frame)
    assert result["early"]["rounds"] == result["late"]["rounds"] == 2
    assert result["early"]["sample_count"] == result["late"]["sample_count"] == 2


@pytest.mark.skipif(ML_MISSING, reason="需要可选 ml 依赖")
def test_gpu_proof_requires_our_pid_and_correct_physical_device() -> None:
    from cs2_pipeline.phase2_gpu import observation_for_pid

    rows = [["GPU-other", "123", "128"], ["GPU-expected", "999", "256"]]
    assert observation_for_pid(rows, 123, "GPU-expected") is None
    rows.append(["GPU-expected", "123", "[N/A]"])
    assert observation_for_pid(rows, 123, "GPU-expected") == {
        "gpu_uuid": "GPU-expected",
        "pid": 123,
        "memory_mib": "[N/A]",
    }

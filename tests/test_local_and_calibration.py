"""Synthetic regression tests plus a separate real-local-corpus integrity check."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.calibrate import (
    CALIBRATOR,
    calibrate_experiment,
    cluster_bootstrap,
    experiment_frame,
    fit_calibrator,
)
from cs2_pipeline.dataset import build_dataset, read_dataset, validate_dataset_tables, write_json
from cs2_pipeline.local import check_round_sequence
from cs2_pipeline.parse import sha256_file, write_outputs
from cs2_pipeline.predict import predict_logits, predict_probabilities
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.split import frozen_split
from cs2_pipeline.train import train_baseline


@pytest.mark.parametrize("parts,expected", [([[1, 2], [3, 4]], 4), ([[1]], 1)])
def test_recording_boundaries_preserve_original_rounds(parts: list[list[int]], expected: int):
    check_round_sequence(parts, expected)


@pytest.mark.parametrize("parts", [[], [[1, 2], [2, 3]], [[1], [3, 4]], [[3, 4], [1, 2]], [[]]])
def test_recording_gaps_duplicates_and_reordering_rejected(parts: list[list[int]]):
    with pytest.raises(PipelineError, match="缺口、重复或顺序"):
        check_round_sequence(parts, 4)


@pytest.fixture
def experiment(phase2_manifest: Path, dataset: tuple, tmp_path: Path) -> tuple:
    reference = tmp_path / "reference"
    build_dataset(phase2_manifest, reference)
    base_dir = tmp_path / "base"
    train_baseline(reference, base_dir)
    base_path = base_dir / "model.json"
    original, rounds, players, snapshots = dataset
    config = json.loads(phase2_manifest.read_text(encoding="utf-8"))
    for index in (3, 4):
        path = tmp_path / f"source{index}"
        write_outputs(
            path,
            {
                **original,
                "tick_rate": 64.0,
                "demo_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
            },
            rounds,
            players,
            snapshots,
        )
        config["sources"].append(
            {
                "path": path.name,
                "match_id": f"synthetic:{index}",
                "match_date": f"2026-05-0{index + 1}",
                "event": "synthetic-test",
            }
        )
    plan = {
        "assignments": dict(
            zip(
                [s["match_id"] for s in config["sources"]],
                ["train", "validation", "calibration", "test", "test"],
                strict=True,
            )
        ),
        "test_events": ["synthetic-test"],
        "feature_version": 1,
        "base_model_sha256": sha256_file(base_path),
        "reference_dataset_sha256": sha256_file(reference / "dataset.json"),
        "calibrator": CALIBRATOR,
        "bootstrap": {"draws": 50, "seed": 42},
    }
    config["split_plan"] = plan
    write_json(phase2_manifest, config)
    built = tmp_path / "fourway"
    build_dataset(phase2_manifest, built)
    return built, reference, base_path, plan


def test_frozen_four_way_rejects_missing_ids_same_day_and_test_event_leaks():
    sources = [
        {"match_id": str(i), "match_date": f"2026-05-0{i + 1}", "event": str(i)} for i in range(4)
    ]
    plan = {
        "assignments": dict(
            zip(["0", "1", "2", "3"], ["train", "validation", "calibration", "test"], strict=True)
        ),
        "test_events": ["3"],
    }
    _, policy = frozen_split(sources, plan)
    assert len(policy["plan_sha256"]) == 64
    with pytest.raises(PipelineError, match="覆盖全部比赛"):
        frozen_split(sources[:-1], plan)
    bad = copy.deepcopy(sources)
    bad[2]["match_date"] = bad[1]["match_date"]
    with pytest.raises(PipelineError, match="严格先后"):
        frozen_split(bad, plan)
    bad = copy.deepcopy(sources)
    bad[1]["event"] = "3"
    with pytest.raises(PipelineError, match="测试赛事"):
        frozen_split(bad, plan)


def test_frozen_plan_is_enforced_on_reread(experiment: tuple):
    built, _, _, _ = experiment
    metadata, tables = read_dataset(built)
    assert set(tables["splits"]["split"]) == {"train", "validation", "calibration", "test"}
    tables["splits"] = tables["splits"].with_columns(
        pl.when(pl.col("split") == "calibration")
        .then(pl.lit("test"))
        .otherwise(pl.col("split"))
        .alias("split")
    )
    with pytest.raises(PipelineError, match="冻结计划"):
        validate_dataset_tables(tables, metadata)


def test_only_calibration_rows_affect_calibrator(experiment: tuple):
    built, _, base_path, _ = experiment
    _, tables = read_dataset(built)
    frame = experiment_frame(tables)
    base = json.loads(base_path.read_text())
    before = fit_calibrator(base, frame)
    changed = frame.with_columns(
        pl.when(pl.col("split") != "calibration")
        .then(1 - pl.col("y_ct_win"))
        .otherwise(pl.col("y_ct_win"))
        .cast(pl.UInt8)
        .alias("y_ct_win"),
        pl.when(pl.col("split") != "calibration")
        .then(1e9)
        .otherwise(pl.col("ct_alive_money_sum"))
        .alias("ct_alive_money_sum"),
    )
    after = fit_calibrator(base, changed)
    assert before == after
    assert before["parameters"] == base["parameters"]
    assert "calibration" not in base
    assert before["calibration"]["match_ids"] == ["synthetic:2"]
    logits = predict_logits(base, tables["features"])
    probabilities = predict_probabilities(before, tables["features"])
    assert probabilities["p_ct_win"].is_between(0, 1).all()
    ordered = logits.join(probabilities, on="sample_id").sort("logit")
    assert ordered["p_ct_win"].to_list() == sorted(ordered["p_ct_win"].to_list())
    corrupted = copy.deepcopy(before)
    corrupted["calibration"]["slope"] = -1.0
    with pytest.raises(PipelineError, match="校准参数损坏"):
        predict_probabilities(corrupted, tables["features"])
    with pytest.raises(PipelineError, match="CT 胜和 T 胜"):
        fit_calibrator(base, frame.with_columns(pl.lit(1).alias("y_ct_win")))


def test_calibration_experiment_roundtrip_and_provenance(experiment: tuple, tmp_path: Path):
    built, reference, base_path, _ = experiment
    out = tmp_path / "calibrated"
    report = calibrate_experiment(built, reference, base_path, out)
    assert report["counts"]["calibration"]["matches"] == 1
    assert report["counts"]["test"]["matches"] == 2
    assert report["calibrated_model_sha256"] == sha256_file(out / "model.json")
    assert report["test"]["calibrated"]["ece_10_bins"] >= 0
    assert pl.read_parquet(out / "predictions.parquet").height == 12
    assert report["paired_match_bootstrap"]["match_count"] == 2
    with pytest.raises(PipelineError, match="已存在"):
        calibrate_experiment(built, reference, base_path, out)
    altered = tmp_path / "altered.json"
    altered.write_text(base_path.read_text() + "\n")
    with pytest.raises(PipelineError, match="预先声明"):
        calibrate_experiment(built, reference, altered, tmp_path / "rejected")


def test_bootstrap_resamples_matches_and_preserves_pairing():
    frame = pl.DataFrame(
        {
            "match_id": ["a", "a", "b"],
            "demo_sha256": ["a", "a", "b"],
            "round_id": [1, 1, 1],
            "y_ct_win": [1, 1, 0],
        }
    )
    frame = frame.with_columns(pl.col("y_ct_win").cast(pl.UInt8))
    result = cluster_bootstrap(frame, [0.8, 0.8, 0.2], [0.8, 0.8, 0.2], draws=100)
    assert result["match_count"] == 2
    assert result["percentile_95_intervals"]["calibrated_minus_base"]["log_loss"] == [0.0, 0.0]
    # Each match contributes one equally weighted round despite different tick counts.
    interval = result["percentile_95_intervals"]["base"]["brier_score"]
    assert interval == pytest.approx([0.04, 0.04])
    assert result["percentile_95_intervals"]["base"]["log_loss"] == pytest.approx(
        [0.2231435513142097, 0.2231435513142097]
    )


@pytest.mark.integration
def test_real_cologne_import_and_dataset_integrity():
    path = Path("data/datasets/mirage_cologne_experiment_v1")
    if not path.exists():
        pytest.skip("需要用户提供的真实 Cologne Demo 解析结果")
    metadata, tables = read_dataset(path)
    test = (
        tables["samples"].join(tables["splits"], on="sample_id").filter(pl.col("split") == "test")
    )
    assert test["match_id"].n_unique() == 14
    assert test["demo_sha256"].n_unique() == 17
    assert test.select("demo_sha256", "round_id").unique().height == 333
    assert test["round_num"].max() == 47
    assert tables["nodes"].schema["steamid"] == pl.UInt64
    assert metadata["split_policy"]["match_counts"] == {
        "train": 7,
        "validation": 1,
        "calibration": 2,
        "test": 14,
    }

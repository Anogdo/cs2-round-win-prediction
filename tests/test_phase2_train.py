"""ML tests use explicitly synthetic table fixtures, not independent match evidence."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.compare import compare_runs
from cs2_pipeline.dataset import build_dataset, read_dataset, write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_file, predict_probabilities
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import (
    fit_baseline,
    probability_metrics,
    round_weights,
    train_baseline,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="安装 --extra ml 后运行基线测试"
)


@pytest.fixture
def ml_dataset(phase2_manifest: Path, tmp_path: Path) -> Path:
    out = tmp_path / "ml_dataset"
    build_dataset(phase2_manifest, out)
    return out


def test_each_round_has_equal_total_weight() -> None:
    frame = pl.DataFrame({"demo_sha256": ["a"] * 4, "round_id": [1, 1, 1, 2]})
    weights = round_weights(frame)
    assert sum(weights[:3]) == pytest.approx(1.0) and weights[3] == 1.0


def test_formal_baseline_produces_independent_metrics_and_portable_model(
    ml_dataset: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "model"
    report = train_baseline(ml_dataset, out)
    assert report["independent_evaluation_performed"] is True
    assert set(report["metrics"]) == {"validation", "test"}
    assert report["training_round_prior"] == pytest.approx(1 / 3)
    assert report["training"]["round_count"] == 3 and report["training"]["match_count"] == 1
    assert report["metrics"]["test"]["model"]["round_weight_sum"] == pytest.approx(3)
    assert report["metrics"]["test"]["model"]["calibration_bins"]
    model = json.loads((out / "model.json").read_text(encoding="utf-8"))
    _, tables = read_dataset(ml_dataset)
    predicted = predict_probabilities(model, tables["features"])
    assert predicted["p_ct_win"].is_between(0, 1).all()
    assert predicted["p_ct_win"].is_finite().all()
    file = tmp_path / "portable_predictions.parquet"
    assert predict_file(out / "model.json", ml_dataset / "features.parquet", file).equals(predicted)
    with pytest.raises(PipelineError, match="已存在"):
        predict_file(out / "model.json", ml_dataset / "features.parquet", file)
    with pytest.raises(PipelineError, match="已存在"):
        train_baseline(ml_dataset, out)


def test_heldout_extremes_do_not_change_training_preprocessing_or_coefficients(
    ml_dataset: Path,
    tmp_path: Path,
) -> None:
    train_baseline(ml_dataset, tmp_path / "before")
    metadata, tables = read_dataset(ml_dataset)
    heldout_ids = tables["splits"].filter(pl.col("split") != "train")["sample_id"].to_list()
    tables["features"] = tables["features"].with_columns(
        pl.when(pl.col("sample_id").is_in(heldout_ids))
        .then(1000000.0)
        .otherwise(pl.col("ct_alive_money_sum"))
        .alias("ct_alive_money_sum"),
    )
    tables["features"].write_parquet(ml_dataset / "features.parquet")
    metadata["file_sha256"]["features.parquet"] = sha256_file(ml_dataset / "features.parquet")
    write_json(ml_dataset / "dataset.json", metadata)
    train_baseline(ml_dataset, tmp_path / "after")
    before = json.loads((tmp_path / "before/model.json").read_text(encoding="utf-8"))
    after = json.loads((tmp_path / "after/model.json").read_text(encoding="utf-8"))
    assert before["parameters"] == after["parameters"]


def test_all_missing_training_feature_does_not_borrow_heldout_values(
    ml_dataset: Path,
    tmp_path: Path,
) -> None:
    metadata, tables = read_dataset(ml_dataset)
    train_ids = tables["splits"].filter(pl.col("split") == "train")["sample_id"].to_list()
    tables["features"] = tables["features"].with_columns(
        pl.when(pl.col("sample_id").is_in(train_ids))
        .then(None)
        .otherwise(999999.0)
        .cast(pl.Float64)
        .alias("ct_alive_money_sum"),
    )
    tables["features"].write_parquet(ml_dataset / "features.parquet")
    metadata["file_sha256"]["features.parquet"] = sha256_file(ml_dataset / "features.parquet")
    write_json(ml_dataset / "dataset.json", metadata)
    report = train_baseline(ml_dataset, tmp_path / "model")
    model = json.loads((tmp_path / "model/model.json").read_text(encoding="utf-8"))
    index = model["feature_columns"].index("ct_alive_money_sum")
    assert model["parameters"]["imputation_values"][index] == 0.0
    assert index in model["parameters"]["indicator_indices"]
    assert "ct_alive_money_sum" in report["training"]["all_missing_training_features"]


def test_one_match_requires_explicit_smoke_mode_and_has_no_metrics(
    phase2_manifest: Path,
    tmp_path: Path,
) -> None:
    config = json.loads(phase2_manifest.read_text(encoding="utf-8"))
    config["sources"] = config["sources"][:1]
    write_json(phase2_manifest, config)
    path = tmp_path / "one_match"
    build_dataset(phase2_manifest, path)
    with pytest.raises(PipelineError, match="--smoke-test"):
        train_baseline(path, tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    report = train_baseline(path, tmp_path / "smoke", smoke_test=True)
    assert report["metrics"] is None and not report["independent_evaluation_performed"]
    assert report["run_mode"] == "smoke_test"
    predictions = pl.read_parquet(tmp_path / "smoke/predictions.parquet")
    assert predictions["split"].unique().to_list() == ["train_smoke"]


def test_single_class_training_is_actionable(ml_dataset: Path) -> None:
    _, tables = read_dataset(ml_dataset)
    frame = (
        tables["features"]
        .join(
            tables["samples"].select("sample_id", "match_id", "demo_sha256", "round_id"),
            on="sample_id",
        )
        .with_columns(pl.lit(1, dtype=pl.UInt8).alias("y_ct_win"))
    )
    with pytest.raises(PipelineError, match="同时有 CT 胜和 T 胜"):
        fit_baseline(frame)


def test_single_class_holdout_has_no_fake_auc() -> None:
    metrics = probability_metrics([1, 1], [0.4, 0.6], [0.5, 0.5])
    assert metrics["roc_auc"] is None and metrics["log_loss"] > 0


def test_corrupt_model_parameters_fail(ml_dataset: Path, tmp_path: Path) -> None:
    train_baseline(ml_dataset, tmp_path / "model")
    model = json.loads((tmp_path / "model/model.json").read_text(encoding="utf-8"))
    model["parameters"]["coefficients"] = []
    _, tables = read_dataset(ml_dataset)
    with pytest.raises(PipelineError, match="参数损坏"):
        predict_probabilities(model, tables["features"])


@pytest.mark.integration
def test_real_phase2_training_is_only_smoke_evidence(tmp_path: Path) -> None:
    source = Path("data/processed/match")
    if not (source / "metadata.json").is_file():
        pytest.skip("没有真实 Phase 1 数据，不能声称实际训练成功")
    config = json.loads(Path("configs/phase2.json").read_text(encoding="utf-8"))
    config["sources"][0]["path"] = str(source.resolve())
    manifest = tmp_path / "real_manifest.json"
    write_json(manifest, config)
    dataset = tmp_path / "real_dataset"
    build_dataset(manifest, dataset)
    report = train_baseline(dataset, tmp_path / "real_model", smoke_test=True)
    assert report["metrics"] is None and report["training"]["round_count"] == 18
    predicted = predict_file(
        tmp_path / "real_model/model.json",
        dataset / "features.parquet",
        tmp_path / "real_prediction.parquet",
    )
    assert predicted.height == 295 and predicted["p_ct_win"].is_between(0, 1).all()


def test_comparison_recalculates_scores_and_baseline_wins_validation_ties(
    ml_dataset: Path,
    tmp_path: Path,
) -> None:
    model = tmp_path / "compare_model"
    train_baseline(ml_dataset, model)
    comparison = compare_runs(ml_dataset, model, ml_dataset, model)
    assert comparison["selected_by_validation"] == "baseline"
    assert comparison["baseline"]["metrics"] == comparison["candidate"]["metrics"]


def test_comparison_refuses_changed_targets(ml_dataset: Path, tmp_path: Path) -> None:
    other = tmp_path / "other_dataset"
    shutil.copytree(ml_dataset, other)
    metadata, tables = read_dataset(other)
    tables["labels"].with_columns(
        (1 - pl.col("y_ct_win")).cast(pl.UInt8).alias("y_ct_win")
    ).write_parquet(other / "labels.parquet")
    metadata["file_sha256"]["labels.parquet"] = sha256_file(other / "labels.parquet")
    write_json(other / "dataset.json", metadata)
    with pytest.raises(PipelineError, match="labels.*不一致"):
        compare_runs(ml_dataset, tmp_path / "unused1", other, tmp_path / "unused2")

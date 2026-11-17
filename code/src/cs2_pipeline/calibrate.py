"""Calibrate a frozen baseline on older matches, then evaluate a new event once."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_logits, predict_probabilities
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import probability_metrics, round_weights

CALIBRATOR = {
    "method": "sigmoid_logit",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "random_state": 42,
}


def experiment_frame(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return (
        tables["features"]
        .join(tables["labels"], on="sample_id", validate="1:1")
        .join(
            tables["samples"].select("sample_id", "demo_sha256", "round_id", "match_id"),
            on="sample_id",
            validate="1:1",
        )
        .join(tables["splits"], on="sample_id", validate="1:1")
        .sort("sample_id")
    )


def fit_calibrator(base: dict[str, Any], frame: pl.DataFrame) -> dict[str, Any]:
    """Only calibration rows may influence the two additional model parameters."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    if "calibration" in base:
        raise PipelineError("输入必须是未校准的冻结模型，不能叠加校准。")
    calibration = frame.filter(pl.col("split") == "calibration").sort("sample_id")
    if calibration.is_empty() or calibration["y_ct_win"].n_unique() != 2:
        raise PipelineError("校准集合必须同时包含 CT 胜和 T 胜；请增加独立的旧比赛。")
    logits = predict_logits(base, calibration.select("sample_id", *base["feature_columns"]))
    estimator = LogisticRegression(**{k: v for k, v in CALIBRATOR.items() if k != "method"})
    estimator.fit(
        np.array(logits["logit"]).reshape(-1, 1),
        calibration["y_ct_win"].to_numpy(),
        sample_weight=round_weights(calibration),
    )
    slope, intercept = float(estimator.coef_[0, 0]), float(estimator.intercept_[0])
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise PipelineError("校准斜率非正或参数无效；保留原模型并增加校准比赛，不反转胜率排序。")
    if estimator.n_iter_[0] >= CALIBRATOR["max_iter"]:
        raise PipelineError("校准尚未收敛；请检查数据，不使用未收敛结果。")
    model = copy.deepcopy(base)
    model["calibration"] = {
        **CALIBRATOR,
        "slope": slope,
        "intercept": intercept,
        "match_ids": sorted(calibration["match_id"].unique().to_list()),
        "sample_count": calibration.height,
        "round_count": calibration.select("demo_sha256", "round_id").unique().height,
        "weighting": "one total weight per round; no base-model refitting",
    }
    return model


def metrics(frame: pl.DataFrame, probabilities: list[float]) -> dict[str, Any]:
    result = probability_metrics(frame["y_ct_win"].to_list(), probabilities, round_weights(frame))
    result["ece_10_bins"] = (
        sum(
            b["weight"] * abs(b["mean_probability"] - b["ct_win_frequency"])
            for b in result["calibration_bins"]
        )
        / result["round_weight_sum"]
    )
    return result


def cluster_bootstrap(
    frame: pl.DataFrame,
    base: list[float],
    calibrated: list[float],
    *,
    draws: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired bootstrap of whole matches, never treating correlated ticks as independent."""
    import numpy as np

    matches = sorted(frame["match_id"].unique().to_list())
    if len(matches) < 2 or draws < 2:
        raise PipelineError("比赛聚类 bootstrap 至少需要两场比赛和两次抽样。")
    # Labels are UInt8 in Parquet: convert before negation to avoid unsigned wraparound.
    y = frame["y_ct_win"].to_numpy().astype(np.float64)
    w = np.asarray(round_weights(frame))
    identifiers = frame["match_id"].to_numpy()
    components = []
    # Fixed clip only avoids log(0); it is never optimized against test results.
    for p in (np.clip(base, 1e-15, 1 - 1e-15), np.clip(calibrated, 1e-15, 1 - 1e-15)):
        losses = -y * np.log(p) - (1 - y) * np.log1p(-p)
        briers = (p - y) ** 2
        components.append(
            np.array(
                [
                    [
                        np.sum(w[identifiers == m] * losses[identifiers == m]),
                        np.sum(w[identifiers == m] * briers[identifiers == m]),
                        np.sum(w[identifiers == m]),
                    ]
                    for m in matches
                ]
            )
        )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(matches), size=(draws, len(matches)))
    estimates = [c[indices].sum(axis=1) for c in components]
    scores = [c[:, :2] / c[:, 2, None] for c in estimates]
    intervals = {}
    for name, values in zip(
        ("base", "calibrated", "calibrated_minus_base"),
        (scores[0], scores[1], scores[1] - scores[0]),
        strict=True,
    ):
        intervals[name] = {
            metric: [float(v) for v in np.quantile(values[:, i], [0.025, 0.975])]
            for i, metric in enumerate(("log_loss", "brier_score"))
        }
    return {
        "unit": "match_id",
        "match_count": len(matches),
        "draws": draws,
        "seed": seed,
        "percentile_95_intervals": intervals,
        "interpretation": "Within-event match uncertainty; not between-event variation.",
    }


def calibrate_experiment(
    dataset: Path, reference_dataset: Path, base_model: Path, out: Path
) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"输出目录已存在：{out}；请使用新路径，不能覆盖已评估实验。")
    metadata, tables = read_dataset(dataset)
    policy = metadata["split_policy"]
    if policy.get("method") != "frozen_four_way":
        raise PipelineError("校准需要冻结的 train/validation/calibration/test 四集合计划。")
    plan = policy["plan"]
    if (
        sha256_file(base_model) != plan.get("base_model_sha256")
        or sha256_file(reference_dataset / "dataset.json") != plan.get("reference_dataset_sha256")
        or plan.get("calibrator") != CALIBRATOR
    ):
        raise PipelineError("冻结模型、参考训练数据或校准方法与预先声明的计划不符。")
    reference_metadata, reference_tables = read_dataset(reference_dataset)
    base = json.loads(base_model.read_text(encoding="utf-8"))
    if (
        base.get("dataset_manifest_sha256") != plan["reference_dataset_sha256"]
        or base.get("map_name") != metadata["map_name"]
        or reference_metadata["map_name"] != metadata["map_name"]
        or base.get("feature_version", 1) != metadata.get("feature_version", 1)
        or plan.get("feature_version") != metadata.get("feature_version", 1)
    ):
        raise PipelineError("冻结模型与当前数据集地图/特征版本或原始训练来源不一致。")
    frame, reference = experiment_frame(tables), experiment_frame(reference_tables)
    columns = feature_columns(metadata.get("feature_version", 1))
    training = frame.filter(pl.col("split") == "train")
    old_training = reference.filter(pl.col("split") == "train")
    comparison = ["sample_id", *columns, "y_ct_win", "match_id", "demo_sha256", "round_id"]
    if not training.select(comparison).equals(old_training.select(comparison)):
        raise PipelineError("当前训练样本与冻结模型原始训练数据不同；拒绝悄悄重训或借用新赛事。")
    model = fit_calibrator(base, frame)
    model.update(
        base_model_sha256=plan["base_model_sha256"],
        dataset_manifest_sha256=sha256_file(dataset / "dataset.json"),
        split_plan_sha256=policy["plan_sha256"],
        run_mode="frozen_event_test_calibrated",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".calibration-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        # Freeze/reload the calibrated artifact before reading test labels for scoring.
        write_json(stage / "model.json", model)
        model = json.loads((stage / "model.json").read_text(encoding="utf-8"))
        frozen_hash = sha256_file(stage / "model.json")
        test = frame.filter(pl.col("split") == "test")
        x = test.select("sample_id", *columns)
        base_p = predict_probabilities(base, x)["p_ct_win"].to_list()
        calibrated_p = predict_probabilities(model, x)["p_ct_win"].to_list()
        prior = sum(
            y * w for y, w in zip(training["y_ct_win"], round_weights(training), strict=True)
        )
        prior /= sum(round_weights(training))
        prediction = test.select("sample_id", "match_id", "demo_sha256", "round_id").with_columns(
            pl.Series("p_ct_win_base", base_p),
            pl.Series("p_ct_win", calibrated_p),
            pl.lit("test").alias("split"),
        )
        per_match = []
        for match in sorted(test["match_id"].unique()):
            selected = test.filter(pl.col("match_id") == match)
            p = prediction.filter(pl.col("match_id") == match)
            per_match.append(
                {
                    "match_id": match,
                    "base": metrics(selected, p["p_ct_win_base"].to_list()),
                    "calibrated": metrics(selected, p["p_ct_win"].to_list()),
                }
            )
        counts = {}
        for split in policy["partitions"]:
            subset = frame.filter(pl.col("split") == split)
            counts[split] = {
                "matches": subset["match_id"].n_unique(),
                "rounds": subset.select("demo_sha256", "round_id").unique().height,
                "samples": subset.height,
            }
        report = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "map_name": metadata["map_name"],
            "dataset_manifest_sha256": model["dataset_manifest_sha256"],
            "calibrated_model_sha256": frozen_hash,
            "split_plan_sha256": policy["plan_sha256"],
            "base_model_sha256": plan["base_model_sha256"],
            "counts": counts,
            "calibration": model["calibration"],
            "training_round_prior": prior,
            "test": {
                "base": metrics(test, base_p),
                "calibrated": metrics(test, calibrated_p),
                "constant_training_prior": metrics(test, [prior] * test.height),
            },
            "paired_match_bootstrap": cluster_bootstrap(
                test,
                base_p,
                calibrated_p,
                draws=plan["bootstrap"]["draws"],
                seed=plan["bootstrap"]["seed"],
            ),
            "per_match": per_match,
            "versions": {name: version(name) for name in ("scikit-learn", "numpy", "polars")},
            "limitations": [
                "Test was used once for reporting, never choosing model/features/calibration.",
                "Calibration reuses the former Rio pilot test; old Rio scores are historical.",
                "Two calibration matches and one test event do not establish market readiness.",
                "V1 demo elapsed time includes pauses; no bomb context/rating/Elo inputs.",
                "This Cologne test is now observed; further tuning needs another unseen event.",
            ],
        }
        prediction.write_parquet(stage / "predictions.parquet", compression="zstd")
        if not pl.read_parquet(stage / "predictions.parquet").equals(prediction):
            raise PipelineError("校准预测 Parquet 重读不一致。")
        if sha256_file(stage / "model.json") != frozen_hash:
            raise PipelineError("测试评分期间模型发生变化。")
        write_json(stage / "report.json", report)
        stage.rename(out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference-dataset", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = calibrate_experiment(
            args.dataset, args.reference_dataset, args.base_model, args.out
        )
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"校准实验失败：{exc}", file=sys.stderr)
        return 1
    for name, values in report["test"].items():
        print(
            f"Cologne test {name}: LogLoss={values['log_loss']:.6f}, "
            f"Brier={values['brier_score']:.6f}, ECE={values['ece_10_bins']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Fit prespecified, event-separated exploratory models against the frozen V1."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import held_out_event, read_dataset, write_json
from cs2_pipeline.features import build_features, feature_columns, validate_feature_table
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_data import ROOT, context_for_samples, sources
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import round_weights

PLAN = ROOT / "configs/phase2_comparison_plan.json"


def check_split(training: pl.DataFrame, exploratory: pl.DataFrame, plan: dict[str, Any]) -> None:
    """Verify complete event/match separation and strict calendar ordering."""
    groups = [
        set(plan["candidate_fit_events"]),
        set(plan["candidate_calibration_events"]),
        set(plan["candidate_validation_events"]),
        set(exploratory["event"].unique().to_list()),
    ]
    for i, events in enumerate(groups):
        if not events or any(events & other for other in groups[i + 1 :]):
            raise PipelineError("拟合、校准、验证和探索集的赛事必须相互隔离。")
    if any(held_out_event(event) for events in groups for event in events):
        raise PipelineError("Bucharest 保留赛事不得参与实验。")
    parts = [training.filter(pl.col("event").is_in(list(events))) for events in groups[:3]]
    parts.append(exploratory)
    if any(part.is_empty() for part in parts):
        raise PipelineError("拟合、校准、验证或探索集为空。")
    for left, right in zip(parts, parts[1:], strict=False):
        if left["match_date"].max() >= right["match_date"].min():
            raise PipelineError("拟合、校准、验证和探索集日期必须严格先后排列。")
    ids = [set(part["match_id"].to_list()) for part in parts]
    for i, match_ids in enumerate(ids):
        if any(match_ids & other for other in ids[i + 1 :]):
            raise PipelineError("同一比赛不得跨数据集合。")
    full = set(plan["full_baseline_fit_events"])
    if full != groups[0] | groups[1] | groups[2]:
        raise PipelineError("完整训练基线与原 V1 的训练赛事不一致。")


def fit_logistic(frame: pl.DataFrame, columns: list[str]) -> dict[str, Any]:
    """Training-only median imputation and round-weighted scale and logistic fit."""
    if set(columns) - (set(feature_columns(1)) | set(feature_columns(2))):
        raise PipelineError("模型输入必须来自固定因果特征白名单。")
    y = frame["y_ct_win"].to_numpy()
    if set(y.tolist()) != {0, 1}:
        raise PipelineError("拟合集必须同时含双方胜利回合。")
    x = frame.select(columns).to_numpy()
    weights = np.asarray(round_weights(frame), dtype=float)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = imputer.fit_transform(x)
    mean = np.average(transformed, axis=0, weights=weights)
    variance = np.average((transformed - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[(np.ptp(transformed, axis=0) == 0) | (scale == 0)] = 1.0
    estimator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit((transformed - mean) / scale, y, sample_weight=weights)
    model = {
        "algorithm": "logistic_regression",
        "feature_columns": columns,
        "medians": imputer.statistics_.tolist(),
        "indicator_indices": imputer.indicator_.features_.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "iterations": int(estimator.n_iter_[0]),
    }
    if not np.allclose(
        predict_logistic(model, frame),
        estimator.predict_proba((transformed - mean) / scale)[:, 1],
        atol=1e-12,
        rtol=1e-12,
    ):
        raise PipelineError("逻辑回归导出推理不一致。")
    return model


def predict_logistic(model: dict[str, Any], frame: pl.DataFrame) -> np.ndarray:
    x = frame.select(model["feature_columns"]).to_numpy().astype(float)
    missing = np.isnan(x)
    x = np.where(missing, np.asarray(model["medians"]), x)
    indicators = model["indicator_indices"]
    if indicators:
        x = np.column_stack([x, missing[:, indicators].astype(float)])
    logits = ((x - np.asarray(model["mean"])) / np.asarray(model["scale"])) @ np.asarray(
        model["coefficients"]
    ) + model["intercept"]
    return sigmoid(logits)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -700, 700)))


def fit_sigmoid(p: np.ndarray, calibration: pl.DataFrame) -> dict[str, Any]:
    """Fit only on the separate calibration event, never on exploration rows."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    estimator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42)
    estimator.fit(
        logits, calibration["y_ct_win"].to_numpy(), sample_weight=round_weights(calibration)
    )
    slope, intercept = float(estimator.coef_[0, 0]), float(estimator.intercept_[0])
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise PipelineError("校准参数无效或反转排序；请增加独立校准数据。")
    return {"method": "sigmoid_logit", "slope": slope, "intercept": intercept}


def apply_sigmoid(p: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return sigmoid(calibrator["slope"] * np.log(p / (1 - p)) + calibrator["intercept"])


def load_frame(path: Path, items: list[dict[str, Any]]) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    metadata, tables = read_dataset(path)
    samples = tables["samples"]
    contextual, quality = context_for_samples(ROOT, path.name, items, samples)
    v2 = build_features(tables["nodes"], contextual, feature_version=2)
    validate_feature_table(v2, feature_version=2)
    base = samples.select(
        "sample_id", "match_id", "match_date", "event", "demo_sha256", "round_id"
    ).join(tables["labels"], on="sample_id", validate="1:1")
    first = base.join(tables["features"], on="sample_id", validate="1:1").sort("sample_id")
    second = base.join(v2, on="sample_id", validate="1:1").sort("sample_id")
    if not first["sample_id"].equals(second["sample_id"]):
        raise PipelineError("V1 和 V2 样本不完全相同。")
    common = feature_columns(1)[1:]
    if not first.select(common).equals(second.select(common)):
        raise PipelineError("V1 与 V2 的共同状态特征不一致，拒绝不公平的比较。")
    return first, second, {"metadata": metadata, "context_quality": quality}


def counts(frame: pl.DataFrame) -> dict[str, Any]:
    return {
        "matches": frame["match_id"].n_unique(),
        "rounds": frame.select("demo_sha256", "round_id").n_unique(),
        "samples": frame.height,
        "first_date": frame["match_date"].min(),
        "last_date": frame["match_date"].max(),
        "events": sorted(frame["event"].unique().to_list()),
    }


def run(dataset: Path, out: Path, reporter: ProgressReporter | None = None) -> dict[str, Any]:
    import joblib

    if out.exists():
        raise PipelineError(f"输出目录已存在：{out}；请使用新路径。")
    plan = json.loads(PLAN.read_text("utf-8"))
    if dataset.resolve() != (ROOT / plan["original_training_dataset"]).resolve():
        raise PipelineError("该固定实验仅允许使用计划中已验证的原始训练数据集。")
    evaluation = ROOT / plan["exploratory_dataset"]
    frozen_path = ROOT / plan["frozen_v1"]
    if sha256_file(frozen_path) != plan["frozen_v1_sha256"]:
        raise PipelineError("固定 V1 模型 SHA256 改变。")
    emit(reporter, "load", "active", "核对数据、标签、上下文与赛事时间划分。")
    items = sources(ROOT)
    old1, old2, old_info = load_frame(dataset, items)
    test1, test2, test_info = load_frame(evaluation, items)
    if test_info["metadata"]["split_policy"]["method"] != "holdout_only":
        raise PipelineError("50 场数据必须保持不可训练的 holdout_only 策略。")
    check_split(old1, test1, plan)

    def part(frame: pl.DataFrame, key: str) -> pl.DataFrame:
        return frame.filter(pl.col("event").is_in(plan[key]))

    fit1, fit2 = (part(frame, "candidate_fit_events") for frame in (old1, old2))
    cal1, cal2 = (part(frame, "candidate_calibration_events") for frame in (old1, old2))
    val1, val2 = (part(frame, "candidate_validation_events") for frame in (old1, old2))
    full1 = part(old1, "full_baseline_fit_events")
    frozen = json.loads(frozen_path.read_text("utf-8"))
    emit(
        reporter,
        "load",
        "completed",
        "数据与按赛事、时间隔离检查通过。",
        fit_matches=fit1["match_id"].n_unique(),
        exploration_matches=test1["match_id"].n_unique(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".comparison-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_json(stage / "plan.json", plan)
        simple = plan["model_specs"]["simple"]["features"]
        models = {}
        emit(reporter, "fit", "active", "拟合与 V1 相同 34 场训练数据的人数差＋时间基线。")
        models["simple_full34"] = fit_logistic(full1, simple)
        for name, frame, columns in (
            ("simple_fit18", fit1, simple),
            ("v1_fit18", fit1, feature_columns(1)),
            ("v2_fit18", fit2, feature_columns(2)),
        ):
            emit(reporter, "fit", "active", f"拟合 {name}；仅使用 Rio 和 Atlanta。")
            models[name] = fit_logistic(frame, columns)
        tree_spec = plan["model_specs"]["v1_tree"]
        parameters = {
            k: v for k, v in tree_spec.items() if k not in ("algorithm", "feature_version")
        }
        tree = HistGradientBoostingClassifier(**parameters, random_state=plan["seed"])
        emit(reporter, "fit", "active", "拟合浅层非线性模型；禁用随机时点早停。")
        tree.fit(
            fit1.select(feature_columns(1)).to_numpy(),
            fit1["y_ct_win"].to_numpy(),
            sample_weight=round_weights(fit1),
        )
        joblib.dump(tree, stage / "v1_tree_fit18.joblib")
        restored = joblib.load(stage / "v1_tree_fit18.joblib")
        if not np.array_equal(
            tree.predict_proba(val1.select(feature_columns(1)).to_numpy()),
            restored.predict_proba(val1.select(feature_columns(1)).to_numpy()),
        ):
            raise PipelineError("非线性模型保存后重读不一致。")
        emit(
            reporter, "fit", "completed", "所有候选模型已拟合；现使用独立 Cologne 赛事拟合校准器。"
        )
        calibrators = {}
        for name in ("simple_fit18", "v1_fit18", "v2_fit18", "v1_tree_fit18"):
            frame = cal2 if name == "v2_fit18" else cal1
            p = (
                tree.predict_proba(frame.select(feature_columns(1)).to_numpy())[:, 1]
                if name == "v1_tree_fit18"
                else predict_logistic(models[name], frame)
            )
            calibrators[name] = fit_sigmoid(p, frame)
        write_json(stage / "logistic_models.json", models)
        write_json(stage / "calibrators.json", calibrators)
        emit(reporter, "evaluate", "active", "比较相同时点的未校准、独立校准及固定 V1 概率。")
        report: dict[str, Any] = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256_file(PLAN),
            "frozen_v1_sha256": sha256_file(frozen_path),
            "dataset_sha256": sha256_file(dataset / "dataset.json"),
            "exploratory_dataset_sha256": sha256_file(evaluation / "dataset.json"),
            "code_sha256": {
                name: sha256_file(ROOT / name)
                for name in (
                    "code/src/cs2_pipeline/phase2_compare.py",
                    "code/src/cs2_pipeline/phase2_data.py",
                    "code/src/cs2_pipeline/features.py",
                    "code/src/cs2_pipeline/context.py",
                )
            },
            "versions": {name: version(name) for name in ("scikit-learn", "numpy", "polars")},
            "partitions": {
                "fit": counts(fit1),
                "calibration": counts(cal1),
                "validation": counts(val1),
                "full_v1_train": counts(full1),
                "observed_exploration": counts(test1),
            },
            "context_quality": {
                "historical": old_info["context_quality"],
                "exploration": test_info["context_quality"],
            },
            "calibrators": calibrators,
            "results": {},
            "interpretation": "All 50 scores are exploratory, not a new blind test. "
            "The frozen V1 and simple_full34 use more fit data than fit18 candidates.",
        }
        for split, f1, f2 in (("validation", val1, val2), ("exploration", test1, test2)):
            # V1 and full34 were fitted on validation Porto; omit their validation scores.
            preds = {}
            if split == "exploration":
                preds["frozen_v1"] = predict_probabilities(
                    frozen, f1.select("sample_id", *feature_columns(1))
                )["p_ct_win"].to_numpy()
                preds["simple_full34"] = predict_logistic(models["simple_full34"], f1)
            for name in ("simple_fit18", "v1_fit18", "v2_fit18", "v1_tree_fit18"):
                frame = f2 if name == "v2_fit18" else f1
                p = (
                    tree.predict_proba(frame.select(feature_columns(1)).to_numpy())[:, 1]
                    if name == "v1_tree_fit18"
                    else predict_logistic(models[name], frame)
                )
                preds[name] = p
                preds[name + "_calibrated"] = apply_sigmoid(p, calibrators[name])
            results = {name: metrics(f1, p.tolist()) for name, p in preds.items()}
            paired = {}
            if split == "exploration":
                for name, p in preds.items():
                    if name == "frozen_v1":
                        continue
                    paired[name] = cluster_bootstrap(
                        f1, preds["frozen_v1"].tolist(), p.tolist(), draws=2000, seed=42
                    )
                pred_frame = f1.select("sample_id").with_columns(
                    [pl.Series(name, p, dtype=pl.Float64) for name, p in preds.items()]
                )
                pred_frame.write_parquet(stage / "predictions.parquet", compression="zstd")
                if not pl.read_parquet(stage / "predictions.parquet").equals(pred_frame):
                    raise PipelineError("预测 Parquet 保存后不一致。")
            report["results"][split] = results
            report.setdefault("paired_match_bootstrap", {})[split] = paired
        # Preserve the existing monitor's well-defined metrics schema and explicitly label the data.
        report["metrics"] = {
            "validation": {"model": report["results"]["validation"]["v1_fit18"]},
            "test": {"model": report["results"]["exploration"]["frozen_v1"]},
        }
        report["run_mode"] = "observed_exploration_not_blind_test"
        report["artifact_sha256"] = {
            name: sha256_file(stage / name)
            for name in (
                "plan.json",
                "logistic_models.json",
                "calibrators.json",
                "v1_tree_fit18.joblib",
                "predictions.parquet",
            )
        }
        emit(
            reporter,
            "evaluate",
            "completed",
            "验证及已观察 50 场的成对比较完成。",
            split="validation",
            metrics=report["metrics"]["validation"]["model"],
        )
        emit(reporter, "save", "active", "保存模型、概率、校准参数及可复现报告。")
        write_json(stage / "report.json", report)
        stage.rename(out)
    emit(reporter, "save", "completed", "实验输出已保存。")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/datasets/mirage_expanded_2026_v1"
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args(argv)
    reporter = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        run(args.dataset, args.out, reporter)
        emit(reporter, "run", "completed", "探索性模型对照完成。")
        return 0
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pl.exceptions.PolarsError,
        ConvergenceWarning,
    ) as exc:
        emit(reporter, "run", "failed", f"模型对照失败：{exc}")
        print(f"模型对照失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

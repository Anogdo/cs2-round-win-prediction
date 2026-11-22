"""Train a round-balanced logistic baseline without cross-match leakage."""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import HISTORICAL_VALIDATION_ONLY, read_dataset, write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError


def round_weights(frame: pl.DataFrame) -> list[float]:
    """Every round has total weight one regardless of its number of sample ticks."""
    return frame.select((1.0 / pl.len().over("demo_sha256", "round_id")).alias("weight"))[
        "weight"
    ].to_list()


def probability_metrics(
    labels: list[int],
    probabilities: list[float],
    weights: list[float],
) -> dict[str, Any]:
    """Round-weighted proper scores, with weighted calibration bins and optional AUC."""
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

    result = {
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1], sample_weight=weights)),
        "brier_score": float(brier_score_loss(labels, probabilities, sample_weight=weights)),
        "accuracy_at_0_5": float(
            accuracy_score(
                labels,
                [p >= 0.5 for p in probabilities],
                sample_weight=weights,
            )
        ),
        "roc_auc": float(roc_auc_score(labels, probabilities, sample_weight=weights))
        if len(set(labels)) == 2
        else None,
        "sample_count": len(labels),
        "round_weight_sum": float(sum(weights)),
    }
    bins = []
    for index in range(10):
        selected = [
            (y, p, w)
            for y, p, w in zip(labels, probabilities, weights, strict=True)
            if min(9, int(p * 10)) == index
        ]
        if not selected:
            continue
        mass = sum(row[2] for row in selected)
        bins.append(
            {
                "lower": index / 10,
                "upper": (index + 1) / 10,
                "count": len(selected),
                "weight": mass,
                "mean_probability": sum(p * w for _, p, w in selected) / mass,
                "ct_win_frequency": sum(y * w for y, _, w in selected) / mass,
            }
        )
    result["calibration_bins"] = bins
    return result


def fit_baseline(
    training: pl.DataFrame,
    *,
    feature_version: int = 1,
    progress: ProgressReporter | None = None,
) -> tuple[dict[str, Any], float]:
    """Fit imputer, scaler and classifier exclusively on the training partition."""
    emit(progress, "fit", "active", "加载训练依赖，准备训练集预处理。")
    try:
        import numpy as np
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise PipelineError(
            "缺少 ML 依赖；请执行 uv sync --frozen --extra dev --extra ml。"
        ) from exc
    labels = training["y_ct_win"].to_list()
    columns = feature_columns(feature_version)
    if set(labels) != {0, 1}:
        raise PipelineError("训练集必须同时有 CT 胜和 T 胜回合；请增加独立比赛。")
    weights = round_weights(training)

    class StableWeightedScaler(StandardScaler):
        """Two-pass variance stays nonnegative for constant weighted columns.

        sklearn's incremental weighted variance can round slightly below zero.
        Keep the same round weights and train-only fit, avoiding a sqrt warning.
        """

        def fit(self, x: Any, y: Any = None, sample_weight: Any = None) -> Any:
            super().fit(x, y)
            if sample_weight is not None:
                self.mean_ = np.average(x, axis=0, weights=sample_weight)
                self.var_ = np.average((x - self.mean_) ** 2, axis=0, weights=sample_weight)
                self.scale_ = np.sqrt(self.var_)
                self.scale_[(np.ptp(x, axis=0) == 0) | (self.scale_ == 0)] = 1.0
                self.n_samples_seen_ = float(sum(sample_weight))
            return self

    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StableWeightedScaler()),
            (
                "classifier",
                LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42),
            ),
        ]
    )
    x = training.select(columns).to_numpy()
    emit(
        progress,
        "fit",
        "active",
        "正在拟合训练集预处理和逻辑回归。",
        sample_count=training.height,
        feature_count=len(columns),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            pipeline.fit(x, labels, scale__sample_weight=weights, classifier__sample_weight=weights)
        except ConvergenceWarning as exc:
            raise PipelineError("逻辑回归未收敛；请检查异常特征和样本量后重新训练。") from exc
    imputer, scale, classifier = (
        pipeline.named_steps[k] for k in ("imputer", "scale", "classifier")
    )
    model: dict[str, Any] = {
        "model_schema_version": 1,
        "algorithm": "logistic_regression",
        "target": "CT round win probability",
        "feature_columns": columns,
        "feature_version": feature_version,
        "transformed_feature_names": imputer.get_feature_names_out(columns).tolist(),
        "parameters": {
            "imputation_values": imputer.statistics_.tolist(),
            "indicator_indices": imputer.indicator_.features_.tolist(),
            "scale_mean": scale.mean_.tolist(),
            "scale_std": scale.scale_.tolist(),
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
        },
        "training": {
            "optimizer_iterations": int(classifier.n_iter_[0]),
            "sample_count": training.height,
            "round_count": training.select("demo_sha256", "round_id").unique().height,
            "match_count": training["match_id"].n_unique(),
            "sample_weight": "1 / samples_in_same_round; each round totals one",
            "scaling": "two-pass weighted variance; constant columns use unit scale",
            "all_missing_training_features": [
                c for c in columns if training[c].null_count() == training.height
            ],
        },
        "versions": {k: version(k) for k in ("scikit-learn", "numpy", "polars")},
    }
    # Exported JSON must predict exactly as the sklearn pipeline before publication.
    exported = predict_probabilities(model, training.select("sample_id", *columns))
    if not np.allclose(
        exported["p_ct_win"].to_numpy(), pipeline.predict_proba(x)[:, 1], rtol=1e-12, atol=1e-12
    ):
        raise PipelineError("导出模型与 sklearn 推理不一致，拒绝保存。")
    prior = sum(y * w for y, w in zip(labels, weights, strict=True)) / sum(weights)
    emit(
        progress,
        "fit",
        "completed",
        "模型拟合完成，JSON 推理一致性检查通过。",
        iterations=int(classifier.n_iter_[0]),
        training_log_loss=probability_metrics(labels, exported["p_ct_win"].to_list(), weights)[
            "log_loss"
        ]
        if progress is not None
        else None,
    )
    return model, prior


def train_baseline(
    dataset: Path,
    out: Path,
    *,
    smoke_test: bool = False,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    emit(progress, "load", "active", "读取数据集，校验文件哈希、特征和集合划分。")
    if out.exists():
        raise PipelineError(f"模型输出目录已存在：{out}；请使用新的 --out。")
    metadata, tables = read_dataset(dataset)
    columns = feature_columns(metadata.get("feature_version", 1))
    if metadata["split_policy"].get("method") in ("holdout_only", HISTORICAL_VALIDATION_ONLY):
        raise PipelineError("独立留出数据集不能用于训练，包含 smoke test。")
    if not smoke_test and not metadata["split_policy"]["evaluation_ready"]:
        raise PipelineError(
            "独立比赛/日期区间不足，不能正式评估。请增加至少 3 个独立时间区间的比赛；"
            "如仅验证训练与推理流程，请显式使用 --smoke-test。"
        )
    frame = (
        tables["features"]
        .join(tables["labels"], on="sample_id", validate="1:1")
        .join(
            tables["samples"].select("sample_id", "demo_sha256", "round_id", "match_id", "event"),
            on="sample_id",
            validate="1:1",
        )
        .join(tables["splits"], on="sample_id", validate="1:1")
        .sort("sample_id")
    )
    if not smoke_test and set(frame["split"].to_list()) != {"train", "validation", "test"}:
        raise PipelineError("正式评估必须同时有 train/validation/test 三个非空集合。")
    training = frame.filter(pl.col("split") == "train")
    emit(
        progress,
        "load",
        "completed",
        "数据校验完成，训练集合已隔离。",
        sample_count=training.height,
        feature_count=len(columns),
        round_count=training.select("demo_sha256", "round_id").unique().height,
        match_count=training["match_id"].n_unique(),
    )
    model, prior = fit_baseline(
        training, feature_version=metadata.get("feature_version", 1), progress=progress
    )
    model["map_name"] = metadata["map_name"]
    model["run_mode"] = "smoke_test" if smoke_test else "holdout_evaluation"
    model["dataset_manifest_sha256"] = sha256_file(dataset / "dataset.json")
    predictions = []
    metrics: dict[str, Any] | None = None if smoke_test else {}
    emit(progress, "evaluate", "active", "使用冻结参数推理并计算回合加权指标。")
    for split in ["train"] if smoke_test else ["validation", "test"]:
        partition = frame.filter(pl.col("split") == split)
        predicted = predict_probabilities(model, partition.select("sample_id", *columns))
        predictions.append(
            predicted.with_columns(
                pl.lit("train_smoke" if smoke_test else split).alias("split"),
                pl.lit(prior).alias("training_round_prior"),
            )
        )
        if metrics is not None:
            y, weights = partition["y_ct_win"].to_list(), round_weights(partition)
            metrics[split] = {
                "model": probability_metrics(y, predicted["p_ct_win"].to_list(), weights),
                "constant_training_prior": probability_metrics(y, [prior] * len(y), weights),
                "match_count": partition["match_id"].n_unique(),
                "round_count": partition.select("demo_sha256", "round_id").unique().height,
            }
            emit(
                progress,
                "evaluate",
                "active",
                f"{split} 评分完成。",
                split=split,
                metrics=metrics[split]["model"],
            )
    emit(progress, "evaluate", "completed", "评估完成。")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": model["run_mode"],
        "independent_evaluation_performed": not smoke_test,
        "metrics": metrics,
        "training_round_prior": prior,
        "training": model["training"],
        "split_events": {
            split: sorted(frame.filter(pl.col("split") == split)["event"].unique().to_list())
            for split in frame["split"].unique().to_list()
        },
        "feature_count": len(columns),
        "split_policy": metadata["split_policy"],
        "dataset_manifest_sha256": model["dataset_manifest_sha256"],
        "limitations": metadata["limitations"]
        + (
            [
                "Smoke-test: fitted and predicted on the same matches; no holdout score exists.",
            ]
            if smoke_test
            else [
                "No tuning/calibration; small holdouts do not establish probability reliability.",
            ]
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    emit(progress, "save", "active", "保存模型、报告和预测，重新读取 Parquet 校验。")
    with tempfile.TemporaryDirectory(prefix=".model-stage-", dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        write_json(stage / "model.json", model)
        write_json(stage / "report.json", report)
        prediction_frame = pl.concat(predictions)
        prediction_frame.write_parquet(stage / "predictions.parquet", compression="zstd")
        if not pl.read_parquet(stage / "predictions.parquet").equals(prediction_frame):
            raise PipelineError("训练预测文件无法正确重读。")
        stage.rename(out)
    emit(progress, "save", "completed", "文件已保存，重读校验通过。", output=str(out.resolve()))
    emit(progress, "run", "completed", "训练任务完成。")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/mirage_v1"))
    parser.add_argument("--out", type=Path, default=Path("models/logistic_v1"))
    parser.add_argument("--smoke-test", action="store_true", help="仅验证流程，不报告评估指标")
    parser.add_argument("--progress-file", type=Path, help="可选 JSONL 实时阶段日志")
    args = parser.parse_args(argv)
    try:
        report = train_baseline(
            args.dataset,
            args.out,
            smoke_test=args.smoke_test,
            progress=ProgressReporter(args.progress_file) if args.progress_file else None,
        )
    except (
        PipelineError,
        OSError,
        ValueError,
        TypeError,
        ImportError,
        pl.exceptions.PolarsError,
    ) as exc:
        if args.progress_file:
            emit(ProgressReporter(args.progress_file), "run", "failed", f"训练失败：{exc}")
        print(f"训练失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"模型：{args.out.resolve()} | 模式：{report['run_mode']} | "
        f"训练回合：{report['training']['round_count']} | 特征：{report['feature_count']}"
    )
    if args.smoke_test:
        print("已完成训练/推理流程检查；没有独立评估，不报告准确率或泛化性能。")
    else:
        for split, values in report["metrics"].items():
            print(
                f"{split}: log_loss={values['model']['log_loss']:.4f}, "
                f"Brier={values['model']['brier_score']:.4f}, "
                f"constant_prior_log_loss={values['constant_training_prior']['log_loss']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

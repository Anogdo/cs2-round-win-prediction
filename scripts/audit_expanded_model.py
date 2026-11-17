"""Recompute metrics for predeclared models on the same verified holdout samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from cs2_pipeline.calibrate import metrics
from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.schema import PipelineError


def audit(dataset: Path, manifest: Path, run: Path) -> dict:
    metadata, tables = read_dataset(dataset)
    config = json.loads(manifest.read_text(encoding="utf-8"))
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    model = json.loads((run / "model.json").read_text(encoding="utf-8"))
    if metadata["source_manifest_sha256"] != sha256_file(manifest):
        raise PipelineError("数据集与预先声明的输入清单不符。")
    digest = sha256_file(dataset / "dataset.json")
    if model["dataset_manifest_sha256"] != digest or report["dataset_manifest_sha256"] != digest:
        raise PipelineError("模型或训练报告没有绑定此数据集。")
    previous_path = Path(config["model_plan"]["comparison_model"])
    if sha256_file(previous_path) != config["model_plan"]["comparison_model_sha256"]:
        raise PipelineError("事先声明的旧模型已改变。")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if model.get("calibration") or model["feature_columns"] != feature_columns(1):
        raise PipelineError("扩充模型必须是预先声明的未校准 V1。")
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
    saved = pl.read_parquet(run / "predictions.parquet").sort("sample_id")
    holdout = frame.filter(pl.col("split").is_in(["validation", "test"]))
    if saved["sample_id"].to_list() != holdout["sample_id"].to_list():
        raise PipelineError("预测文件与验证/测试样本不一致。")
    result = {}
    for split in ("validation", "test"):
        part = holdout.filter(pl.col("split") == split)
        x = part.select("sample_id", *model["feature_columns"])
        new_predictions = predict_probabilities(model, x)
        old_predictions = predict_probabilities(previous, x)
        stored = saved.filter(pl.col("split") == split).select("sample_id", "p_ct_win")
        if not stored.equals(new_predictions):
            raise PipelineError(f"{split} 模型 JSON 重新推理与已保存预测不同。")
        new_metrics = metrics(part, new_predictions["p_ct_win"].to_list())
        old_metrics = metrics(part, old_predictions["p_ct_win"].to_list())
        for key in ("log_loss", "brier_score", "roc_auc", "accuracy_at_0_5"):
            actual, original = new_metrics[key], report["metrics"][split]["model"][key]
            if actual is None and original is None:
                continue
            if actual is None or original is None or abs(actual - original) > 1e-12:
                raise PipelineError(f"{split} 的 {key} 与训练报告不一致。")
        per_match = []
        for match in sorted(part["match_id"].unique().to_list()):
            subset = part.filter(pl.col("match_id") == match)
            new_p = new_predictions.join(subset.select("sample_id"), on="sample_id", how="semi")
            old_p = old_predictions.join(subset.select("sample_id"), on="sample_id", how="semi")
            per_match.append(
                {
                    "match_id": match,
                    "rounds": subset["round_id"].n_unique(),
                    "expanded": metrics(subset, new_p["p_ct_win"].to_list()),
                    "previous": metrics(subset, old_p["p_ct_win"].to_list()),
                }
            )
        result[split] = {
            "events": sorted(part["event"].unique().to_list()),
            "matches": part["match_id"].n_unique(),
            "rounds": part.select("demo_sha256", "round_id").unique().height,
            "samples": part.height,
            "expanded": new_metrics,
            "previous": old_metrics,
            "constant_prior": report["metrics"][split]["constant_training_prior"],
            "per_match": per_match,
        }
    return {
        "passed": True,
        "dataset_sha256": digest,
        "model_sha256": sha256_file(run / "model.json"),
        "previous_model_sha256": sha256_file(previous_path),
        "training": model["training"],
        "partitions": result,
        "limits": [
            "Only two validation and two test matches; no robust cross-event claim.",
            "Old model used older parser during fitting; both were scored on Awpy 3.3 data.",
            "This BLAST test is now observed and cannot be reused for further tuning.",
            "Current new model is uncalibrated; no calibration was fit on the test set.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/datasets/mirage_expanded_2026_v1")
    )
    parser.add_argument("--manifest", type=Path, default=Path("configs/phase2_expanded_2026.json"))
    parser.add_argument("--out", type=Path, default=Path("reports/expanded_2026_model_audit.json"))
    args = parser.parse_args()
    if args.out.exists():
        raise PipelineError("审计报告已存在；请使用新的 --out，避免覆盖。")
    result = audit(args.dataset, args.manifest, args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    for split, item in result["partitions"].items():
        print(
            f"{split}: 扩充 LogLoss={item['expanded']['log_loss']:.6f}, "
            f"旧模型 LogLoss={item['previous']['log_loss']:.6f}, "
            f"常数先验 LogLoss={item['constant_prior']['log_loss']:.6f}"
        )


if __name__ == "__main__":
    main()

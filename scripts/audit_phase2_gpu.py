"""Independently reload GPU models and reproduce their observed-match scores."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.calibrate import metrics
from cs2_pipeline.dataset import write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import apply_sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_gpu import load_v1, predict
from cs2_pipeline.schema import PipelineError


def run(directory: Path) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise PipelineError("请先安装 gpu 依赖以重读 XGBoost 模型。") from exc
    report = json.loads((directory / "report.json").read_text("utf-8"))
    plan = json.loads((directory / "plan.json").read_text("utf-8"))
    if sha256_file(ROOT / "src/cs2_pipeline/phase2_gpu.py") != report["code_sha256"]:
        raise PipelineError("GPU 训练代码哈希已改变。")
    for filename, expected in report["artifact_sha256"].items():
        if sha256_file(directory / filename) != expected:
            raise PipelineError(f"GPU 产物哈希不匹配：{filename}")
    dataset = ROOT / plan["exploratory_dataset"]
    if sha256_file(dataset / "dataset.json") != report["exploratory_dataset_sha256"]:
        raise PipelineError("GPU 报告与探索数据集不一致。")
    meta, frame = load_v1(dataset)
    if meta["split_policy"]["method"] != "holdout_only":
        raise PipelineError("探索集不再处于 holdout_only 模式。")
    saved = pl.read_parquet(directory / "predictions.parquet").sort("sample_id")
    if not saved["sample_id"].equals(frame["sample_id"]):
        raise PipelineError("模型概率与探索样本未对齐。")
    calibrator = json.loads((directory / "calibrator.json").read_text("utf-8"))
    proofs = {}
    maximum_difference = {}
    for name, column in (("fit18", "gpu_fit18"), ("full34", "gpu_full34")):
        evidence = report["training"][name]
        if evidence["device"] != "cuda:0" or evidence["boosting_rounds"] != 120:
            raise PipelineError(f"{name} 缺少实际 CUDA 训练配置。")
        if not any(
            proof["gpu_uuid"] == report["gpu"]["uuid"] and proof["pid"] > 0
            for proof in evidence["compute_app_observations"]
        ):
            raise PipelineError(f"{name} 缺少训练进程在对应 GPU 上的记录。")
        booster = xgb.Booster()
        booster.load_model(directory / f"xgboost_{name}.json")
        # Independent CPU inference checks the saved artifact, not the training device.
        booster.set_param({"device": "cpu"})
        predicted = predict(booster, frame)
        difference = float(np.max(np.abs(predicted - saved[column].to_numpy())))
        if difference > 1e-6:
            raise PipelineError(f"重读模型 {name} 的概率差异过大：{difference}")
        maximum_difference[column] = difference
        proofs[name] = evidence["compute_app_observations"]
    calibrated = apply_sigmoid(saved["gpu_fit18"].to_numpy(), calibrator)
    difference = float(np.max(np.abs(calibrated - saved["gpu_fit18_calibrated"].to_numpy())))
    if difference > 1e-12:
        raise PipelineError("独立校准参数不能重现预测。")
    maximum_difference["gpu_fit18_calibrated"] = difference
    scores = {}
    for column in ("gpu_fit18", "gpu_fit18_calibrated", "gpu_full34"):
        score = metrics(frame, saved[column].to_list())
        for metric in ("log_loss", "brier_score", "roc_auc", "ece_10_bins"):
            if abs(score[metric] - report["results"]["exploration"][column][metric]) > 1e-12:
                raise PipelineError(f"{column} 的 {metric} 与训练报告不符。")
        scores[column] = {
            k: score[k] for k in ("log_loss", "brier_score", "roc_auc", "ece_10_bins")
        }
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "gpu": report["gpu"],
        "cuda_build": report["xgboost_build_info"]["USE_CUDA"],
        "gpu_training_process_observations": proofs,
        "cpu_reload_max_probability_difference": maximum_difference,
        "matches": frame["match_id"].n_unique(),
        "rounds": frame.select("demo_sha256", "round_id").n_unique(),
        "samples": frame.height,
        "recomputed_scores": scores,
        "note": "CPU is used only for this independent model-reload audit; "
        "training was verified on CUDA.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-out", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/phase2_gpu_audit.json")
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise PipelineError("审计输出已存在；请指定新的 --out。")
        write_json(args.out, run(args.model_out))
        print(f"GPU 训练证据和模型重读审计通过：{args.out}")
        return 0
    except (OSError, PipelineError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        print(f"GPU 审计失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

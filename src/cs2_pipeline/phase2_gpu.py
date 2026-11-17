"""Train a CUDA-verified XGBoost candidate on fixed Phase 2 partitions."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import apply_sigmoid, check_split, counts, fit_sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import round_weights

GPU_PLAN = ROOT / "configs/phase2_gpu_plan.json"


def nvidia_csv(query: str, *, apps: bool = False) -> list[list[str]]:
    """Use NVML's supported CLI to independently observe the active process."""
    try:
        command = [
            "nvidia-smi",
            f"--query-{'compute-apps' if apps else 'gpu'}={query}",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PipelineError(f"无法核对 NVIDIA GPU：{exc}") from exc
    return [
        [value.strip() for value in row] for row in csv.reader(result.stdout.splitlines()) if row
    ]


def observation_for_pid(rows: list[list[str]], pid: int, uuid: str) -> dict[str, Any] | None:
    for row in rows:
        if len(row) >= 3 and row[0] == uuid and row[1].isdigit() and int(row[1]) == pid:
            return {"gpu_uuid": row[0], "pid": pid, "memory_mib": row[2]}
    return None


def load_v1(path: Path) -> tuple[dict[str, Any], pl.DataFrame]:
    meta, tables = read_dataset(path)
    frame = (
        tables["samples"]
        .select("sample_id", "match_id", "match_date", "event", "demo_sha256", "round_id")
        .join(tables["labels"], on="sample_id", validate="1:1")
        .join(tables["features"], on="sample_id", validate="1:1")
        .sort("sample_id")
    )
    return meta, frame


def fit_cuda(
    frame: pl.DataFrame,
    name: str,
    plan: dict[str, Any],
    gpu: dict[str, str],
    reporter: ProgressReporter | None,
) -> tuple[Any, dict[str, Any]]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise PipelineError(
            "缺少 GPU 依赖；请运行 uv sync --frozen --extra dev --extra ml "
            "--extra context --extra gpu。"
        ) from exc
    if not xgb.build_info().get("USE_CUDA"):
        raise PipelineError("XGBoost 安装包未启用 CUDA，拒绝 CPU 回退。")
    columns = feature_columns(1)
    recorded: list[dict[str, Any]] = []

    class CheckGPU(xgb.callback.TrainingCallback):
        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            step = epoch + 1
            if step in (1, 3, 10) and not recorded:
                record = observation_for_pid(
                    nvidia_csv("gpu_uuid,pid,used_gpu_memory", apps=True), os.getpid(), gpu["uuid"]
                )
                if record:
                    recorded.append({**record, "iteration": step})
            if step == 1 or step % 10 == 0 or step == plan["parameters"]["n_estimators"]:
                emit(
                    reporter,
                    "fit",
                    "active",
                    f"GPU 正在拟合 {name}：{step}/{plan['parameters']['n_estimators']} 轮。",
                    iterations=step,
                    total_iterations=plan["parameters"]["n_estimators"],
                    model=name,
                )
            return False

    estimator = xgb.XGBClassifier(
        **plan["parameters"], device=plan["device"], callbacks=[CheckGPU()]
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(
            frame.select(columns).to_numpy(),
            frame["y_ct_win"].to_numpy(),
            sample_weight=round_weights(frame),
        )
    messages = [str(item.message) for item in caught]
    prohibited = ("no visible gpu", "not compiled with cuda", "setting device to cpu")
    if any(any(message.lower().find(word) >= 0 for word in prohibited) for message in messages):
        raise PipelineError(f"检测到 CUDA 回退警告，拒绝报告 GPU 训练成功：{messages}")
    booster = estimator.get_booster()
    device = json.loads(booster.save_config())["learner"]["generic_param"]["device"]
    if device != plan["device"] or not recorded:
        raise PipelineError(f"无法独立确认 GPU 执行：booster={device}，进程记录={recorded}")
    return booster, {
        "fit_partition": counts(frame),
        "device": device,
        "boosting_rounds": booster.num_boosted_rounds(),
        "elapsed_seconds": time.perf_counter() - started,
        "compute_app_observations": recorded,
        "warnings": messages,
    }


def predict(booster: Any, frame: pl.DataFrame) -> np.ndarray:
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(frame.select(feature_columns(1)).to_numpy())).astype(float)


def run(dataset: Path, out: Path, reporter: ProgressReporter | None = None) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise PipelineError("缺少 XGBoost；请安装 gpu 依赖。") from exc
    if out.exists():
        raise PipelineError(f"输出目录已存在：{out}。")
    plan = json.loads(GPU_PLAN.read_text("utf-8"))
    comparison_plan = json.loads((ROOT / plan["comparison_plan"]).read_text("utf-8"))
    if dataset.resolve() != (ROOT / plan["training_dataset"]).resolve():
        raise PipelineError("GPU 实验仅允许使用固定计划中的训练数据。")
    gpu_rows = nvidia_csv("index,name,uuid,driver_version,memory.total")
    if len(gpu_rows) != 1 or len(gpu_rows[0]) < 5 or gpu_rows[0][0] != "0":
        raise PipelineError("无法唯一确认 cuda:0 的物理 GPU；拒绝猜测设备。")
    gpu = dict(
        zip(("index", "name", "uuid", "driver", "memory_total_mib"), gpu_rows[0], strict=True)
    )
    emit(reporter, "load", "active", "校验固定数据划分、历史预测及 CUDA 设备。")
    _, old = load_v1(dataset)
    test_meta, test = load_v1(ROOT / plan["exploratory_dataset"])
    if test_meta["split_policy"]["method"] != "holdout_only":
        raise PipelineError("50 场数据必须保持 holdout_only。")
    check_split(old, test, comparison_plan)
    reference = ROOT / plan["reference_predictions"]
    previous = json.loads((reference / "report.json").read_text("utf-8"))
    if (
        previous["dataset_sha256"] != sha256_file(dataset / "dataset.json")
        or previous["plan_sha256"] != sha256_file(ROOT / plan["comparison_plan"])
        or previous["exploratory_dataset_sha256"]
        != sha256_file(ROOT / plan["exploratory_dataset"] / "dataset.json")
        or previous["artifact_sha256"]["predictions.parquet"]
        != sha256_file(reference / "predictions.parquet")
    ):
        raise PipelineError("参考预测与固定数据的哈希绑定不一致。")
    ref = pl.read_parquet(reference / "predictions.parquet").sort("sample_id")
    if not ref["sample_id"].equals(test["sample_id"]):
        raise PipelineError("GPU 结果必须与 CPU 基线使用完全相同的时点。")

    def part(key: str) -> pl.DataFrame:
        return old.filter(pl.col("event").is_in(comparison_plan[key]))

    fit18, cal, validation, full34 = (
        part(key)
        for key in (
            "candidate_fit_events",
            "candidate_calibration_events",
            "candidate_validation_events",
            "full_baseline_fit_events",
        )
    )
    emit(reporter, "load", "completed", f"已确认 {gpu['name']}；开始严格 CUDA 训练。")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gpu-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_json(stage / "plan.json", plan)
        gpu18, proof18 = fit_cuda(fit18, "18 场", plan, gpu, reporter)
        gpu34, proof34 = fit_cuda(full34, "34 场", plan, gpu, reporter)
        for name, booster in (("fit18", gpu18), ("full34", gpu34)):
            file = stage / f"xgboost_{name}.json"
            booster.save_model(file)
            restored = xgb.Booster()
            restored.load_model(file)
            restored.set_param({"device": plan["device"]})
            if not np.allclose(
                predict(booster, validation), predict(restored, validation), rtol=0, atol=1e-7
            ):
                raise PipelineError(f"GPU 模型 {name} 保存后重读的概率不一致。")
        emit(reporter, "fit", "completed", "两个模型均完成 CUDA 训练且 GPU 进程核对通过。")
        calibrator = fit_sigmoid(predict(gpu18, cal), cal)
        write_json(stage / "calibrator.json", calibrator)
        emit(
            reporter, "evaluate", "active", "在独立 Porto 与已观察 50 场上计算概率与比赛聚类区间。"
        )
        val18 = predict(gpu18, validation)
        test18 = predict(gpu18, test)
        test34 = predict(gpu34, test)
        probs = {
            "gpu_fit18": test18,
            "gpu_fit18_calibrated": apply_sigmoid(test18, calibrator),
            "gpu_full34": test34,
        }
        model_metrics = {name: metrics(test, p.tolist()) for name, p in probs.items()}
        val_metrics = {
            "gpu_fit18": metrics(validation, val18.tolist()),
            "gpu_fit18_calibrated": metrics(validation, apply_sigmoid(val18, calibrator).tolist()),
        }
        paired = {}
        for left, right, left_p, right_p in (
            ("frozen_v1", "gpu_full34", ref["frozen_v1"].to_numpy(), test34),
            ("v1_fit18", "gpu_fit18", ref["v1_fit18"].to_numpy(), test18),
            ("v1_tree_fit18", "gpu_fit18", ref["v1_tree_fit18"].to_numpy(), test18),
            ("gpu_fit18", "gpu_fit18_calibrated", test18, probs["gpu_fit18_calibrated"]),
        ):
            paired[f"{right}_minus_{left}"] = cluster_bootstrap(
                test, left_p.tolist(), right_p.tolist(), draws=2000, seed=42
            )
        predictions = test.select("sample_id").with_columns(
            [pl.Series(name, p, dtype=pl.Float64) for name, p in probs.items()]
        )
        predictions.write_parquet(stage / "predictions.parquet", compression="zstd")
        if not pl.read_parquet(stage / "predictions.parquet").equals(predictions):
            raise PipelineError("GPU 预测结果重读不一致。")
        report = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "gpu_plan_sha256": sha256_file(GPU_PLAN),
            "comparison_plan_sha256": sha256_file(ROOT / plan["comparison_plan"]),
            "dataset_sha256": sha256_file(dataset / "dataset.json"),
            "exploratory_dataset_sha256": sha256_file(
                ROOT / plan["exploratory_dataset"] / "dataset.json"
            ),
            "reference_report_sha256": sha256_file(reference / "report.json"),
            "reference_predictions_sha256": sha256_file(reference / "predictions.parquet"),
            "code_sha256": sha256_file(Path(__file__)),
            "versions": {
                name: version(name) for name in ("xgboost", "numpy", "polars", "scikit-learn")
            },
            "gpu": gpu,
            "xgboost_build_info": xgb.build_info(),
            "training": {"fit18": proof18, "full34": proof34},
            "calibration": calibrator,
            "results": {"validation": val_metrics, "exploration": model_metrics},
            "paired_match_bootstrap": paired,
            "metrics": {
                "validation": {"model": val_metrics["gpu_fit18"]},
                "test": {"model": model_metrics["gpu_full34"]},
            },
            "run_mode": "observed_exploration_not_blind_test",
            "artifact_sha256": {
                name: sha256_file(stage / name)
                for name in (
                    "plan.json",
                    "xgboost_fit18.json",
                    "xgboost_full34.json",
                    "calibrator.json",
                    "predictions.parquet",
                )
            },
            "interpretation": "Both models ran on the verified CUDA GPU. "
            "The 50 observed matches are exploratory, not a blind test.",
        }
        emit(
            reporter,
            "evaluate",
            "completed",
            "GPU 候选模型已完成同样本评分。",
            split="validation",
            metrics=val_metrics["gpu_fit18"],
        )
        emit(reporter, "save", "active", "保存 GPU 模型、逐时点概率及设备证据。")
        write_json(stage / "report.json", report)
        stage.rename(out)
    emit(reporter, "save", "completed", "GPU 实验产物已保存并验证。")
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
        emit(reporter, "run", "completed", "CUDA 模型对照完成。")
        return 0
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pl.exceptions.PolarsError,
    ) as exc:
        emit(reporter, "run", "failed", f"CUDA 模型对照失败：{exc}")
        print(f"CUDA 模型对照失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        emit(reporter, "run", "failed", f"CUDA 模型对照失败（{type(exc).__name__}）：{exc}")
        print(f"CUDA 模型对照失败（{type(exc).__name__}）：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

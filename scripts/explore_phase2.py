"""Explore frozen-model errors and paired candidate scores on the observed 50 maps."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_data import ROOT, context_for_samples, sources
from cs2_pipeline.schema import PipelineError

DATASET = ROOT / "data/datasets/mirage_holdout_50_v1"


def evaluate(frame: pl.DataFrame, column: str = "frozen_v1") -> dict[str, Any]:
    result = metrics(frame, frame[column].to_list())
    result.update(
        matches=frame["match_id"].n_unique(),
        rounds=frame.select("demo_sha256", "round_id").n_unique(),
    )
    if result["matches"] >= 2:
        interval = cluster_bootstrap(
            frame, frame[column].to_list(), frame[column].to_list(), draws=2000, seed=42
        )
        result["match_bootstrap_95"] = interval["percentile_95_intervals"]["base"]
    return result


def group_evaluations(frame: pl.DataFrame, column: str) -> dict[str, Any]:
    output = {}
    for (key,), subset in frame.partition_by(column, as_dict=True, maintain_order=True).items():
        output[str(key)] = evaluate(subset)
    return output


def paired_times(frame: pl.DataFrame, early: float = 0, late: float = 60) -> dict[str, Any]:
    keys = ["demo_sha256", "round_id"]
    first = frame.filter(pl.col("elapsed_since_freeze_end_seconds") == early).select(
        *keys, "match_id", "y_ct_win", pl.col("frozen_v1").alias("early_p")
    )
    second = frame.filter(pl.col("elapsed_since_freeze_end_seconds") == late).select(
        *keys, pl.col("y_ct_win").alias("late_y"), pl.col("frozen_v1").alias("late_p")
    )
    paired = first.join(second, on=keys, how="inner", validate="1:1").sort(keys)
    if paired.is_empty() or paired.filter(pl.col("y_ct_win") != pl.col("late_y")).height:
        raise PipelineError("早晚时点无法按同一回合正确配对。")
    result = {
        "early_seconds": early,
        "late_seconds": late,
        "condition": "Only rounds observed at both ticks; conditional on lasting 60 seconds.",
        "early": evaluate(paired, "early_p"),
        "late": evaluate(paired, "late_p"),
        "late_minus_early_match_bootstrap": cluster_bootstrap(
            paired, paired["early_p"].to_list(), paired["late_p"].to_list(), draws=2000, seed=42
        ),
    }
    return result


def run(model_out: Path) -> dict[str, Any]:
    meta, tables = read_dataset(DATASET)
    if meta["split_policy"]["method"] != "holdout_only":
        raise PipelineError("探索数据不再是仅评估数据集。")
    report = json.loads((model_out / "report.json").read_text("utf-8"))
    if report["exploratory_dataset_sha256"] != sha256_file(DATASET / "dataset.json"):
        raise PipelineError("模型对照与探索数据集的哈希不一致。")
    samples = tables["samples"]
    context, quality = context_for_samples(ROOT, DATASET.name, sources(ROOT), samples)
    predictions = pl.read_parquet(model_out / "predictions.parquet")
    frame = (
        context.join(tables["labels"], on="sample_id", validate="1:1")
        .join(
            tables["features"].select("sample_id", "ct_minus_t_alive_count"),
            on="sample_id",
            validate="1:1",
        )
        .join(predictions, on="sample_id", validate="1:1")
        .sort("sample_id")
        .with_columns(
            pl.when(pl.col("ct_minus_t_alive_count") <= -3)
            .then(pl.lit("CT少3人及以上"))
            .when(pl.col("ct_minus_t_alive_count") >= 3)
            .then(pl.lit("CT多3人及以上"))
            .otherwise(pl.col("ct_minus_t_alive_count").cast(pl.Int64).cast(pl.String))
            .alias("alive_difference"),
            pl.when(pl.col("elapsed_since_freeze_end_seconds") < 15)
            .then(pl.lit("00–15秒"))
            .when(pl.col("elapsed_since_freeze_end_seconds") < 30)
            .then(pl.lit("15–30秒"))
            .when(pl.col("elapsed_since_freeze_end_seconds") < 45)
            .then(pl.lit("30–45秒"))
            .when(pl.col("elapsed_since_freeze_end_seconds") < 60)
            .then(pl.lit("45–60秒"))
            .otherwise(pl.lit("60秒及以后"))
            .alias("time_band"),
            pl.when(pl.col("bomb_planted") == 1)
            .then(pl.lit("已下包"))
            .otherwise(pl.lit("未下包"))
            .alias("bomb_state"),
        )
    )
    if frame.height != meta["sample_count"] or frame["frozen_v1"].null_count():
        raise PipelineError("预测与固定探索集的时点未完整匹配。")
    original = pl.read_parquet(ROOT / "models/evaluations/holdout_50_v1/predictions.parquet")
    if not original.sort("sample_id").equals(
        frame.select("sample_id", pl.col("frozen_v1").alias("p_ct_win")).sort("sample_id")
    ):
        raise PipelineError("本次冻结 V1 推理与原始预测不一致。")
    # Only directly observed game-rule state is used in the bomb-stratified audit.
    reliable = frame.filter(~pl.col("match_id").is_in(quality["event_only_matches"]))
    comparisons = {}
    for first, second in (
        ("frozen_v1", "simple_full34"),
        ("v1_fit18", "v2_fit18"),
        ("v1_fit18", "v1_tree_fit18"),
        ("v1_fit18", "v1_fit18_calibrated"),
        ("v2_fit18", "v2_fit18_calibrated"),
        ("v1_tree_fit18", "v1_tree_fit18_calibrated"),
    ):
        comparisons[f"{second}_minus_{first}"] = cluster_bootstrap(
            frame, frame[first].to_list(), frame[second].to_list(), draws=2000, seed=42
        )
    columns = [c for c in predictions.columns if c != "sample_id"]
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_report_sha256": sha256_file(model_out / "report.json"),
        "predictions_sha256": sha256_file(model_out / "predictions.parquet"),
        "dataset_sha256": sha256_file(DATASET / "dataset.json"),
        "interpretation": [
            "The 50 matches have been observed; all results are exploratory, not a blind test.",
            "Weights sum to one per round within each subgroup; subgroups overlap in rounds.",
            "Match bootstrap respects shared match/round/tick dependence, but cannot estimate "
            "variation between future events from these five selected tournaments.",
            "Paired time comparison is conditional on rounds surviving to 60 seconds and "
            "does not measure a causal effect of time.",
        ],
        "overall_frozen": evaluate(frame),
        "by_event": group_evaluations(frame, "event"),
        "by_alive_difference": group_evaluations(frame, "alive_difference"),
        "by_elapsed": group_evaluations(frame, "time_band"),
        "paired_0_vs_60_seconds": paired_times(frame),
        "bomb_analysis": {
            "excluded_event_only_matches": quality["event_only_matches"],
            "basis": "Observed game-rules bomb status; event-only files excluded.",
            "groups": group_evaluations(reliable, "bomb_state"),
        },
        "same_reliable_47_maps_all_models": {name: evaluate(reliable, name) for name in columns},
        "paired_candidate_differences": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/phase2_exploration.json")
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise PipelineError(f"报告已存在：{args.out}；请使用新路径。")
        result = run(args.model_out)
        write_json(args.out, result)
        print(f"分组与配对分析已保存：{args.out}")
        return 0
    except (PipelineError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        print(f"探索分析失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit fixed baseline comparisons with identical labels and chronological splits."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import probability_metrics, round_weights


def compare_runs(
    baseline_dataset: Path,
    baseline_model: Path,
    candidate_dataset: Path,
    candidate_model: Path,
) -> dict[str, Any]:
    metadata1, tables1 = read_dataset(baseline_dataset)
    metadata2, tables2 = read_dataset(candidate_dataset)
    for name in ("labels", "splits"):
        if not tables1[name].sort("sample_id").equals(tables2[name].sort("sample_id")):
            raise PipelineError(f"两个实验的 {name} 不一致，不能比较分数。")
    keys = ["sample_id", "demo_sha256", "round_id", "tick", "match_id", "match_date", "map_name"]
    if (
        not tables1["samples"]
        .select(keys)
        .sort("sample_id")
        .equals(tables2["samples"].select(keys).sort("sample_id"))
    ):
        raise PipelineError("两个实验使用的真实时点或比赛身份不同。")
    reports = []
    recalculated = []
    for dataset, model_path, tables in (
        (baseline_dataset, baseline_model, tables1),
        (candidate_dataset, candidate_model, tables2),
    ):
        report = json.loads((model_path / "report.json").read_text(encoding="utf-8"))
        if not report.get("independent_evaluation_performed"):
            raise PipelineError("流程检查模型没有独立指标，不能纳入正式比较。")
        if report.get("dataset_manifest_sha256") != sha256_file(dataset / "dataset.json"):
            raise PipelineError("模型报告与数据集清单哈希不同，不能复用旧指标。")
        predictions = pl.read_parquet(model_path / "predictions.parquet")
        expected = tables["splits"].filter(pl.col("split").is_in(["validation", "test"]))
        if (
            not predictions.select("sample_id", "split")
            .sort("sample_id")
            .equals(expected.sort("sample_id"))
        ):
            raise PipelineError("预测时点/划分与留出集合不一致。")
        joined = predictions.join(tables["labels"], on="sample_id", validate="1:1").join(
            tables["samples"].select("sample_id", "demo_sha256", "round_id", "match_id"),
            on="sample_id",
            validate="1:1",
        )
        result = {}
        for split in ("validation", "test"):
            frame = joined.filter(pl.col("split") == split).sort("sample_id")
            if frame.filter(
                ~pl.col("p_ct_win").is_between(0, 1) | pl.col("p_ct_win").is_null()
            ).height:
                raise PipelineError("预测概率缺失或不在 0..1。")
            metrics = probability_metrics(
                frame["y_ct_win"].to_list(), frame["p_ct_win"].to_list(), round_weights(frame)
            )
            for name in ("log_loss", "brier_score"):
                if not math.isclose(
                    metrics[name],
                    report["metrics"][split]["model"][name],
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise PipelineError("重新计算的指标与已保存报告不同。")
            result[split] = metrics
        recalculated.append(result)
        reports.append(report)
    # The held-out test score never enters the selection rule.
    selected = (
        "candidate"
        if recalculated[1]["validation"]["log_loss"] < recalculated[0]["validation"]["log_loss"]
        else "baseline"
    )
    return {
        "comparison_schema_version": 1,
        "selection_rule": "lower validation round-weighted log loss; baseline wins ties",
        "selected_by_validation": selected,
        "baseline": {
            "model": baseline_model.as_posix(),
            "feature_count": reports[0]["feature_count"],
            "metrics": recalculated[0],
        },
        "candidate": {
            "model": candidate_model.as_posix(),
            "feature_count": reports[1]["feature_count"],
            "metrics": recalculated[1],
        },
        "split_policy": metadata1["split_policy"],
        "round_count": metadata1["round_count"],
        "sample_count": metadata1["sample_count"],
        "node_count": metadata1["node_count"],
        "demo_sha256": sorted(s["demo_sha256"] for s in metadata1["sources"]),
        "context_non_null_counts": {
            c: metadata2["sample_count"] - tables2["samples"][c].null_count()
            for c in (
                "round_elapsed_game_seconds",
                "bomb_seconds_since_plant",
                "bomb_time_remaining_seconds",
            )
            if c in tables2["samples"].columns
        },
        "limitations": [
            "Pilot within one MVP event; validation and test each contain one match.",
            "Fixed candidates; no tuning, probability calibration, rating or Elo.",
            "No statistical claim of cross-event or trading reliability.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dataset", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--candidate-dataset", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise PipelineError("比较报告已存在，请使用新的 --out。")
        report = compare_runs(
            args.baseline_dataset, args.baseline_model, args.candidate_dataset, args.candidate_model
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.out, report)
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(f"比较失败：{exc}", file=sys.stderr)
        return 1
    print(f"验证集选择：{report['selected_by_validation']}；已独立重算指标：{args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

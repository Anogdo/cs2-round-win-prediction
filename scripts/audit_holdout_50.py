"""Reread the 50-match artifacts and independently reproduce their scores."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.calibrate import metrics
from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.parse import prepare_rounds, sha256_file
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.recover import ROUND_COLUMNS, recover_rounds, verify_with_awpy
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory
from scripts.evaluate_holdout_50 import DATASET, EVALUATION
from scripts.import_holdout_50 import (
    EXTRACTED,
    PLAN,
    REFERENCE,
    ROOT,
    check_disjoint,
    load_plans,
    published_score,
)


def audit() -> dict[str, Any]:
    plan, reference = load_plans()
    check_disjoint(plan)
    meta, tables = read_dataset(DATASET)
    report = json.loads((EVALUATION / "report.json").read_text(encoding="utf-8"))
    if (
        report["dataset_sha256"] != sha256_file(DATASET / "dataset.json")
        or meta["holdout_plan_sha256"] != sha256_file(PLAN)
        or meta["published_scores_manifest_sha256"] != sha256_file(REFERENCE)
        or report["frozen_model_sha256"] != plan["frozen_model_sha256"]
        or set(tables["splits"]["split"]) != {"test"}
    ):
        raise PipelineError("模型、计划、数据集或评估文件的绑定不一致。")
    warnings: Counter[str] = Counter()
    by_match: dict[str, list[Path]] = defaultdict(list)
    for source in meta["sources"]:
        path = ROOT / source["path"]
        for filename, checksum in source["source_file_sha256"].items():
            if sha256_file(path / filename) != checksum:
                raise PipelineError(f"来源文件哈希不匹配：{path / filename}")
        quality = validate_directory(path)
        if not quality["passed"]:
            raise PipelineError(f"数据质量检查失败：{path}: {quality['errors']}")
        states = pl.read_parquet(path / "snapshots.parquet").filter(pl.col("team").is_not_null())
        team_on_both_sides = (
            states.group_by("tick", "team")
            .agg(pl.col("side").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )
        side_has_multiple_teams = (
            states.group_by("tick", "side")
            .agg(pl.col("team").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
        )
        if team_on_both_sides.height or side_has_multiple_teams.height:
            raise PipelineError(f"同一时点的队伍身份与阵营不一致：{path}")
        for warning in quality["warnings"]:
            warnings[warning["code"]] += warning["count"]
        by_match[source["match_id"]].append(path)
    if set(by_match) != set(reference):
        raise PipelineError("已解析比赛与固定清单不一致。")
    for match_id, paths in by_match.items():
        published_score(paths, reference[match_id]["expected_score"])
        receipt_dir = EXTRACTED / match_id.replace(":", "-")
        receipt = json.loads((receipt_dir / "extraction.json").read_text(encoding="utf-8"))
        for part in receipt["parts"]:
            if sha256_file(receipt_dir / part["entry"]) != part["sha256"]:
                raise PipelineError(f"Demo 文件哈希不匹配：{receipt_dir / part['entry']}")
    recovered: dict[str, int] = {}
    for match_id in ("hltv:2398745", "hltv:2398721"):
        folder = EXTRACTED / match_id.replace(":", "-")
        parts = sorted(folder.glob("*.dem"))
        if len(parts) != 2:
            raise PipelineError(f"分段 Demo 数量异常：{match_id}")
        first, second = recover_rounds(parts[0]), recover_rounds(parts[1])
        verify_with_awpy(parts[1], second)
        first_path = next(
            path
            for path in by_match[match_id]
            if json.loads((path / "metadata.json").read_text(encoding="utf-8")).get(
                "rounds_recovery"
            )
        )
        expected = prepare_rounds(first).select(ROUND_COLUMNS).sort("round_num")
        actual = (
            pl.read_parquet(first_path / "rounds.parquet").select(ROUND_COLUMNS).sort("round_num")
        )
        if not actual.equals(expected):
            raise PipelineError(f"恢复的首段回合与重新读取的游戏规则状态不同：{match_id}")
        recovered[match_id] = second.height
    model = json.loads((ROOT / plan["frozen_model"]).read_text(encoding="utf-8"))
    frame = (
        tables["features"]
        .join(tables["labels"], on="sample_id", validate="1:1")
        .join(
            tables["samples"].select("sample_id", "match_id", "event", "demo_sha256", "round_id"),
            on="sample_id",
            validate="1:1",
        )
        .sort("sample_id")
    )
    prediction = predict_probabilities(model, frame.select("sample_id", *model["feature_columns"]))
    saved = pl.read_parquet(EVALUATION / "predictions.parquet").sort("sample_id")
    if not prediction.sort("sample_id").equals(saved):
        raise PipelineError("重新推理与保存的逐时点概率不一致。")
    recalculated = metrics(frame, prediction["p_ct_win"].to_list())
    for name in ("log_loss", "brier_score", "roc_auc", "ece_10_bins"):
        if abs(recalculated[name] - report["overall"]["model"][name]) > 1e-12:
            raise PipelineError(f"重新计算的 {name} 与报告不一致。")
    for event, expected in report["by_event"].items():
        part = frame.filter(pl.col("event") == event)
        probs = prediction.join(part.select("sample_id"), on="sample_id", how="semi").sort(
            "sample_id"
        )
        calculated = metrics(part, probs["p_ct_win"].to_list())
        if abs(calculated["log_loss"] - expected["model"]["log_loss"]) > 1e-12:
            raise PipelineError(f"赛事结果重新计算不一致：{event}")
    return {
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "passed": True,
        "matches": len(by_match),
        "recordings": len(meta["sources"]),
        "rounds": meta["round_count"],
        "samples": meta["sample_count"],
        "nodes": meta["node_count"],
        "model_sha256": plan["frozen_model_sha256"],
        "dataset_sha256": sha256_file(DATASET / "dataset.json"),
        "predictions_sha256": sha256_file(EVALUATION / "predictions.parquet"),
        "source_score_checks_passed": len(by_match),
        "predictions_reproduced_exactly": True,
        "non_null_team_side_consistency_checked": True,
        "quality_warning_counts": dict(warnings),
        "recovered_companion_rounds_checked": recovered,
        "overall_model": {
            key: recalculated[key] for key in ("log_loss", "brier_score", "roc_auc", "ece_10_bins")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/holdout_50_audit.json")
    args = parser.parse_args()
    try:
        output = args.out
        if output.exists():
            raise PipelineError(f"审计报告已存在，请使用新的 --out：{output}")
        result = audit()
        write_json(output, result)
        print(f"独立审计通过：{result['matches']} 场、{result['samples']} 个时点。")
        return 0
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(f"独立审计失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

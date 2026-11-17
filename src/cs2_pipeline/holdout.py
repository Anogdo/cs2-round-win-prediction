"""Build a test-only event dataset and score a model frozen before receiving it."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import (
    TABLE_NAMES,
    build_map_tables,
    read_dataset,
    source_identity,
    validate_dataset_tables,
    write_json,
)
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import round_weights
from cs2_pipeline.validate import validate_directory

DEFAULT_TRAIN = Path("data/datasets/mirage_expanded_2026_v1")
DEFAULT_PLAN = Path("configs/esl24_holdout_plan.json")
DEFAULT_SCORES = Path("configs/esl24_user_hltv_sources.json")
SCORE_TEAM_ALIASES = {
    "Team Falcons": "Falcons",
    "Team Spirit": "Spirit",
    "Team Vitality": "Vitality",
    "Aurora Gaming": "Aurora",
}


def check_published_score(rounds: pl.DataFrame, snapshots: pl.DataFrame, expected: dict) -> None:
    """Check published scores using team identities; aliases are for this check only."""
    winners = (
        snapshots.sort("tick")
        .group_by("round_id", "team")
        .first()
        .join(rounds.select("round_id", "winner_side"), on="round_id")
        .filter(pl.col("side") == pl.col("winner_side"))
        .group_by("team")
        .len()
    )
    actual: dict[str, int] = {}
    for row in winners.iter_rows(named=True):
        name = SCORE_TEAM_ALIASES.get(row["team"], row["team"])
        if name in actual:
            raise PipelineError(f"比分核验存在重名队伍：{name}")
        actual[name] = row["len"]
    # A team that won zero rounds will not occur in the winners table.
    actual = {name: actual.get(name, 0) for name in expected} | {
        name: count for name, count in actual.items() if name not in expected
    }
    if actual != expected:
        raise PipelineError(f"Demo 比分与公开比分不一致：解析 {actual}，预期 {expected}")


def verify_frozen(plan_path: Path, training_dataset: Path) -> tuple[dict, dict, dict, dict]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    model_path = (plan_path.resolve().parent.parent / plan["frozen_model"]).resolve()
    if sha256_file(model_path) != plan["frozen_model_sha256"]:
        raise PipelineError("冻结模型 SHA256 不符；禁止在验证前改变模型。")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    meta, tables = read_dataset(training_dataset)
    if model.get("dataset_manifest_sha256") != sha256_file(training_dataset / "dataset.json"):
        raise PipelineError("冻结模型与参考训练集不匹配。")
    if model.get("feature_version", 1) != 1 or model.get("map_name") != plan["map_name"]:
        raise PipelineError("模型地图或特征版本与独立赛事不符。")
    train_samples = (
        tables["samples"].join(tables["splits"], on="sample_id").filter(pl.col("split") == "train")
    )
    if train_samples.is_empty() or train_samples["match_date"].max() != plan["training_cutoff"]:
        raise PipelineError("实际训练截止日期与计划不一致。")
    if any(item["match_date"] <= plan["training_cutoff"] for item in plan["matches"]):
        raise PipelineError("独立测试比赛必须晚于模型训练截止日期。")
    return plan, model, meta, tables


def build_holdout(
    manifest: Path, plan_path: Path, training_dataset: Path, out: Path, scores_path: Path
) -> dict:
    if out.exists():
        raise PipelineError("测试数据集已存在；请使用新输出目录。")
    plan, _, _, training = verify_frozen(plan_path, training_dataset)
    config = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {m["match_id"]: m for m in plan["matches"]}
    score_entries = json.loads(scores_path.read_text(encoding="utf-8"))["maps"]
    score_by_match = {m["match_id"]: m for m in score_entries}
    if len(score_entries) != len(expected) or set(score_by_match) != set(expected):
        raise PipelineError("比分清单与冻结计划不符。")
    entries = config["sources"]
    if len(expected) != len(plan["matches"]) or {e["match_id"] for e in entries} != set(expected):
        raise PipelineError("独立赛事比赛清单不完整或重复，与预先计划不一致。")
    if len(entries) != len(expected):
        raise PipelineError("独立赛事包含重复比赛或意外分段，请核验清单。")
    old_ids = set(training["samples"]["match_id"].unique().to_list())
    old_hashes = set(training["samples"]["demo_sha256"].unique().to_list())
    if old_ids & set(expected):
        raise PipelineError("独立赛事包含已用于原数据集的比赛。")
    collected: dict[str, list[pl.DataFrame]] = {k: [] for k in TABLE_NAMES[:-1]}
    sources, hashes = [], set()
    for entry in entries:
        match = expected[entry["match_id"]]
        score_entry = score_by_match[entry["match_id"]]
        if any(
            score_entry.get(key) != match.get(key)
            for key in ("match_date", "map_index", "expected_rounds")
        ) or score_entry.get("match_url") != match.get("url"):
            raise PipelineError(f"比分来源与冻结计划不符：{entry['match_id']}")
        if entry["event"] != plan["event"] or entry["match_date"] != match["match_date"]:
            raise PipelineError(f"赛事或日期与冻结计划不符：{entry['match_id']}")
        if entry.get("map_id") != f"{entry['match_id']}:m{match['map_index']}":
            raise PipelineError(f"地图序号与冻结计划不符：{entry['match_id']}")
        path = (manifest.resolve().parent / entry["path"]).resolve()
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        checksum = metadata["demo_sha256"]
        if checksum in old_hashes or checksum in hashes:
            raise PipelineError("测试 Demo 与既有来源重复。")
        hashes.add(checksum)
        if (
            metadata.get("parser_version") != "3.3.0"
            or metadata.get("map_name") != plan["map_name"]
        ):
            raise PipelineError("独立赛事必须使用同一版本的 Awpy 解析同一地图。")
        quality = validate_directory(path)
        if not quality["passed"] or quality["round_count"] != match["expected_rounds"]:
            raise PipelineError(f"比赛质量或回合数不符：{entry['match_id']}")
        rounds = pl.read_parquet(path / "rounds.parquet")
        snapshots = pl.read_parquet(path / "snapshots.parquet")
        check_published_score(rounds, snapshots, score_entry["expected_score"])
        tables, counts = build_map_tables(
            metadata,
            rounds,
            snapshots,
            source_identity(entry, metadata),
            feature_version=1,
        )
        for name, table in tables.items():
            collected[name].append(table)
        sources.append(
            {
                "match_id": entry["match_id"],
                "match_date": entry["match_date"],
                "event": entry["event"],
                "demo_sha256": checksum,
                "path": entry["path"],
                "published_score_checked": True,
                **counts,
                "source_file_sha256": {
                    name: sha256_file(path / name)
                    for name in (
                        "metadata.json",
                        "rounds.parquet",
                        "players.parquet",
                        "snapshots.parquet",
                    )
                },
            }
        )
    tables = {name: pl.concat(frames).sort("sample_id") for name, frames in collected.items()}
    tables["splits"] = tables["samples"].select("sample_id", pl.lit("test").alias("split"))
    meta: dict[str, Any] = {
        "dataset_schema_version": 1,
        "feature_version": 1,
        "map_name": plan["map_name"],
        "feature_columns": feature_columns(1),
        "label": "y_ct_win; separate labels.parquet",
        "source_manifest_sha256": sha256_file(manifest),
        "published_scores_manifest_sha256": sha256_file(scores_path),
        "holdout_plan_sha256": sha256_file(plan_path),
        "frozen_model_sha256": plan["frozen_model_sha256"],
        "reference_dataset_sha256": sha256_file(training_dataset / "dataset.json"),
        "split_policy": {
            "group_by": "event",
            "method": "holdout_only",
            "evaluation_ready": False,
            "reason": "仅供冻结模型评估，禁止用于训练。",
        },
        "sources": sources,
        "sample_count": tables["samples"].height,
        "node_count": tables["nodes"].height,
        "round_count": tables["samples"].select("demo_sha256", "round_id").unique().height,
    }
    validate_dataset_tables(tables, meta)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".holdout-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        for name, table in tables.items():
            file = stage / f"{name}.parquet"
            table.write_parquet(file, compression="zstd")
            if (
                not pl.read_parquet(file).equals(table)
                or pq.read_table(file).num_rows != table.height
            ):
                raise PipelineError(f"{name} Parquet 重读不一致。")
        meta["file_sha256"] = {
            f"{name}.parquet": sha256_file(stage / f"{name}.parquet") for name in TABLE_NAMES
        }
        write_json(stage / "dataset.json", meta)
        stage.rename(out)
    return meta


def score_holdout(
    dataset: Path, plan_path: Path, training_dataset: Path, out: Path, scores_path: Path
) -> dict:
    if out.exists():
        raise PipelineError("评估目录已存在；请使用新目录。")
    plan, model, _, old_tables = verify_frozen(plan_path, training_dataset)
    metadata, tables = read_dataset(dataset)
    if (
        metadata.get("holdout_plan_sha256") != sha256_file(plan_path)
        or metadata.get("frozen_model_sha256") != plan["frozen_model_sha256"]
        or metadata.get("reference_dataset_sha256")
        != sha256_file(training_dataset / "dataset.json")
        or metadata.get("published_scores_manifest_sha256") != sha256_file(scores_path)
        or metadata["split_policy"].get("method") != "holdout_only"
        or metadata["split_policy"].get("evaluation_ready")
    ):
        raise PipelineError("测试集与冻结计划不符，或并非只读留出数据集。")
    if set(tables["splits"]["split"]) != {"test"}:
        raise PipelineError("独立赛事中的所有样本必须为 test。")
    expected = {m["match_id"]: m["match_date"] for m in plan["matches"]}
    actual = tables["samples"].select("match_id", "match_date", "event").unique()
    if (
        actual.height != len(expected)
        or {r["match_id"]: r["match_date"] for r in actual.iter_rows(named=True)} != expected
        or set(actual["event"]) != {plan["event"]}
        or set(tables["samples"]["demo_sha256"]) & set(old_tables["samples"]["demo_sha256"])
        or set(tables["samples"]["match_id"]) & set(old_tables["samples"]["match_id"])
    ):
        raise PipelineError("独立赛事比赛、日期、赛事身份或 Demo 与冻结计划不符。")
    frame = (
        tables["features"]
        .join(tables["labels"], on="sample_id", validate="1:1")
        .join(
            tables["samples"].select("sample_id", "demo_sha256", "round_id", "match_id"),
            on="sample_id",
            validate="1:1",
        )
        .sort("sample_id")
    )
    prediction = predict_probabilities(model, frame.select("sample_id", *model["feature_columns"]))
    train = (
        old_tables["samples"]
        .join(old_tables["splits"], on="sample_id")
        .filter(pl.col("split") == "train")
        .join(old_tables["labels"], on="sample_id")
    )
    weights = round_weights(train)
    prior = sum(y * w for y, w in zip(train["y_ct_win"], weights, strict=True)) / sum(weights)
    probabilities = prediction["p_ct_win"].to_list()
    per_match = []
    for match_id in sorted(frame["match_id"].unique().to_list()):
        part = frame.filter(pl.col("match_id") == match_id)
        p = prediction.join(part.select("sample_id"), on="sample_id", how="semi").sort("sample_id")
        per_match.append(
            {
                "match_id": match_id,
                "rounds": part["round_id"].n_unique(),
                "model": metrics(part, p["p_ct_win"].to_list()),
            }
        )
    uncertainty = cluster_bootstrap(
        frame, [prior] * frame.height, probabilities, draws=1000, seed=42
    )
    intervals = uncertainty["percentile_95_intervals"]
    uncertainty["percentile_95_intervals"] = {
        "constant_training_prior": intervals["base"],
        "model": intervals["calibrated"],
        "model_minus_constant_training_prior": intervals["calibrated_minus_base"],
    }
    report = {
        "event": plan["event"],
        "frozen_model_sha256": plan["frozen_model_sha256"],
        "holdout_plan_sha256": sha256_file(plan_path),
        "dataset_sha256": sha256_file(dataset / "dataset.json"),
        "matches": frame["match_id"].n_unique(),
        "rounds": frame.select("demo_sha256", "round_id").unique().height,
        "samples": frame.height,
        "model": metrics(frame, probabilities),
        "constant_training_prior": metrics(frame, [prior] * frame.height),
        "training_prior": prior,
        "per_match": per_match,
        "paired_match_bootstrap": uncertainty,
        "limitations": [
            "Selected six matches from one event, not the complete event.",
            "One additional event cannot establish long-term calibration.",
            "Model was fixed before this event was requested; no fitting here.",
            "Earlier test results had already been observed before selecting this event.",
            "Matches precede Beijing validation; this is a retrospective time holdout.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluate-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        prediction.write_parquet(stage / "predictions.parquet", compression="zstd")
        if not pl.read_parquet(stage / "predictions.parquet").equals(prediction):
            raise PipelineError("预测文件往返读取失败。")
        write_json(stage / "report.json", report)
        stage.rename(out)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "score"])
    parser.add_argument("--manifest", type=Path, default=Path("configs/esl24_holdout_sources.json"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--training-dataset", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/esl24_holdout_v1"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "build":
            report = build_holdout(
                args.manifest,
                args.plan,
                args.training_dataset,
                args.out or args.dataset,
                args.scores,
            )
            print(f"独立赛事数据：{report['round_count']} 回合，{report['sample_count']} 时点。")
        else:
            report = score_holdout(
                args.dataset,
                args.plan,
                args.training_dataset,
                args.out or Path("models/evaluations/esl24_v1"),
                args.scores,
            )
            print(
                f"ESL 独立赛事：模型 LogLoss={report['model']['log_loss']:.6f}，"
                f"Brier={report['model']['brier_score']:.6f}；训练先验 "
                f"LogLoss={report['constant_training_prior']['log_loss']:.6f}"
            )
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(f"独立赛事评估失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

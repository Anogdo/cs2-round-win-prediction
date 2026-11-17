"""Build and score the fixed-model, 50-match retrospective holdout."""

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
from scripts.import_holdout_50 import (
    PLAN,
    PRIOR_ESL,
    PROCESSED,
    REFERENCE,
    ROOT,
    TRAINING,
    check_disjoint,
    load_plans,
    published_score,
)

DATASET = ROOT / "data/datasets/mirage_holdout_50_v1"
EVALUATION = ROOT / "models/evaluations/holdout_50_v1"


def build_dataset(out: Path) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"测试数据集已存在：{out}；请使用新目录。")
    plan, references = load_plans()
    check_disjoint(plan)
    old_hashes: set[str] = set()
    for old in (TRAINING, PRIOR_ESL):
        _, prior = read_dataset(old)
        old_hashes.update(prior["samples"]["demo_sha256"].unique().to_list())
    seen: set[str] = set()
    collected: dict[str, list[pl.DataFrame]] = {name: [] for name in TABLE_NAMES[:-1]}
    source_rows: list[dict[str, Any]] = []
    for item in plan["matches"]:
        ref = references[item["match_id"]]
        folder = PROCESSED / item["match_id"].replace(":", "-")
        verified = json.loads((folder / "checked.json").read_text(encoding="utf-8"))
        manifest = folder / "sources.json"
        entries = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
        if (
            verified["match_id"] != item["match_id"]
            or verified["event"] != item["event"]
            or verified["match_date"] != ref["match_date"]
            or verified["published_score_verified"] != ref["expected_score"]
            or len(entries) != verified["part_count"]
        ):
            raise PipelineError(f"测试比赛核验记录不符：{item['match_id']}")
        paths = [(manifest.parent / entry["path"]).resolve() for entry in entries]
        published_score(paths, ref["expected_score"])
        for entry, path in zip(entries, paths, strict=True):
            meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
            checksum = meta["demo_sha256"]
            if (
                entry["match_id"] != item["match_id"]
                or entry["event"] != item["event"]
                or entry["match_date"] != ref["match_date"]
                or entry["map_id"] != f"{item['match_id']}:m{item['map_index']}"
                or checksum not in verified["demo_sha256"]
                or checksum in seen
                or checksum in old_hashes
                or meta.get("parser_version") != "3.3.0"
                or meta.get("map_name") != "de_mirage"
                or not validate_directory(path)["passed"]
            ):
                raise PipelineError(f"数据来源、解析版本或重复校验失败：{path}")
            seen.add(checksum)
            tables, counts = build_map_tables(
                meta,
                pl.read_parquet(path / "rounds.parquet"),
                pl.read_parquet(path / "snapshots.parquet"),
                source_identity(entry, meta),
                feature_version=1,
            )
            for name, table in tables.items():
                collected[name].append(table)
            source_rows.append(
                {
                    **entry,
                    "path": path.relative_to(ROOT).as_posix(),
                    "demo_sha256": checksum,
                    "archive_sha256": verified["archive_sha256"],
                    "published_score_verified": True,
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
    expected_rounds = sum(ref["expected_rounds"] for ref in references.values())
    actual_rounds = tables["samples"].select("demo_sha256", "round_id").unique().height
    metadata: dict[str, Any] = {
        "dataset_schema_version": 1,
        "feature_version": 1,
        "map_name": plan["map_name"],
        "feature_columns": feature_columns(1),
        "label": "y_ct_win; separate labels.parquet",
        "source_manifest_sha256": sha256_file(PLAN),
        "holdout_plan_sha256": sha256_file(PLAN),
        "published_scores_manifest_sha256": sha256_file(REFERENCE),
        "reference_dataset_sha256": sha256_file(TRAINING / "dataset.json"),
        "frozen_model_sha256": plan["frozen_model_sha256"],
        "split_policy": {
            "group_by": "event",
            "method": "holdout_only",
            "evaluation_ready": False,
            "reason": "仅供固定模型回顾性评估，禁止用于训练和校准。",
        },
        "sources": source_rows,
        "match_count": len(plan["matches"]),
        "round_count": actual_rounds,
        "expected_round_count": expected_rounds,
        "sample_count": tables["samples"].height,
        "node_count": tables["nodes"].height,
    }
    if actual_rounds != expected_rounds:
        raise PipelineError(f"可建模回合 {actual_rounds} 与已核对回合 {expected_rounds} 不同。")
    validate_dataset_tables(tables, metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".holdout50-stage-", dir=out.parent) as temporary:
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
        metadata["file_sha256"] = {
            f"{name}.parquet": sha256_file(stage / f"{name}.parquet") for name in TABLE_NAMES
        }
        write_json(stage / "dataset.json", metadata)
        stage.rename(out)
    return metadata


def score_dataset(dataset: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"评估目录已存在：{out}；请使用新目录。")
    plan, references = load_plans()
    check_disjoint(plan)
    metadata, tables = read_dataset(dataset)
    if (
        metadata.get("holdout_plan_sha256") != sha256_file(PLAN)
        or metadata.get("published_scores_manifest_sha256") != sha256_file(REFERENCE)
        or metadata.get("frozen_model_sha256") != plan["frozen_model_sha256"]
        or metadata.get("reference_dataset_sha256") != sha256_file(TRAINING / "dataset.json")
        or metadata.get("split_policy", {}).get("method") != "holdout_only"
        or set(tables["splits"]["split"]) != {"test"}
    ):
        raise PipelineError("测试集与固定模型、计划或数据范围不符。")
    expected = {
        m["match_id"]: (m["event"], references[m["match_id"]]["match_date"])
        for m in plan["matches"]
    }
    actual = tables["samples"].select("match_id", "event", "match_date").unique()
    if (
        actual.height != len(expected)
        or {
            row["match_id"]: (row["event"], row["match_date"])
            for row in actual.iter_rows(named=True)
        }
        != expected
    ):
        raise PipelineError("测试数据集比赛、日期或赛事不完整。")
    model = json.loads((ROOT / plan["frozen_model"]).read_text(encoding="utf-8"))
    frame = (
        tables["features"]
        .join(tables["labels"], on="sample_id", validate="1:1")
        .join(
            tables["samples"].select(
                "sample_id",
                "demo_sha256",
                "round_id",
                "match_id",
                "event",
                "elapsed_since_freeze_end_seconds",
            ),
            on="sample_id",
            validate="1:1",
        )
        .sort("sample_id")
    )
    prediction = predict_probabilities(model, frame.select("sample_id", *model["feature_columns"]))
    frame = frame.join(prediction, on="sample_id", validate="1:1").sort("sample_id")
    _, old = read_dataset(TRAINING)
    train = (
        old["samples"]
        .join(old["splits"], on="sample_id")
        .filter(pl.col("split") == "train")
        .join(old["labels"], on="sample_id")
    )
    weights = round_weights(train)
    prior = sum(y * w for y, w in zip(train["y_ct_win"], weights, strict=True)) / sum(weights)

    def summary(part: pl.DataFrame) -> dict[str, Any]:
        probs = part["p_ct_win"].to_list()
        return {
            "matches": part["match_id"].n_unique(),
            "rounds": part.select("demo_sha256", "round_id").unique().height,
            "samples": part.height,
            "model": metrics(part, probs),
            "constant_training_prior": metrics(part, [prior] * part.height),
        }

    uncertainty = cluster_bootstrap(frame, [prior] * frame.height, frame["p_ct_win"].to_list())
    intervals = uncertainty["percentile_95_intervals"]
    uncertainty["percentile_95_intervals"] = {
        "constant_training_prior": intervals["base"],
        "model": intervals["calibrated"],
        "model_minus_constant_training_prior": intervals["calibrated_minus_base"],
    }
    new_ids = [
        item["match_id"] for item in plan["matches"] if item["selection_group"] == "new_event"
    ]
    report: dict[str, Any] = {
        "event": "50-match retrospective holdout",
        "frozen_model_sha256": plan["frozen_model_sha256"],
        "holdout_plan_sha256": sha256_file(PLAN),
        "dataset_sha256": sha256_file(dataset / "dataset.json"),
        "training_prior": prior,
        "overall": summary(frame),
        "new_events": summary(frame.filter(pl.col("match_id").is_in(new_ids))),
        "previously_observed_events_new_matches": summary(
            frame.filter(~pl.col("match_id").is_in(new_ids))
        ),
        "by_event": {
            event: summary(frame.filter(pl.col("event") == event))
            for event in sorted(frame["event"].unique().to_list())
        },
        "by_match": {
            match: summary(frame.filter(pl.col("match_id") == match))
            for match in sorted(frame["match_id"].unique().to_list())
        },
        "by_elapsed_seconds": {
            name: summary(frame.filter(condition))
            for name, condition in (
                ("0_to_15", pl.col("elapsed_since_freeze_end_seconds") < 15),
                (
                    "15_to_30",
                    (pl.col("elapsed_since_freeze_end_seconds") >= 15)
                    & (pl.col("elapsed_since_freeze_end_seconds") < 30),
                ),
                (
                    "30_to_60",
                    (pl.col("elapsed_since_freeze_end_seconds") >= 30)
                    & (pl.col("elapsed_since_freeze_end_seconds") < 60),
                ),
                ("60_plus", pl.col("elapsed_since_freeze_end_seconds") >= 60),
            )
            if frame.filter(condition).height
        },
        "paired_match_bootstrap": uncertainty,
        "limitations": [
            "Selection occurred after previous evaluations were observed; not prospectively blind.",
            "Three events are new; other matches are from events with previously observed results.",
            "Matches follow the training cutoff but some precede Beijing model validation.",
            "Model is uncalibrated; match bootstrap does not measure between-event uncertainty.",
            "Only selected Mirage matches; no claim about full-event or other-map performance.",
            "Time-window metrics normalize each represented round within that window separately.",
        ],
    }
    # The monitor's history reads a top-level model score for evaluation records.
    report["model"] = report["overall"]["model"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".score50-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        prediction.write_parquet(stage / "predictions.parquet", compression="zstd")
        if not pl.read_parquet(stage / "predictions.parquet").equals(prediction):
            raise PipelineError("预测文件往返读取失败。")
        write_json(stage / "report.json", report)
        stage.rename(out)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "score"])
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "build":
            result = build_dataset(args.out or args.dataset)
            print(f"测试集：{result['match_count']} 场、{result['round_count']} 回合。")
        else:
            result = score_dataset(args.dataset, args.out or EVALUATION)
            print(f"测试 LogLoss: {result['overall']['model']['log_loss']:.6f}")
        return 0
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(f"独立测试失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

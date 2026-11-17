"""Build labelled, chronological datasets from validated Phase 1 outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import polars as pl
import pyarrow.parquet as pq

from cs2_pipeline.context import read_context
from cs2_pipeline.features import build_features, feature_columns, validate_feature_table
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import STATE_SCHEMA, PipelineError
from cs2_pipeline.split import PARTITIONS, chronological_split, frozen_split
from cs2_pipeline.validate import validate_directory

EXCLUDED_EVENT = "pgl-masters-bucharest-2026"
TABLE_NAMES = ("samples", "nodes", "features", "labels", "splits")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def held_out_event(event: str) -> bool:
    return EXCLUDED_EVENT in slug(event)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )


def source_identity(entry: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str]:
    """Dates/groups must be declared and checked against external provenance."""
    for column in ("match_id", "match_date", "event"):
        if not isinstance(entry.get(column), str) or not entry[column].strip():
            raise PipelineError(f"来源必须声明非空 {column}，不能从阵营、比分或文件名猜测。")
    try:
        if date.fromisoformat(entry["match_date"]).isoformat() != entry["match_date"]:
            raise ValueError("日期必须使用带连字符的标准格式")
    except ValueError as exc:
        raise PipelineError("match_date 必须是 YYYY-MM-DD 格式的真实比赛日期。") from exc
    external = metadata.get("external_source") or {}
    if not isinstance(external, dict):
        raise PipelineError("Phase 1 external_source 格式错误。")
    if external:
        if external.get("sha256") != metadata.get("demo_sha256"):
            raise PipelineError("Phase 1 外部来源与 Demo SHA256 不符。")
        if external.get("match_date") and external["match_date"] != entry["match_date"]:
            raise PipelineError("清单比赛日期与 Phase 1 外部来源不一致。")
        if external.get("event") and slug(external["event"]) != slug(entry["event"]):
            raise PipelineError("清单赛事与 Phase 1 外部来源不一致。")
        match = re.search(r"/matches/(\d+)(?:/|$)", urlparse(external.get("match_url", "")).path)
        if match and entry["match_id"] != f"hltv:{match[1]}":
            raise PipelineError("同一 HLTV 比赛必须使用同一个 hltv:<比赛ID> 分组，不能按地图拆组。")
    checksum = metadata.get("demo_sha256", "")
    if not isinstance(checksum, str) or not re.fullmatch(r"[a-f0-9]{64}", checksum):
        raise PipelineError("Phase 1 Demo SHA256 缺失或格式错误。")
    return {
        **{k: entry[k] for k in ("match_id", "match_date", "event")},
        "map_id": entry.get("map_id", external.get("map_id", f"{entry['match_id']}:{checksum}")),
        "part_index": str(entry.get("part_index", external.get("part_index", 1))),
    }


def build_map_tables(
    metadata: dict[str, Any],
    rounds: pl.DataFrame,
    snapshots: pl.DataFrame,
    identity: dict[str, str],
    *,
    feature_version: int = 1,
    context: pl.DataFrame | None = None,
) -> tuple[dict[str, pl.DataFrame], dict[str, int]]:
    """Filter whole time points, retaining ten identities and explicit node masks."""
    rate = metadata.get("tick_rate")
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
        raise PipelineError("缺少 Demo 实测 tick_rate；请核实计时后重新解析，不能用默认速率训练。")
    checksum = metadata["demo_sha256"]
    valid_rounds = rounds.filter(pl.col("is_valid_round"))
    states = (
        snapshots.filter(pl.col("is_valid_round"))
        .join(
            valid_rounds.select("round_id", "freeze_end_tick"),
            on="round_id",
            how="inner",
            validate="m:1",
        )
        .with_columns(
            pl.concat_str(
                pl.lit(checksum), pl.col("round_id"), pl.col("tick"), separator=":"
            ).alias("sample_id"),
            ((pl.col("tick") - pl.col("freeze_end_tick")) / rate).alias(
                "elapsed_since_freeze_end_seconds"
            ),
        )
    )
    before = states["sample_id"].n_unique()
    complete = (
        states.group_by("sample_id")
        .agg(
            pl.len().alias("n"),
            (pl.col("side") == "CT").sum().alias("ct"),
            (pl.col("side") == "T").sum().alias("t"),
            pl.col("steamid").n_unique().alias("unique_players"),
            pl.col("is_alive").is_not_null().all().alias("alive_known"),
        )
        .filter(
            (pl.col("n") == 10)
            & (pl.col("ct") == 5)
            & (pl.col("t") == 5)
            & (pl.col("unique_players") == 10)
            & pl.col("alive_known")
        )
    )
    states = states.join(complete.select("sample_id"), on="sample_id", how="semi")
    if states.is_empty():
        raise PipelineError(
            "没有双方各 5 名选手且存活状态明确的有效时点；请检查 Phase 1 质量报告。"
        )
    samples = (
        states.select(
            "sample_id",
            "round_id",
            "round_num",
            "tick",
            "elapsed_since_freeze_end_seconds",
        )
        .unique()
        .with_columns(
            pl.lit(checksum).alias("demo_sha256"),
            pl.lit(metadata["map_name"]).alias("map_name"),
            *[pl.lit(value).alias(key) for key, value in identity.items()],
        )
        .sort("sample_id")
    )
    if feature_version == 2:
        if context is None:
            raise PipelineError("V2 需要按同一 Demo SHA256 提取的游戏上下文。")
        if (
            samples.select("tick", "round_id")
            .join(
                context.select("tick", "round_id"),
                on=["tick", "round_id"],
                how="anti",
            )
            .height
        ):
            raise PipelineError("部分样本缺少同 tick/round_id 的游戏上下文，拒绝静默丢行。")
        samples = samples.join(context, on=["tick", "round_id"], how="left", validate="1:1")
    # Labels are never joined into the node or feature frames.
    labels = (
        samples.select("sample_id", "round_id")
        .join(
            valid_rounds.select("round_id", "winner_side"),
            on="round_id",
            validate="m:1",
        )
        .select("sample_id", (pl.col("winner_side") == "CT").cast(pl.UInt8).alias("y_ct_win"))
    )
    missing_columns = set(metadata.get("cleaning", {}).get("missing_snapshot_columns", []))
    nodes = (
        states.select(
            "sample_id",
            *STATE_SCHEMA,
            "team",
            "is_alive",
            "death_observed",
            "state_source",
        )
        .with_columns(
            (
                (pl.col("state_source") == "snapshot")
                & pl.all_horizontal(
                    pl.col(c).is_not_null() & pl.col(c).is_finite() for c in ("x", "y", "z")
                )
            ).alias("position_observed"),
            ((pl.col("state_source") == "snapshot") & pl.col("money").is_not_null()).alias(
                "economy_observed"
            ),
            (
                (pl.col("state_source") == "snapshot") & pl.col("equipment_value").is_not_null()
            ).alias("equipment_observed"),
            (
                pl.lit("primary_weapon" not in missing_columns)
                & (pl.col("state_source") == "snapshot")
            ).alias("primary_slot_observed"),
            (
                pl.lit("secondary_weapon" not in missing_columns)
                & (pl.col("state_source") == "snapshot")
            ).alias("secondary_slot_observed"),
        )
        .sort("sample_id", "steamid")
    )
    features = build_features(nodes, samples, feature_version=feature_version)
    validate_feature_table(features, feature_version=feature_version)
    return {"samples": samples, "nodes": nodes, "features": features, "labels": labels}, {
        "sample_count": samples.height,
        "node_count": nodes.height,
        "round_count": samples["round_id"].n_unique(),
        "incomplete_timepoints_excluded": before - samples.height,
    }


def validate_dataset_tables(tables: dict[str, pl.DataFrame], metadata: dict[str, Any]) -> None:
    """Reject dropped labels, duplicate keys, cross-split matches and temporal leaks."""
    validate_feature_table(tables["features"], feature_version=metadata.get("feature_version", 1))
    sample_ids = set(tables["features"]["sample_id"].to_list())
    for name in ("samples", "labels", "splits"):
        frame = tables[name]
        if "sample_id" not in frame.columns or frame["sample_id"].null_count():
            raise PipelineError(f"{name} 缺少有效的 sample_id。")
        if frame["sample_id"].n_unique() != frame.height:
            raise PipelineError(f"{name} 的 sample_id 重复。")
        if set(frame["sample_id"].to_list()) != sample_ids:
            raise PipelineError(f"{name} 的 sample_id 与特征不一致，拒绝静默丢行。")
    nodes, samples, labels, splits = (tables[n] for n in ("nodes", "samples", "labels", "splits"))
    if nodes.schema.get("steamid") != pl.UInt64:
        raise PipelineError("nodes.steamid 必须保持 UInt64。")
    if nodes.select(pl.struct("sample_id", "steamid").is_duplicated().any()).item():
        raise PipelineError("节点 (sample_id, steamid) 重复。")
    if set(nodes["sample_id"].to_list()) != sample_ids:
        raise PipelineError("节点与时点索引不一致。")
    if nodes.group_by("sample_id").len().filter(pl.col("len") != 10).height:
        raise PipelineError("每个时点必须保留 10 名选手节点。")
    if (
        labels.schema.get("y_ct_win") != pl.UInt8
        or labels.filter(pl.col("y_ct_win").is_null() | ~pl.col("y_ct_win").is_in([0, 1])).height
    ):
        raise PipelineError("标签必须是 UInt8 的 0/1，CT 胜为 1。")
    policy = metadata["split_policy"]
    partitions = (
        PARTITIONS if policy.get("method") == "frozen_four_way" else ("train", "validation", "test")
    )
    if splits.filter(pl.col("split").is_null() | ~pl.col("split").is_in(partitions)).height:
        raise PipelineError("划分名称与声明的三集合/四集合策略不一致。")
    for name in ("nodes", "samples", "features", "splits"):
        if {"winner_side", "y_ct_win", "reason", "reason_name", "end_tick"} & set(
            tables[name].columns
        ):
            raise PipelineError(f"{name} 含未来结果；标签必须独立保存。")
    if samples.filter(pl.col("event").map_elements(held_out_event, return_dtype=pl.Boolean)).height:
        raise PipelineError("训练数据混入保留赛事 pgl-masters-bucharest-2026。")
    joined = samples.join(splits, on="sample_id", validate="1:1")
    expected = metadata.get("split_expectation")
    if expected is not None:
        if not isinstance(expected, dict) or expected.get("group_by") != "event":
            raise PipelineError("预先声明的赛事划分格式无效。")
        if policy.get("group_by") != "event":
            raise PipelineError("此清单要求按赛事划分；请使用 --group-by event。")
        for partition in ("train", "validation", "test"):
            declared = expected.get(partition)
            if (
                not isinstance(declared, list)
                or not declared
                or len(set(declared)) != len(declared)
            ):
                raise PipelineError(f"预先声明的 {partition} 赛事列表无效。")
            actual = set(joined.filter(pl.col("split") == partition)["event"].to_list())
            if actual != set(declared):
                raise PipelineError(f"{partition} 赛事与预先声明的划分不一致。")
    if policy.get("method") == "frozen_four_way":
        assignments, verified = frozen_split(metadata["sources"], policy["plan"])
        if verified != policy:
            raise PipelineError("固定划分策略或计划哈希已改变。")
        if joined.filter(pl.col("split") != pl.col("match_id").replace_strict(assignments)).height:
            raise PipelineError("样本划分与冻结计划不同。")
    if "map_id" in samples.columns:
        if (
            samples.group_by("map_id", "round_num")
            .agg(pl.col("demo_sha256").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        ):
            raise PipelineError("同一地图回合出现在多个录制文件中，需核验分段重叠。")
    for group in ("match_id", "demo_sha256"):
        if (
            joined.group_by(group)
            .agg(pl.col("split").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        ):
            raise PipelineError(f"同一 {group} 跨越不同集合，存在数据泄漏。")
    if metadata["split_policy"]["group_by"] == "event":
        if (
            joined.group_by("event")
            .agg(pl.col("split").n_unique().alias("n"))
            .filter(pl.col("n") > 1)
            .height
        ):
            raise PipelineError("按赛事划分时，同一赛事不能跨集合。")
    boundaries = []
    for split in partitions:
        days = joined.filter(pl.col("split") == split)["match_date"].to_list()
        if days:
            boundaries.append((min(days), max(days)))
    if any(left[1] >= right[0] for left, right in zip(boundaries, boundaries[1:], strict=False)):
        raise PipelineError("集合日期重叠或倒序；同日/重叠区间必须留在同一集合。")
    targets = samples.select("sample_id", "demo_sha256", "round_id").join(labels, on="sample_id")
    if (
        targets.group_by("demo_sha256", "round_id")
        .agg(pl.col("y_ct_win").n_unique().alias("n"))
        .filter(pl.col("n") != 1)
        .height
    ):
        raise PipelineError("同一回合出现不同监督标签。")


def build_dataset(
    manifest: Path,
    out: Path,
    *,
    group_by: str = "match",
    feature_version: int | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"输出目录已存在：{out}；请指定新的 --out。")
    config = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise PipelineError("清单必须是含 sources 列表的 JSON 对象。")
    feature_version = (
        feature_version if feature_version is not None else config.get("feature_version", 1)
    )
    columns = feature_columns(feature_version)
    map_name = config.get("map_name")
    if not isinstance(map_name, str) or not map_name:
        raise PipelineError("清单必须声明 map_name；基线每次只处理一张地图。")
    collected: dict[str, list[pl.DataFrame]] = {k: [] for k in TABLE_NAMES[:-1]}
    included, excluded = [], []
    seen: dict[str, dict[str, str]] = {}
    for entry in config["sources"]:
        if not isinstance(entry, dict):
            raise PipelineError("sources 中每项必须是对象。")
        if held_out_event(str(entry.get("event", ""))):
            excluded.append({"path": entry.get("path"), "reason": "reserved_test_event"})
            continue
        if not isinstance(entry.get("path"), str):
            raise PipelineError("每个来源必须声明 Phase 1 输出目录 path。")
        source_path = (manifest.resolve().parent / entry["path"]).resolve()
        metadata = json.loads((source_path / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise PipelineError("Phase 1 metadata.json 必须是对象。")
        external = metadata.get("external_source") or {}
        if not isinstance(external, dict):
            raise PipelineError("Phase 1 external_source 必须是对象。")
        if held_out_event(str(external.get("event", ""))) or held_out_event(
            str(external.get("match_url", ""))
        ):
            excluded.append({"path": entry["path"], "reason": "reserved_test_event"})
            continue
        identity = source_identity(entry, metadata)
        if metadata.get("map_name") != map_name:
            raise PipelineError(f"{entry['path']} 不是 {map_name}；请为不同地图单独构建数据集。")
        checksum = metadata["demo_sha256"]
        if checksum in seen:
            if seen[checksum] != identity:
                raise PipelineError("同一个 Demo SHA256 的比赛/日期/赛事声明冲突。")
            excluded.append({"path": entry["path"], "reason": "duplicate_demo_sha256"})
            continue
        validation = validate_directory(source_path)
        if not validation["passed"]:
            raise PipelineError(f"{entry['path']} 未通过 Phase 1 重读验证：{validation['errors']}")
        tables, counts = build_map_tables(
            metadata,
            pl.read_parquet(source_path / "rounds.parquet"),
            pl.read_parquet(source_path / "snapshots.parquet"),
            identity,
            feature_version=feature_version,
            context=read_context(
                (manifest.resolve().parent / entry["context_path"]).resolve(),
                source_path,
                checksum,
            )
            if feature_version == 2 and entry.get("context_path")
            else None,
        )
        seen[checksum] = identity
        for name, table in tables.items():
            collected[name].append(table)
        included.append(
            {
                **identity,
                "demo_sha256": checksum,
                "path": entry["path"],
                **counts,
                "phase1_warnings": validation["warnings"],
                "source_file_sha256": {
                    name: sha256_file(source_path / name)
                    for name in (
                        "metadata.json",
                        "rounds.parquet",
                        "players.parquet",
                        "snapshots.parquet",
                    )
                },
            }
        )
        if feature_version == 2:
            context_path = (manifest.resolve().parent / entry["context_path"]).resolve()
            included[-1]["context_path"] = entry["context_path"]
            included[-1]["context_files_sha256"] = {
                f: sha256_file(context_path / f)
                for f in ("context.parquet", "context_metadata.json")
            }
    if not included:
        raise PipelineError("没有可用训练来源；请检查清单、重复数据和保留赛事排除规则。")
    tables = {name: pl.concat(frames).sort("sample_id") for name, frames in collected.items()}
    assignments, split_policy = (
        frozen_split(included, config["split_plan"])
        if "split_plan" in config
        else chronological_split(included, group_by=group_by)
    )
    tables["splits"] = tables["samples"].select(
        "sample_id",
        pl.col("match_id").replace_strict(assignments).alias("split"),
    )
    metadata = {
        "dataset_schema_version": 1,
        "feature_version": feature_version,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "map_name": map_name,
        "feature_columns": columns,
        "label": "y_ct_win: CT=1, T=0; labels stored separately",
        "sources": included,
        "excluded_sources": excluded,
        "reserved_event": EXCLUDED_EVENT,
        "split_policy": split_policy,
        "source_manifest_sha256": sha256_file(manifest),
        "split_expectation": config.get("split_expected"),
        "sample_count": tables["samples"].height,
        "node_count": tables["nodes"].height,
        "round_count": tables["samples"].select("demo_sha256", "round_id").unique().height,
        "limitations": [
            "Elapsed demo seconds include pauses; they are not an authoritative round countdown.",
            "No rating/Elo or map/match win model; unrecorded fuse duration remains unknown."
            if feature_version == 2
            else "No bomb state/timer, rating, Elo or map/match win model.",
            "Dead state remains null; only living players contribute to positional/equipment sums.",
        ],
    }
    validate_dataset_tables(tables, metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-stage-", dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        for name, table in tables.items():
            file = stage / f"{name}.parquet"
            table.write_parquet(file, compression="zstd")
            if (
                not pl.read_parquet(file).equals(table)
                or pq.read_table(file).num_rows != table.height
            ):
                raise PipelineError(f"{name} Parquet 往返验证失败。")
        metadata["file_sha256"] = {
            f"{name}.parquet": sha256_file(stage / f"{name}.parquet") for name in TABLE_NAMES
        }
        write_json(stage / "dataset.json", metadata)
        stage.rename(out)
    return metadata


def read_dataset(path: Path) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    metadata = json.loads((path / "dataset.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("dataset_schema_version") != 1:
        raise PipelineError("不支持的数据集格式；请重新构建。")
    if metadata.get("feature_columns") != feature_columns(metadata.get("feature_version", 1)):
        raise PipelineError("数据集特征白名单与当前代码不一致。")
    policy = metadata.get("split_policy")
    if not isinstance(policy, dict) or policy.get("group_by") not in ("match", "event"):
        raise PipelineError("数据集缺少有效的比赛/赛事划分策略；请重新构建。")
    if not isinstance(policy.get("evaluation_ready"), bool):
        raise PipelineError("数据集缺少正式评估可用性标记；请重新构建。")
    tables = {}
    for name in TABLE_NAMES:
        filename = f"{name}.parquet"
        if metadata.get("file_sha256", {}).get(filename) != sha256_file(path / filename):
            raise PipelineError(f"{filename} 已改变或哈希缺失；请从 Phase 1 重新构建。")
        tables[name] = pl.read_parquet(path / filename)
    validate_dataset_tables(tables, metadata)
    return metadata, tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/phase2.json"))
    parser.add_argument("--out", type=Path, default=Path("data/datasets/mirage_v1"))
    parser.add_argument("--group-by", choices=["match", "event"], default="match")
    parser.add_argument("--feature-version", type=int, choices=[1, 2])
    args = parser.parse_args(argv)
    try:
        metadata = build_dataset(
            args.manifest, args.out, group_by=args.group_by, feature_version=args.feature_version
        )
    except (PipelineError, OSError, ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        print(f"数据集构建失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"数据集：{args.out.resolve()}\n地图：{metadata['map_name']} | "
        f"回合：{metadata['round_count']} | 时点：{metadata['sample_count']} | "
        f"玩家节点：{metadata['node_count']} | 特征：{len(metadata['feature_columns'])}"
    )
    print("集合比赛数：" + json.dumps(metadata["split_policy"]["match_counts"], ensure_ascii=False))
    print(
        "正式评估："
        + (
            "已具备独立划分"
            if metadata["split_policy"]["evaluation_ready"]
            else metadata["split_policy"]["reason"]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independently audit labels, combat windows, identities and context alignment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import held_out_event, read_dataset, write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_data import DATASETS, ROOT, context_for_samples, sources
from cs2_pipeline.schema import PipelineError


def audit_sample_rounds(
    samples: pl.DataFrame, labels: pl.DataFrame, rounds: pl.DataFrame, tick_rate: float
) -> dict[str, int]:
    """Validate every label and tick against the corresponding recorded round."""
    joined = samples.join(labels, on="sample_id", validate="1:1").join(
        rounds.select(
            "round_id",
            pl.col("round_num").alias("source_round_num"),
            "freeze_end_tick",
            "end_tick",
            "winner_side",
            "is_valid_round",
        ),
        on="round_id",
        how="left",
        validate="m:1",
    )
    violations = {
        "missing_or_invalid_round": joined.filter(
            pl.col("is_valid_round").is_null() | ~pl.col("is_valid_round")
        ).height,
        "wrong_round_number": joined.filter(
            pl.col("round_num") != pl.col("source_round_num")
        ).height,
        "outside_combat_window": joined.filter(
            (pl.col("tick") < pl.col("freeze_end_tick")) | (pl.col("tick") >= pl.col("end_tick"))
        ).height,
        "wrong_elapsed": joined.filter(
            (
                pl.col("elapsed_since_freeze_end_seconds")
                - (pl.col("tick") - pl.col("freeze_end_tick")) / tick_rate
            ).abs()
            > 1e-9
        ).height,
        "wrong_label": joined.filter(
            pl.col("winner_side").is_null()
            | ~pl.col("winner_side").is_in(["CT", "T"])
            | (pl.col("y_ct_win") != (pl.col("winner_side") == "CT").cast(pl.UInt8))
        ).height,
    }
    if any(violations.values()):
        raise PipelineError(f"回合关联检查失败：{violations}")
    return violations


def audit_identities(nodes: pl.DataFrame, samples: pl.DataFrame) -> dict[str, int]:
    if nodes.schema["steamid"] != pl.UInt64:
        raise PipelineError("Steam ID 必须是 UInt64，不能转为浮点数。")
    states = nodes.join(
        samples.select("sample_id", "match_id", "demo_sha256", "round_id"),
        on="sample_id",
        validate="m:1",
    )
    roster = states.group_by("sample_id").agg(
        pl.len().alias("n"),
        pl.col("steamid").n_unique().alias("ids"),
        (pl.col("side") == "CT").sum().alias("ct"),
        (pl.col("side") == "T").sum().alias("t"),
    )
    bad_roster = roster.filter(
        (pl.col("n") != 10) | (pl.col("ids") != 10) | (pl.col("ct") != 5) | (pl.col("t") != 5)
    ).height
    known = states.filter(pl.col("team").is_not_null())
    changed_team = (
        known.group_by("match_id", "steamid")
        .agg(pl.col("team").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    team_both_sides = (
        known.group_by("sample_id", "team")
        .agg(pl.col("side").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    multiple_teams_per_side = (
        known.group_by("sample_id", "side")
        .agg(pl.col("team").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    side_changes_in_round = (
        states.group_by("demo_sha256", "round_id", "steamid")
        .agg(pl.col("side").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )
    death_alive = states.filter(pl.col("death_observed") & pl.col("is_alive")).height
    sorted_states = states.sort("demo_sha256", "round_id", "steamid", "tick").with_columns(
        (~pl.col("is_alive")).cum_max().over("demo_sha256", "round_id", "steamid").alias("died")
    )
    resurrection = sorted_states.filter(pl.col("is_alive") & pl.col("died")).height
    violations = {
        "bad_roster_ticks": bad_roster,
        "players_changing_team_within_match": changed_team,
        "team_on_both_sides": team_both_sides,
        "multiple_teams_per_side": multiple_teams_per_side,
        "side_changes_in_round": side_changes_in_round,
        "alive_after_observed_death": death_alive,
        "alive_after_prior_dead_state": resurrection,
    }
    if any(violations.values()):
        raise PipelineError(f"选手身份或存活状态检查失败：{violations}")
    return {**violations, "missing_team_nodes": states["team"].null_count(), "nodes": states.height}


def run(root: Path) -> dict[str, Any]:
    items = sources(root)
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "passed": False,
        "datasets": {},
        "feature_allowlists": {},
        "interpretation": "Outcome labels are separate. Combat membership uses the recorded "
        "round end retrospectively; no end tick or winner enters model features.",
    }
    identifiers: dict[str, set[str]] = {}
    for dataset in DATASETS:
        directory = root / "data/datasets" / dataset
        meta, tables = read_dataset(directory)
        samples, labels, nodes = (tables[name] for name in ("samples", "labels", "nodes"))
        identifiers[dataset] = set(samples["match_id"].to_list())
        if any(held_out_event(event) for event in samples["event"].unique()):
            raise PipelineError("输入含 Bucharest 保留赛事。")
        checked_rounds = 0
        for item in (i for i in items if i["dataset"] == dataset):
            source, path, metadata = item["source"], item["path"], item["metadata"]
            for filename, expected in source["source_file_sha256"].items():
                if sha256_file(path / filename) != expected:
                    raise PipelineError(f"来源文件哈希不符：{path / filename}")
            rounds = pl.read_parquet(path / "rounds.parquet")
            selection = samples.filter(pl.col("demo_sha256") == metadata["demo_sha256"])
            audit_sample_rounds(
                selection,
                labels.join(selection.select("sample_id"), on="sample_id", how="semi"),
                rounds,
                metadata["tick_rate"],
            )
            checked_rounds += selection.select("round_id").n_unique()
        identities = audit_identities(nodes, samples)
        _, context = context_for_samples(root, dataset, items, samples)
        report["datasets"][dataset] = {
            "dataset_sha256": sha256_file(directory / "dataset.json"),
            "matches": samples["match_id"].n_unique(),
            "recordings": len(meta["sources"]),
            "rounds_checked": checked_rounds,
            "samples_checked": samples.height,
            "identities": identities,
            "context": context,
        }
    if identifiers[DATASETS[0]] & identifiers[DATASETS[1]]:
        raise PipelineError("训练和观察过的 50 场存在比赛重叠。")
    forbidden = {
        "winner_side",
        "reason",
        "reason_name",
        "end_tick",
        "y_ct_win",
        "round_num",
        "match_id",
        "steamid",
        "team",
    }
    for version in (1, 2):
        columns = feature_columns(version)
        if forbidden & set(columns):
            raise PipelineError("最终结果、身份或比赛编号进入特征白名单。")
        report["feature_allowlists"][f"v{version}"] = {"count": len(columns), "columns": columns}
    report["previous_recovery_audit"] = json.loads(
        (root / "reports/holdout_50_audit_v2.json").read_text("utf-8")
    )["recovered_companion_rounds_checked"]
    report["passed"] = True
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "reports/phase2_data_audit.json")
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise PipelineError(f"报告已存在：{args.out}；请指定新路径。")
        report = run(ROOT)
        write_json(args.out, report)
        print(f"数据与因果上下文核对通过：{args.out}")
        return 0
    except (PipelineError, OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        print(f"数据审计失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

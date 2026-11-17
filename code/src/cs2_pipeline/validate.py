"""Validate tables in memory or reread a persisted CS2 dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

from cs2_pipeline.schema import PLAYER_SCHEMA, ROUND_SCHEMA, SNAPSHOT_SCHEMA


def validate_frames(
    metadata: dict[str, Any],
    rounds: pl.DataFrame,
    players: pl.DataFrame,
    snapshots: pl.DataFrame,
) -> dict[str, Any]:
    """Return machine-readable errors, warnings, missingness and counts."""
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def issue(code: str, count: int, message: str, *, warning: bool = False) -> None:
        if count:
            (warnings if warning else errors).append(
                {"code": code, "count": int(count), "message": message}
            )

    tables = {"rounds": rounds, "players": players, "snapshots": snapshots}
    report: dict[str, Any] = {
        "map_name": metadata.get("map_name"),
        "round_count": rounds.height,
        "unique_players": players["steamid"].n_unique() if "steamid" in players.columns else 0,
        "snapshot_count": snapshots.height,
        "observed_snapshot_count": snapshots.filter(pl.col("state_source") == "snapshot").height
        if "state_source" in snapshots.columns
        else 0,
        "event_derived_snapshot_count": snapshots.filter(
            pl.col("state_source") == "death_event"
        ).height
        if "state_source" in snapshots.columns
        else 0,
        "sample_tick_count": snapshots["tick"].n_unique() if "tick" in snapshots.columns else 0,
        "valid_sample_tick_count": 0,
        "missing_values": {name: df.null_count().row(0, named=True) for name, df in tables.items()},
        "errors": errors,
        "warnings": warnings,
    }

    def finish() -> dict[str, Any]:
        report.update(
            anomaly_count=sum(item["count"] for item in errors),
            warning_count=sum(item["count"] for item in warnings),
            passed=not errors,
        )
        return report

    for name, schema in (
        ("rounds", ROUND_SCHEMA),
        ("players", PLAYER_SCHEMA),
        ("snapshots", SNAPSHOT_SCHEMA),
    ):
        for column, dtype in schema.items():
            issue(
                "schema",
                int(tables[name].schema.get(column) != dtype),
                f"{name}.{column} 应为 {dtype}；请用本项目解析器重新导出。",
            )
    if errors:
        return finish()
    issue("empty_snapshots", int(snapshots.is_empty()), "Snapshot 为空；检查战斗区间及采样设置。")
    issue("empty_rounds", int(rounds.is_empty()), "未找到回合；请提供完整比赛 Demo。")
    issue("empty_players", int(players.is_empty()), "未识别参赛选手；检查 Steam ID 和阵营。")
    issue("missing_map", int(not metadata.get("map_name")), "Demo 未提供地图名称。", warning=True)
    issue(
        "tick_rate_fallback",
        int(metadata.get("tick_rate_source") == "awpy_default_64"),
        "Demo 缺少播放计时，采样采用 Awpy 默认 64 tick；使用前请核实。",
        warning=True,
    )
    issue(
        "missing_columns",
        len(metadata.get("cleaning", {}).get("missing_snapshot_columns", [])),
        "部分可选 Snapshot 字段不存在，已存为 null；见 metadata.cleaning。",
        warning=True,
    )
    issue(
        "duplicate_snapshot",
        snapshots.select(pl.struct("tick", "steamid").is_duplicated().sum()).item(),
        "存在重复 (tick, steamid)；请检查上游清洗。",
    )
    issue(
        "duplicate_player",
        players.select(pl.col("steamid").is_duplicated().sum()).item(),
        "players 表中 Steam ID 不唯一。",
    )
    for name, df in (("players", players), ("snapshots", snapshots)):
        issue(
            "steamid",
            df.filter(
                pl.col("steamid").is_null()
                | ~pl.col("steamid").is_between(76561197960265729, 76561202255233023)
            ).height,
            f"{name} 包含无效的个人 SteamID64；请从原始 Demo 重新读取，勿经过浮点数。",
        )
    issue(
        "unknown_player",
        snapshots.join(players.select("steamid"), on="steamid", how="anti").height,
        "Snapshot 的 Steam ID 未在 players 中登记。",
    )
    issue(
        "team_conflict",
        players.filter(pl.col("team_identity_conflict")).height,
        "同一 Steam ID 有多个真实队名；team 已留空，请核实队伍身份。",
        warning=True,
    )
    issue(
        "missing_team",
        players.filter(pl.col("team").is_null()).height,
        "部分选手无 Demo 队名，保持 null；不能用 T/CT 代替。",
        warning=True,
    )
    issue(
        "invalid_round",
        rounds.filter(~pl.col("is_valid_round").fill_null(False)).height,
        "存在非正常回合；已保留原编号并标记，使用前过滤 is_valid_round。",
        warning=True,
    )
    issue(
        "round_id",
        rounds.select(pl.col("round_id").is_duplicated().sum()).item(),
        "round_id 必须唯一；它用于区分重启前后相同的原始回合编号。",
    )
    valid_rounds = rounds.filter(pl.col("is_valid_round"))
    invalid_valid = valid_rounds.filter(
        pl.any_horizontal(
            pl.col(k).is_null()
            for k in ("round_num", "start_tick", "freeze_end_tick", "end_tick", "winner_side")
        )
        | (pl.col("round_num") <= 0)
        | (pl.col("start_tick") < 0)
        | (pl.col("start_tick") > pl.col("freeze_end_tick"))
        | (pl.col("freeze_end_tick") >= pl.col("end_tick"))
        | ~pl.col("winner_side").is_in(["T", "CT"])
        | pl.col("is_knife_round").fill_null(False)
    )
    issue(
        "round_range", invalid_valid.height, "有效回合有非法编号、边界或标签；检查 rounds.parquet。"
    )
    issue(
        "round_number_duplicate",
        valid_rounds.select(pl.col("round_num").is_duplicated().sum()).item(),
        "有效回合编号重复；重启回合必须标为无效，不能重编号。",
    )
    ordered = valid_rounds.sort("start_tick")
    issue(
        "round_overlap",
        ordered.filter(pl.col("start_tick") < pl.col("end_tick").shift(1)).height,
        "有效回合 tick 区间重叠，无法唯一关联 Snapshot。",
    )
    # Verify against independent table contents, not a previously saved 'passed' flag.
    mapping = snapshots.join(
        rounds.select("round_id", "round_num", "freeze_end_tick", "end_tick", "is_valid_round"),
        on="round_id",
        how="left",
        suffix="_round",
    )
    issue(
        "round_mapping",
        mapping.filter(
            pl.col("freeze_end_tick").is_null()
            | pl.col("tick").is_null()
            | (pl.col("tick") < pl.col("freeze_end_tick"))
            | (pl.col("tick") >= pl.col("end_tick"))
            | (pl.col("round_num") != pl.col("round_num_round")).fill_null(True)
            | (pl.col("is_valid_round") != pl.col("is_valid_round_round")).fill_null(True)
        ).height,
        "Snapshot 未正确映射到 [freeze_end_tick, end_tick)；请重新运行解析。",
    )
    issue(
        "health",
        snapshots.filter(
            (pl.col("health").is_null() & (pl.col("state_source") != "death_event"))
            | ~pl.col("health").is_between(0, 100)
        ).height,
        "血量缺失或不在 0..100；检查 Demo 实体字段。",
    )
    issue(
        "alive",
        snapshots.filter(
            pl.col("is_alive").is_null()
            | (pl.col("is_alive") & ((pl.col("health") <= 0) | pl.col("death_observed")))
            | (~pl.col("is_alive") & (pl.col("health") > 0) & ~pl.col("death_observed"))
        ).height,
        "存活状态与血量/已观测死亡不一致；不能让死亡玩家重新存活。",
    )
    issue(
        "side",
        snapshots.filter(pl.col("side").is_null() | ~pl.col("side").is_in(["T", "CT"])).height,
        "无法识别 T/CT 阵营；不要把真实队名当作 side。",
    )
    coordinates_bad = snapshots.filter(
        pl.any_horizontal(pl.col(k).is_null() | ~pl.col(k).is_finite() for k in ("x", "y", "z"))
    )
    issue(
        "coordinates",
        coordinates_bad.filter(pl.col("is_alive").fill_null(True)).height,
        "存活玩家坐标缺失或非有限数值；检查实体状态。",
    )
    issue(
        "dead_coordinates",
        coordinates_bad.filter(~pl.col("is_alive") & (pl.col("state_source") == "snapshot")).height,
        "死亡玩家坐标缺失；保留 null，未来空间模型需用存活掩码。",
        warning=True,
    )
    event_states = snapshots.filter(pl.col("state_source") == "death_event")
    issue(
        "event_derived_dead_states",
        event_states.height,
        "部分死亡玩家状态由此前死亡事件保留；当前坐标、血量和装备未知，保持 null。",
        warning=True,
    )
    issue(
        "state_source",
        snapshots.filter(
            pl.col("state_source").is_null()
            | ~pl.col("state_source").is_in(["snapshot", "death_event"])
            | ((pl.col("state_source") == "death_event") & ~pl.col("death_observed"))
        ).height,
        "state_source 非法或死亡补行缺少死亡证据；请重新解析。",
    )
    for column in (
        "armor",
        "money",
        "equipment_value",
        "smoke_grenades",
        "flashbangs",
        "he_grenades",
        "fire_grenades",
    ):
        issue(
            "negative_value",
            snapshots.filter(pl.col(column) < 0).height,
            f"{column} 出现负值，请核实 Demo。",
        )
    identity = snapshots.join(players.select("steamid", "team"), on="steamid", suffix="_roster")
    issue(
        "team_mapping",
        identity.filter(~pl.col("team").eq_missing(pl.col("team_roster"))).height,
        "Snapshot 的真实队伍身份与 players 不一致；换边不应改变 team。",
    )
    groups = snapshots.group_by("tick").agg(
        pl.len().alias("players"),
        (pl.col("side") == "T").sum().alias("t_count"),
        (pl.col("side") == "CT").sum().alias("ct_count"),
        pl.col("is_alive").sum().alias("alive_count"),
        pl.col("is_valid_round").all().alias("valid_round"),
    )
    issue(
        "player_count",
        groups.filter(
            (pl.col("players") > 10)
            | (pl.col("t_count") > 5)
            | (pl.col("ct_count") > 5)
            | (pl.col("alive_count") > 10)
        ).height,
        "单个采样 tick 超过职业 5v5 人数；检查重复玩家/观战者过滤。",
    )
    incomplete = groups.filter(
        (pl.col("players") != 10) | (pl.col("t_count") != 5) | (pl.col("ct_count") != 5)
    )
    issue(
        "incomplete_roster_tick",
        incomplete.height,
        "部分 tick 未观测到双方各 5 名选手；可能断线或死亡实体消失，不补造状态。",
        warning=True,
    )
    bad_ticks = set(coordinates_bad.filter(pl.col("is_alive"))["tick"].to_list())
    report["valid_sample_tick_count"] = groups.filter(
        (pl.col("players") == 10)
        & (pl.col("t_count") == 5)
        & (pl.col("ct_count") == 5)
        & pl.col("valid_round")
        & ~pl.col("tick").is_in(list(bad_ticks))
    ).height
    issue(
        "no_valid_sample_ticks",
        int(report["valid_sample_tick_count"] == 0),
        "没有双方各 5 名选手的有效采样 tick；请提供完整的职业比赛 GOTV Demo。",
    )
    requested = set(metadata.get("requested_sample_ticks", []))
    if requested:
        actual = set(snapshots["tick"].drop_nulls().to_list())
        issue(
            "missing_sample_ticks",
            len(requested - actual),
            "部分请求 tick 无状态，可能录制缺失或异常回合；请检查 Demo。",
            warning=True,
        )
        issue(
            "unexpected_sample_ticks",
            len(actual - requested),
            "Snapshot 包含未请求的 tick；请检查采样逻辑。",
        )
    for column in ("winner_side", "winner", "reason", "reason_name"):
        issue(
            "label_leakage",
            int(column in snapshots.columns),
            f"Snapshot 混入未来标签 {column}；标签只能保存在 rounds.parquet。",
        )
    return finish()


def validate_directory(path: Path) -> dict[str, Any]:
    """Reopen all four required files; reject unreadable or inconsistent output."""
    try:
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("metadata.json 必须是 JSON 对象")
        if not isinstance(metadata.get("cleaning", {}), dict):
            raise ValueError("metadata.cleaning 必须是 JSON 对象")
        tables = {}
        for name in ("rounds", "players", "snapshots"):
            file = path / f"{name}.parquet"
            tables[name] = pl.read_parquet(file)
            arrow = pq.read_table(file)
            if arrow.num_rows != tables[name].height:
                raise ValueError(f"{name}: Polars 和 Arrow 读取行数不同")
        report = validate_frames(metadata, **tables)
        report["files_readable"] = True
        if metadata.get("total_rounds") != tables["rounds"].height:
            report["errors"].append(
                {
                    "code": "metadata_round_count",
                    "count": 1,
                    "message": "metadata.total_rounds 与回合文件不符。",
                }
            )
            report["anomaly_count"] += 1
            report["passed"] = False
        return report
    except (OSError, ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        return {
            "passed": False,
            "files_readable": False,
            "anomaly_count": 1,
            "errors": [
                {
                    "code": "file_read",
                    "count": 1,
                    "message": f"无法重读四个输出文件：{exc}。请检查路径、权限并重新解析。",
                }
            ],
            "warnings": [],
        }


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"验证：{'通过' if report['passed'] else '失败'}",
        f"地图：{report.get('map_name')} | 回合：{report.get('round_count', 0)} | "
        f"选手：{report.get('unique_players', 0)}",
        f"Snapshot：{report.get('snapshot_count', 0)} | "
        f"采样 tick：{report.get('sample_tick_count', 0)} | "
        f"有效采样 tick：{report.get('valid_sample_tick_count', 0)}",
        f"异常：{report.get('anomaly_count', 0)} | 警告：{report.get('warning_count', 0)}",
    ]
    for severity in ("errors", "warnings"):
        for item in report.get(severity, []):
            lines.append(f"[{severity}/{item['code']}] {item['count']}: {item['message']}")
    missing = report.get("missing_values", {}).get("snapshots", {})
    lines.append(
        "Snapshot 缺失值："
        + json.dumps({k: v for k, v in missing.items() if v}, ensure_ascii=False)
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/processed/match"))
    parser.add_argument("--json", action="store_true", help="打印完整 JSON 验证报告")
    args = parser.parse_args(argv)
    report = validate_directory(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_summary(report))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())

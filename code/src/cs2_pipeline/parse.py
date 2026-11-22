"""Parse one real CS2 demo using the pinned Awpy 3.3.0 API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import tempfile
import time
from bisect import bisect_right
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline import __version__
from cs2_pipeline.schema import (
    PLAYER_SCHEMA,
    ROUND_SCHEMA,
    SNAPSHOT_SCHEMA,
    STATE_SCHEMA,
    PipelineError,
    conform,
    exact_ids,
    normalize_side,
)
from cs2_pipeline.validate import format_summary, validate_directory


def prepare_rounds(raw: pl.DataFrame) -> pl.DataFrame:
    """Retain Awpy round numbers, including overtime and anomalous rounds."""
    for column in ("round_num", "start_tick", "freeze_end_tick", "end_tick", "winner_side"):
        if column not in raw.columns:
            raise PipelineError(f"Awpy 回合表缺少 {column}；请核对安装版本和 Demo 完整性。")
    frame = conform(raw, ROUND_SCHEMA).with_columns(normalize_side("winner_side"))
    rows = frame.sort("start_tick", nulls_last=True).to_dicts()
    counts: dict[int, int] = {}
    for row in rows:
        counts[row["round_num"]] = counts.get(row["round_num"], 0) + 1
    for row in rows:
        reasons = []
        start, freeze, end = (row[k] for k in ("start_tick", "freeze_end_tick", "end_tick"))
        if row["round_num"] is None or row["round_num"] <= 0:
            reasons.append("invalid_round_number")
        if counts[row["round_num"]] > 1:
            reasons.append("repeated_round_number")
        if start is None or freeze is None or end is None:
            reasons.append("missing_boundary")
        elif not 0 <= start <= freeze < end:
            reasons.append("invalid_tick_order")
        if row["winner_side"] not in ("T", "CT"):
            reasons.append("missing_or_noncompetitive_winner")
        if row["is_knife_round"] is True:
            reasons.append("knife_round")
        if row["reason_name"] in ("round_draw", "game_start", "game_commencing"):
            reasons.append("aborted_or_restart")
        row["invalid_reason"] = ";".join(reasons) or None
    # Ambiguous intervals must not acquire an arbitrary round through an asof join.
    for i, row in enumerate(rows):
        for other in rows[i + 1 :]:
            if all(r[k] is not None for r in (row, other) for k in ("start_tick", "end_tick")):
                if max(row["start_tick"], other["start_tick"]) < min(
                    row["end_tick"], other["end_tick"]
                ):
                    for overlap in (row, other):
                        previous = overlap["invalid_reason"]
                        overlap["invalid_reason"] = (
                            previous + ";overlapping_round" if previous else "overlapping_round"
                        )
    for round_id, row in enumerate(rows):
        row["round_id"] = round_id
        row["is_valid_round"] = row["invalid_reason"] is None
    return pl.DataFrame(rows, schema=ROUND_SCHEMA) if rows else pl.DataFrame(schema=ROUND_SCHEMA)


def sampling_ticks(rounds: pl.DataFrame, tick_rate: float, interval: float) -> list[int]:
    """Sample each observed combat window from freeze end; end tick is excluded."""
    if not math.isfinite(interval) or interval <= 0:
        raise PipelineError("--interval 必须是有限的正数（秒）。")
    if not math.isfinite(tick_rate) or tick_rate <= 0 or interval * tick_rate < 1:
        raise PipelineError("采样间隔必须至少为一个 tick；请增大 --interval 或检查 tick rate。")
    ticks: set[int] = set()
    for row in rounds.iter_rows(named=True):
        freeze, end = row["freeze_end_tick"], row["end_tick"]
        if freeze is None or end is None or not 0 <= freeze < end:
            continue
        # Round each absolute offset to avoid cumulative drift with fractional intervals.
        count = math.ceil((end - freeze) / (interval * tick_rate))
        ticks.update(
            freeze + round(i * interval * tick_rate)
            for i in range(count)
            if freeze + round(i * interval * tick_rate) < end
        )
    return sorted(ticks)


def associate_rounds(states: pl.DataFrame, rounds: pl.DataFrame) -> pl.DataFrame:
    """Join by raw tick, retaining only [freeze_end_tick, end_tick) combat states."""
    # Only boundaries and QA flags enter this join: winner/reason never enter state rows.
    bounds = rounds.filter(
        pl.col("start_tick").is_not_null()
        & pl.col("freeze_end_tick").is_not_null()
        & (pl.col("start_tick") <= pl.col("freeze_end_tick"))
        & (pl.col("freeze_end_tick") < pl.col("end_tick"))
        & ~pl.col("invalid_reason").fill_null("").str.contains("overlapping_round")
    ).select("round_id", "round_num", "freeze_end_tick", "end_tick", "is_valid_round")
    if bounds.is_empty():
        return states.head(0).with_columns(
            pl.lit(None, dtype=pl.Int64).alias("round_id"),
            pl.lit(None, dtype=pl.Int64).alias("round_num"),
            pl.lit(None, dtype=pl.Boolean).alias("is_valid_round"),
        )
    return (
        states.sort("tick")
        .join_asof(bounds.sort("freeze_end_tick"), left_on="tick", right_on="freeze_end_tick")
        .filter(
            (pl.col("tick") >= pl.col("freeze_end_tick")) & (pl.col("tick") < pl.col("end_tick"))
        )
        .drop("freeze_end_tick", "end_tick")
    )


def prepare_players(roster: pl.DataFrame, states: pl.DataFrame) -> pl.DataFrame:
    """Track organizations by Steam ID, never by the last observed side."""
    roster = exact_ids(roster)
    names: dict[int, set[str]] = {}
    teams: dict[int, set[str]] = {}
    # Include only real participants seen on T/CT during combat; omit GOTV/spectators.
    ids = set(states["steamid"].drop_nulls().to_list()) - {0}
    for row in roster.to_dicts() + states.select("steamid", "name").to_dicts():
        sid = row["steamid"]
        if sid not in ids:
            continue
        if row.get("name"):
            names.setdefault(sid, set()).add(row["name"])
        if row.get("team_clan_name"):
            teams.setdefault(sid, set()).add(row["team_clan_name"])
    output = []
    for sid in sorted(ids):
        aliases, organizations = sorted(names.get(sid, set())), sorted(teams.get(sid, set()))
        output.append(
            {
                "steamid": sid,
                "name": aliases[0] if aliases else None,
                "names": aliases,
                "team": organizations[0] if len(organizations) == 1 else None,
                "team_names": organizations,
                "team_identity_conflict": len(organizations) > 1,
            }
        )
    return pl.DataFrame(output, schema=PLAYER_SCHEMA)


def clean_snapshots(
    raw: pl.DataFrame,
    rounds: pl.DataFrame,
    roster: pl.DataFrame,
    kills: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Normalize, deduplicate, link identities and conservatively track death."""
    raw = exact_ids(raw)
    if "tick" not in raw.columns:
        raise PipelineError("Awpy Snapshot 缺少 tick；无法关联回合。")
    missing = sorted(set(STATE_SCHEMA) - set(raw.columns))
    frame = conform(raw, STATE_SCHEMA).with_columns(normalize_side("side"))
    anonymous = frame.filter(pl.col("steamid").is_null())
    if anonymous.filter((pl.col("health") != 0).fill_null(True)).height:
        raise PipelineError("存活或未知状态的 Snapshot 缺少 Steam ID；请检查 Demo 完整性。")
    bot_rows = frame.filter(pl.col("steamid") == 0).height
    frame = frame.filter(pl.col("steamid").is_not_null() & (pl.col("steamid") != 0))
    spectators = frame.filter(pl.col("side").is_in(["spectator", "unassigned"]))
    frame = frame.filter(~pl.col("side").is_in(["spectator", "unassigned"]).fill_null(False))
    # Exact duplicates are safe to drop. Conflicting states fail instead of guessing.
    before = frame.height
    frame = frame.unique(maintain_order=True)
    removed = before - frame.height
    if frame.select(pl.struct("tick", "steamid").is_duplicated().any()).item():
        raise PipelineError(
            "同一 (tick, steamid) 存在冲突状态；请检查 Demo 或 Awpy，不能任意保留一条。"
        )
    before_mapping = frame.height
    frame = associate_rounds(frame, rounds)
    outside_combat = before_mapping - frame.height
    frame = frame.with_columns(
        pl.lit(False).alias("death_observed"), pl.lit("snapshot").alias("state_source")
    )
    restored = 0
    if not kills.is_empty():
        deaths = exact_ids(kills.select("tick", "victim_steamid"), "victim_steamid").rename(
            {"victim_steamid": "steamid"}
        )
        deaths = (
            associate_rounds(deaths, rounds)
            .group_by("round_id", "steamid")
            .agg(pl.col("tick").min().alias("death_tick"))
        )
        # Never guess a corpse's ID from coordinates; use only prior death events.
        frame, restored = restore_confirmed_deaths(frame, deaths)
        frame = (
            frame.join(deaths, on=["round_id", "steamid"], how="left")
            .with_columns(
                (pl.col("tick") >= pl.col("death_tick")).fill_null(False).alias("death_observed")
            )
            .drop("death_tick")
        )
    players = prepare_players(roster, frame.filter(pl.col("side").is_in(["T", "CT"])))
    frame = frame.join(players.select("steamid", "team"), on="steamid", how="left")
    # A recorded death is stronger evidence than a stale pawn's positive health.
    frame = frame.with_columns(
        pl.when(pl.col("death_observed"))
        .then(False)
        .otherwise(pl.col("health") > 0)
        .alias("is_alive")
    )
    report = {
        "missing_snapshot_columns": missing,
        "missing_roster_columns": sorted(
            {"steamid", "name", "team_clan_name"} - set(roster.columns)
        ),
        "exact_duplicates_removed": removed,
        "spectator_rows_removed": spectators.height,
        "bot_rows_removed": bot_rows,
        "anonymous_dead_rows_removed": anonymous.height,
        "confirmed_dead_rows_restored": restored,
        "outside_combat_rows_removed": outside_combat,
        "death_health_conflicts": frame.filter(
            pl.col("death_observed") & (pl.col("health") > 0)
        ).height,
    }
    return conform(frame, SNAPSHOT_SCHEMA).sort("tick", "steamid"), players, report


def restore_confirmed_deaths(
    frame: pl.DataFrame,
    deaths: pl.DataFrame,
) -> tuple[pl.DataFrame, int]:
    """Retain missing dead participants using past evidence; unknown state stays null.

    No future observation, position, economy or equipment is copied. A missing
    living/disconnected player is never silently declared dead.
    """
    existing = set(frame.select("tick", "steamid").iter_rows())
    history: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in frame.sort("tick").iter_rows(named=True):
        history.setdefault((row["round_id"], row["steamid"]), []).append(row)
    additions = []
    moments = frame.select("round_id", "round_num", "tick", "is_valid_round").unique()
    for death in deaths.iter_rows(named=True):
        sid, rid = death["steamid"], death["round_id"]
        prior = history.get((rid, sid), [])
        prior_ticks = [row["tick"] for row in prior]
        for moment in moments.filter(
            (pl.col("round_id") == rid) & (pl.col("tick") >= death["death_tick"])
        ).iter_rows(named=True):
            tick = moment["tick"]
            index = bisect_right(prior_ticks, tick) - 1
            if (tick, sid) in existing or index < 0:
                continue
            record = dict.fromkeys(STATE_SCHEMA)
            record.update(
                moment,
                steamid=sid,
                name=prior[index]["name"],
                side=prior[index]["side"],
                death_observed=True,
                state_source="death_event",
            )
            additions.append(record)
            existing.add((tick, sid))
    if additions:
        frame = pl.concat([frame, pl.DataFrame(additions, schema=frame.schema)])
    return frame, len(additions)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_outputs(
    out: Path,
    metadata: dict[str, Any],
    rounds: pl.DataFrame,
    players: pl.DataFrame,
    snapshots: pl.DataFrame,
) -> dict[str, Any]:
    """Stage, reread with Polars + Arrow, validate, then publish the whole directory."""
    if out.exists():
        raise PipelineError(f"输出目录已存在：{out}。请指定新的 --out，避免覆盖已有数据。")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cs2-stage-", dir=out.parent) as temp:
        stage = Path(temp) / "result"
        stage.mkdir()
        for name, frame in (("rounds", rounds), ("players", players), ("snapshots", snapshots)):
            frame.write_parquet(stage / f"{name}.parquet", compression="zstd")
        (stage / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        report = validate_directory(stage)
        metadata["validation"] = report
        (stage / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        # Invalid datasets remain inspectable with validation.passed=false and CLI exit 2.
        for attempt in range(20):
            try:
                stage.rename(out)
                break
            except PermissionError:
                if sys.platform != "win32" or out.exists() or attempt == 19:
                    raise
                time.sleep(0.05)
    return report


def parse_demo(
    demo_path: Path,
    out: Path,
    interval: float = 5.0,
    *,
    source: Path | None = None,
    rounds_override: pl.DataFrame | None = None,
    rounds_recovery: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decode one file. Import Awpy lazily to keep standalone validation lightweight."""
    if not demo_path.is_file():
        raise PipelineError(
            f"找不到 Demo：{demo_path}。请人工下载并解压真实 CS2 .dem 到 data/raw/match.dem，"
            "或通过 --demo 指定本地文件；详见 README.md。"
        )
    if demo_path.suffix.lower() != ".dem":
        raise PipelineError("输入必须是已解压的 .dem 文件；请先解压 .rar/.zip/.bz2。")
    if out.exists():
        raise PipelineError(f"输出目录已存在：{out}；请使用新的 --out。")
    if not math.isfinite(interval) or interval <= 0:
        raise PipelineError("--interval 必须是有限的正数（秒）。")
    from awpy import Demo

    started = time.perf_counter()
    print(f"打开 Demo：{demo_path}", flush=True)
    demo = Demo(demo_path)
    header = dict(demo.header)
    print("读取回合、参赛选手和死亡事件……", flush=True)
    demo.load("rounds", "players", "kills")
    if (rounds_override is None) != (rounds_recovery is None):
        raise PipelineError("外部回合恢复必须同时提供结果和可审计的来源。")
    if rounds_override is not None and not demo.rounds.is_empty():
        raise PipelineError("Awpy 已提供回合，拒绝替换现有解析结果。")
    rounds = prepare_rounds(demo.rounds if rounds_override is None else rounds_override)
    rate = float(demo.tick_rate)
    ticks = sampling_ticks(rounds, rate, interval)
    if not ticks:
        raise PipelineError("Demo 没有可采样的战斗区间；请提供完整的比赛 GOTV Demo。")
    print(f"采样 {len(ticks)} 个 tick，间隔 {interval:g} 秒……", flush=True)
    raw = demo.snapshots(ticks=ticks)
    snapshots, players, cleaning = clean_snapshots(raw, rounds, demo.players, demo.kills)
    duration = time.perf_counter() - started
    checksum = sha256_file(demo_path)
    external_source = None
    if source is not None:
        external_source = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(external_source, dict) or external_source.get("sha256") != checksum:
            raise PipelineError("--source 的 SHA256 与当前 Demo 不符；请使用该 Demo 的来源清单。")
        if external_source.get("map") != header.get("map_name"):
            raise PipelineError("来源清单的地图与 Demo 不同；请核实下载文件。")
    timing_known = bool(header.get("playback_ticks") and header.get("playback_time", 0) > 0)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "map_name": header.get("map_name"),
        "demo_filename": demo_path.name,
        "demo_sha256": checksum,
        "demo_size_bytes": demo_path.stat().st_size,
        "tick_rate": rate if timing_known else None,
        "sampling_tick_rate": rate,
        "tick_rate_source": "demo_playback_timing" if timing_known else "awpy_default_64",
        "demo_version": header.get("demo_version_name"),
        "game_build": header.get("build_num"),
        "parser_version": version("awpy"),
        "pipeline_version": __version__,
        "versions": {k: version(k) for k in ("awpy", "polars", "pyarrow")},
        "python_version": platform.python_version(),
        "total_rounds": rounds.height,
        "round_scope": "Awpy-emitted rounds only; missing warmup/incomplete rounds stay unknown.",
        "valid_rounds": rounds.filter(pl.col("is_valid_round")).height,
        "parsed_at_utc": datetime.now(UTC).isoformat(),
        "parse_duration_seconds": duration,
        "sampling_interval_seconds": interval,
        "requested_sample_ticks": ticks,
        "sampling_policy": "round_freeze_end + round(n * interval * tick_rate), tick < end_tick",
        "combat_window": "[freeze_end_tick, end_tick)",
        "match": {"event": None, "match_date": None},
        "external_source": external_source,
        "teams": players["team"].drop_nulls().unique().sort().to_list(),
        "header": header,
        "convars": dict(demo.convars),
        "cleaning": cleaning,
        "label_policy": "winner_side is only in rounds.parquet; QA flags are not model features",
        "null_policy": "Absent optional columns stay null; weapon null may mean an empty slot.",
        "observation_policy": (
            "snapshot = observed state; death_event = identity/death retained from past evidence, "
            "with unknown current position, health, economy and equipment left null"
        ),
    }
    if rounds_recovery is not None:
        metadata["rounds_recovery"] = rounds_recovery
        metadata["round_scope"] = (
            "Completed rounds recovered from recorded game-rules transitions and Awpy "
            "freeze-end events; method checked against Awpy on the companion recording."
        )
    return write_outputs(out, metadata, rounds, players, snapshots)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, default=Path("data/raw/match.dem"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/match"))
    parser.add_argument("--interval", type=float, default=5.0, help="采样间隔（秒），默认 5")
    parser.add_argument("--source", type=Path, help="可选来源 JSON；必须匹配 Demo SHA256 和地图")
    args = parser.parse_args(argv)
    try:
        report = parse_demo(args.demo, args.out, args.interval, source=args.source)
    except (PipelineError, OSError, ValueError, pl.exceptions.PolarsError) as exc:
        print(f"解析失败：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Native Awpy exceptions can occur lazily during any dataset decode.
        print(
            f"Awpy 无法完成解析（{type(exc).__name__}）：{exc}。"
            "请确认文件是完整、已解压的 CS2 GOTV Demo，且可在游戏中回放。",
            file=sys.stderr,
        )
        return 1
    print(format_summary(report))
    print(f"输出目录：{args.out.resolve()}")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

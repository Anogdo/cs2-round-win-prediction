"""Read causal game rules and bomb events at the exact Phase 1 sample ticks."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.parse import associate_rounds, sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory

CONTEXT_FEATURES = [
    "round_elapsed_game_seconds",
    "round_time_remaining_seconds",
    "bomb_planted",
    "bomb_site_a",
    "bomb_site_b",
    "bomb_seconds_since_plant",
    "bomb_time_remaining_seconds",
    "is_paused",
]
CONTEXT_SCHEMA = {
    "tick": pl.Int64,
    "round_id": pl.Int64,
    **dict.fromkeys(CONTEXT_FEATURES, pl.Float64),
}
RULE_ROOT = "CCSGameRulesProxy.CCSGameRules."
RULE_PROPERTIES = {
    "round_start_time": "m_fRoundStartTime",
    "round_duration": "m_iRoundTime",
    "engine_bomb_planted": "m_bBombPlanted",
    "freeze": "m_bFreezePeriod",
    "game_paused": "m_bGamePaused",
    "technical_timeout": "m_bTechnicalTimeOut",
    "waiting_for_resume": "m_bMatchWaitingForResume",
    "t_timeout": "m_bTerroristTimeOutActive",
    "ct_timeout": "m_bCTTimeOutActive",
}
PAUSE_COLUMNS = (
    "game_paused",
    "technical_timeout",
    "waiting_for_resume",
    "t_timeout",
    "ct_timeout",
)


def known_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def collapse_rules(raw: pl.DataFrame) -> pl.DataFrame:
    """Global rules repeated on player rows must agree, not be arbitrarily selected."""
    columns = ["game_time", *RULE_PROPERTIES]
    expressions = []
    for column in columns:
        if column not in raw.columns:
            raw = raw.with_columns(pl.lit(None).alias(column))
        conflict = raw.group_by("tick").agg(pl.col(column).drop_nulls().n_unique().alias("n"))
        if conflict.filter(pl.col("n") > 1).height:
            raise PipelineError(f"同 tick 的全局游戏规则 {column} 不一致，拒绝任意取值。")
        expressions.append(pl.col(column).drop_nulls().first().alias(column))
    return raw.group_by("tick").agg(expressions).sort("tick")


def derive_context(
    moments: pl.DataFrame,
    rules: pl.DataFrame,
    bomb: pl.DataFrame,
    cvars: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Use only current rules and events at/before tick, resetting each round.

    The round-end tick, winner and later bomb explosions are never consulted.
    Active/unknown pause or freeze state makes clock values unknown.
    """
    by_tick = {r["tick"]: r for r in rules.iter_rows(named=True)}
    event_rounds: dict[int, list[dict[str, Any]]] = {}
    for event in bomb.sort("tick").iter_rows(named=True):
        event_rounds.setdefault(event["round_id"], []).append(event)
    c4_changes = []
    if cvars is not None and {"tick", "cvarname", "cvarvalue"} <= set(cvars.columns):
        for event in (
            cvars.filter(pl.col("cvarname") == "mp_c4timer").sort("tick").iter_rows(named=True)
        ):
            try:
                value = float(event["cvarvalue"])
                if math.isfinite(value) and value > 0:
                    c4_changes.append((event["tick"], value))
            except (TypeError, ValueError):
                pass
    output = []
    for moment in moments.sort("tick").iter_rows(named=True):
        tick, rid = moment["tick"], moment["round_id"]
        state = by_tick.get(tick, {})
        pause_flags = [state.get(c) for c in PAUSE_COLUMNS]
        paused = (
            True
            if any(v is True for v in pause_flags)
            else (False if all(v is False for v in pause_flags) else None)
        )
        safe_clock = paused is False and state.get("freeze") is False
        now, start, duration = (
            state.get(c) for c in ("game_time", "round_start_time", "round_duration")
        )
        elapsed = now - start if safe_clock and known_finite(now) and known_finite(start) else None
        if elapsed is not None and elapsed < -1e-3:
            raise PipelineError("游戏规则时钟早于当前回合开始时间，请检查 tick 对齐。")
        elapsed = max(0.0, elapsed) if elapsed is not None else None
        past = [event for event in event_rounds.get(rid, []) if event["tick"] <= tick]
        plant = None
        for event in past:
            if event["event"] == "finish_plant":
                plant = event
            elif event["event"] == "defuse":
                plant = None
        planted = state.get("engine_bomb_planted")
        # If the engine value is missing, completed round reset + observed events is causal.
        if planted is None:
            planted = plant is not None
        site = plant.get("bombsite") if planted and plant else None
        plant_time = by_tick.get(plant["tick"], {}).get("game_time") if plant else None
        since_plant = (
            now - plant_time
            if planted and safe_clock and known_finite(now) and known_finite(plant_time)
            else None
        )
        if since_plant is not None:
            since_plant = max(0.0, since_plant)
        timer = None
        if plant:
            position = bisect.bisect_right([t for t, _ in c4_changes], plant["tick"]) - 1
            if position >= 0:
                timer = c4_changes[position][1]
        round_remaining = (
            max(0.0, duration - elapsed)
            if (not planted and elapsed is not None and known_finite(duration) and duration > 0)
            else None
        )
        output.append(
            {
                "tick": tick,
                "round_id": rid,
                "round_elapsed_game_seconds": elapsed,
                "round_time_remaining_seconds": round_remaining,
                "bomb_planted": float(planted),
                "bomb_site_a": float(site == "A") if site or not planted else None,
                "bomb_site_b": float(site == "B") if site or not planted else None,
                "bomb_seconds_since_plant": since_plant,
                "bomb_time_remaining_seconds": max(0.0, timer - since_plant)
                if timer is not None and since_plant is not None
                else None,
                "is_paused": float(paused) if paused is not None else None,
            }
        )
    return pl.DataFrame(output, schema=CONTEXT_SCHEMA)


def verify_event_only_coverage(rounds: pl.DataFrame, bomb: pl.DataFrame) -> None:
    """A recorded bomb ending requires a preceding plant event in that round.

    This is a quality check, not an input used to derive individual states.
    """
    expected = rounds.filter(
        pl.col("is_valid_round") & pl.col("reason_name").is_in(["target_bombed", "bomb_defused"])
    ).select("round_id")
    observed = bomb.filter(pl.col("event") == "finish_plant").select("round_id").unique()
    if expected.join(observed, on="round_id", how="anti").height:
        raise PipelineError("炸弹结束的回合缺少已记录的下包事件，不能使用事件回退。")


def enrich_demo(
    demo: Path, processed: Path, out: Path, *, allow_event_only: bool = False
) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"上下文目录已存在：{out}；请指定新的 --out。")
    quality = validate_directory(processed)
    if not quality["passed"]:
        raise PipelineError("Phase 1 输入未通过重新验证，不能补充上下文。")
    metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
    if sha256_file(demo) != metadata["demo_sha256"]:
        raise PipelineError("上下文 Demo 与 Phase 1 SHA256 不一致。")
    try:
        from awpy import Demo
        from demoparser2 import DemoParser
    except ImportError as exc:
        raise PipelineError(
            "请安装补充依赖：uv sync --frozen --extra context --extra ml --extra dev。"
        ) from exc
    rounds = pl.read_parquet(processed / "rounds.parquet")
    moments = pl.read_parquet(processed / "snapshots.parquet").select("tick", "round_id").unique()
    awpy = Demo(demo)
    bomb = associate_rounds(
        awpy.bomb.filter(pl.col("event").is_in(["finish_plant", "defuse"])), rounds
    )
    parser = DemoParser(str(demo))
    rule_failure = None
    try:
        updated = set(parser.list_updated_fields())
        mapping = {
            RULE_ROOT + value: key
            for key, value in RULE_PROPERTIES.items()
            if RULE_ROOT + value in updated
        }
        ticks = sorted(set(moments["tick"].to_list()) | set(bomb["tick"].to_list()))
        print(f"读取 {len(ticks)} 个规则 tick：{demo.name}", flush=True)
        raw = pl.from_pandas(parser.parse_ticks(["game_time", *mapping], ticks=ticks)).rename(
            mapping
        )
        rules = collapse_rules(raw)
    except Exception as exc:
        # Some newer demos' event stream is readable while their rules entities
        # are unsupported. Never infer the missing clock or pause state.
        if not allow_event_only or str(exc) not in {"EntityNotFound", "ClassNotFound"}:
            raise
        verify_event_only_coverage(rounds, bomb)
        mapping = {}
        rule_failure = str(exc)
        rules = pl.DataFrame(schema={"tick": pl.Int64})
        print(f"规则不可读取（{rule_failure}）；仅使用已记录的炸弹事件：{demo.name}", flush=True)
    cvars = awpy.events["server_cvar"] if "server_cvar" in awpy.events.names else None
    context = derive_context(moments, rules, bomb, cvars)
    notes = {
        "context_schema_version": 1,
        "demo_sha256": metadata["demo_sha256"],
        "providers": {
            "bomb": f"awpy {version('awpy')}",
            "game_rules": f"demoparser2 {version('demoparser2')}",
        },
        "missing_rule_properties": sorted(set(RULE_PROPERTIES) - set(mapping.values())),
        "game_rules_status": "event_only" if rule_failure else "available",
        "game_rules_error": rule_failure,
        "bomb_state_policy": (
            "Events at/before tick only; rules unavailable, so event-only inference "
            "is marked and clock/pause values stay null."
            if rule_failure
            else "Current game-rules state, with recorded-event fallback."
        ),
        "row_count": context.height,
        "non_null_counts": {c: context.height - context[c].null_count() for c in CONTEXT_FEATURES},
        "clock_policy": (
            "Current game_time/start/duration; active or unknown pause => null. "
            "Round countdown is not applicable after plant."
        ),
        "fuse_policy": (
            "Only mp_c4timer changes recorded at/before plant; "
            "no default, future explosion or final convar value."
        ),
        "phase1_file_sha256": {
            f: sha256_file(processed / f) for f in ("rounds.parquet", "snapshots.parquet")
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".context-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        context.write_parquet(stage / "context.parquet", compression="zstd")
        if not pl.read_parquet(stage / "context.parquet").equals(context):
            raise PipelineError("上下文 Parquet 往返读取不一致。")
        notes["context_file_sha256"] = sha256_file(stage / "context.parquet")
        (stage / "context_metadata.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        stage.rename(out)
    return notes


def read_context(path: Path, processed: Path, demo_sha256: str) -> pl.DataFrame:
    notes = json.loads((path / "context_metadata.json").read_text(encoding="utf-8"))
    if notes.get("demo_sha256") != demo_sha256 or notes.get("context_schema_version") != 1:
        raise PipelineError("上下文版本或 Demo SHA256 不匹配。")
    if notes.get("context_file_sha256") != sha256_file(path / "context.parquet"):
        raise PipelineError("上下文文件发生变化，请重新提取。")
    for filename in ("rounds.parquet", "snapshots.parquet"):
        if notes.get("phase1_file_sha256", {}).get(filename) != sha256_file(processed / filename):
            raise PipelineError("Phase 1 已变化，不能复用旧上下文。")
    frame = pl.read_parquet(path / "context.parquet")
    if frame.schema != pl.Schema(CONTEXT_SCHEMA) or frame["tick"].n_unique() != frame.height:
        raise PipelineError("上下文 Schema 或 tick 唯一性错误。")
    if frame.filter(
        pl.any_horizontal(
            pl.col(c).is_not_null() & ~pl.col(c).is_finite() for c in CONTEXT_FEATURES
        )
    ).height:
        raise PipelineError("上下文含非有限数值。")
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        notes = enrich_demo(args.demo, args.processed, args.out)
    except Exception as exc:
        print(f"上下文提取失败（{type(exc).__name__}）：{exc}", file=sys.stderr)
        return 1
    print(f"上下文已保存：{args.out.resolve()}；非空统计：{notes['non_null_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

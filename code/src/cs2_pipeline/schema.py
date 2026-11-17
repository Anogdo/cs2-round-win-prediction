"""Stable output schemas; Steam IDs never pass through floating point."""

from __future__ import annotations

import polars as pl

ROUND_SCHEMA = {
    "round_id": pl.Int64,
    "round_num": pl.Int64,
    "start_tick": pl.Int64,
    "freeze_end_tick": pl.Int64,
    "end_tick": pl.Int64,
    "official_end_tick": pl.Int64,
    "winner_side": pl.String,
    "reason": pl.Int64,
    "reason_name": pl.String,
    "is_knife_round": pl.Boolean,
    "is_valid_round": pl.Boolean,
    "invalid_reason": pl.String,
}
STATE_SCHEMA = {
    "tick": pl.Int64,
    "steamid": pl.UInt64,
    "name": pl.String,
    "side": pl.String,
    "x": pl.Float64,
    "y": pl.Float64,
    "z": pl.Float64,
    "health": pl.Int64,
    "armor": pl.Int64,
    "money": pl.Int64,
    "equipment_value": pl.Int64,
    "primary_weapon": pl.String,
    "secondary_weapon": pl.String,
    "has_helmet": pl.Boolean,
    "has_defuser": pl.Boolean,
    "has_bomb": pl.Boolean,
    "smoke_grenades": pl.Int64,
    "flashbangs": pl.Int64,
    "he_grenades": pl.Int64,
    "fire_grenades": pl.Int64,
}
SNAPSHOT_SCHEMA = {
    **STATE_SCHEMA,
    "round_id": pl.Int64,
    "round_num": pl.Int64,
    "team": pl.String,
    "is_alive": pl.Boolean,
    "death_observed": pl.Boolean,
    "state_source": pl.String,
    "is_valid_round": pl.Boolean,
}
PLAYER_SCHEMA = {
    "steamid": pl.UInt64,
    "name": pl.String,
    "names": pl.List(pl.String),
    "team": pl.String,
    "team_names": pl.List(pl.String),
    "team_identity_conflict": pl.Boolean,
}
SIDES = {"T", "CT"}


class PipelineError(Exception):
    """An actionable input, parser, or storage error."""


def normalize_side(column: str) -> pl.Expr:
    """Translate documented Awpy side names; preserve unknowns for validation."""
    return (
        pl.col(column)
        .cast(pl.String)
        .replace({"terrorist": "T", "counter-terrorist": "CT", "2": "T", "3": "CT"})
    )


def exact_ids(frame: pl.DataFrame, column: str = "steamid") -> pl.DataFrame:
    """Refuse floats even if they happen to look integral: precision may be lost."""
    if column not in frame.columns:
        raise PipelineError(f"缺少 {column}；请检查 Demo 和 Awpy 输出字段。")
    if frame.schema[column] in (pl.Float32, pl.Float64):
        raise PipelineError(f"{column} 是浮点数，可能已丢失精度；请重新从 Demo 读取 UInt64。")
    try:
        return frame.with_columns(pl.col(column).cast(pl.UInt64, strict=True))
    except pl.exceptions.PolarsError as exc:
        raise PipelineError(
            f"{column} 含非法 Steam ID，必须是无损的 UInt64 或十进制字符串。"
        ) from exc


def conform(frame: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """Fill absent optional fields with typed nulls, never invented zeros."""
    return frame.select(
        [
            pl.col(name).cast(dtype, strict=True)
            if name in frame.columns
            else pl.lit(None, dtype=dtype).alias(name)
            for name, dtype in schema.items()
        ]
    )

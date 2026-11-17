"""Causal state-only features, separate from labels, IDs and quality flags."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import polars as pl

from cs2_pipeline.context import CONTEXT_FEATURES
from cs2_pipeline.schema import PipelineError

NUMERIC_SUMS = (
    "health",
    "armor",
    "money",
    "equipment_value",
    "smoke_grenades",
    "flashbangs",
    "he_grenades",
    "fire_grenades",
    "has_helmet",
    "has_defuser",
    "has_bomb",
)
SIDE_METRICS = (
    ("alive_count",)
    + tuple(f"alive_{c}_sum" for c in NUMERIC_SUMS)
    + (
        "alive_x_mean",
        "alive_y_mean",
        "alive_z_mean",
    )
)
FEATURE_COLUMNS = (
    ["elapsed_since_freeze_end_seconds"]
    + [f"{side}_{metric}" for side in ("ct", "t") for metric in SIDE_METRICS]
    + [f"ct_minus_t_{metric}" for metric in SIDE_METRICS]
    + ["alive_centroid_distance_xy"]
)
FEATURE_SCHEMA = {"sample_id": pl.String, **dict.fromkeys(FEATURE_COLUMNS, pl.Float64)}
PRIMARY_GROUPS = {
    "awp": {"awp"},
    "ak47": {"ak47"},
    "m4": {"m4a1", "m4a1_silencer"},
    "other_rifle": {"famas", "galilar", "aug", "sg556"},
    "other_sniper": {"ssg08", "scar20", "g3sg1"},
    "smg": {"mac10", "mp9", "mp7", "mp5sd", "ump45", "p90", "bizon"},
    "shotgun": {"nova", "xm1014", "mag7", "sawedoff"},
    "lmg": {"m249", "negev"},
    "empty": set(),
    "unknown": set(),
}
SECONDARY_GROUPS = {
    "heavy_pistol": {"deagle", "revolver"},
    "other_pistol": {
        "glock",
        "hkp2000",
        "usp_silencer",
        "p250",
        "fiveseven",
        "tec9",
        "cz75a",
        "elite",
    },
    "empty": set(),
    "unknown": set(),
}
WEAPON_METRICS = [
    f"{slot}_{group}_count"
    for slot, groups in (("primary", PRIMARY_GROUPS), ("secondary", SECONDARY_GROUPS))
    for group in groups
]


def feature_columns(feature_version: int = 1) -> list[str]:
    if feature_version == 1:
        return FEATURE_COLUMNS
    if feature_version == 2:
        return (
            FEATURE_COLUMNS[1:]
            + CONTEXT_FEATURES
            + [
                f"{side}_alive_{metric}"
                for side in ("ct", "t", "ct_minus_t")
                for metric in WEAPON_METRICS
            ]
        )
    raise PipelineError(f"不支持特征版本 {feature_version}；只能使用 1 或 2。")


def feature_schema(feature_version: int = 1) -> dict:
    return {"sample_id": pl.String, **dict.fromkeys(feature_columns(feature_version), pl.Float64)}


def weapon_counts(living: list[dict[str, Any]], slot: str, groups: dict) -> dict[str, float | None]:
    if any(r.get(f"{slot}_slot_observed") is not True for r in living):
        return dict.fromkeys(groups)
    counts = dict.fromkeys(groups, 0.0)
    for row in living:
        weapon = row.get(f"{slot}_weapon")
        canonical = weapon.removeprefix("weapon_") if isinstance(weapon, str) else None
        group = (
            "empty"
            if canonical is None
            else next((key for key, names in groups.items() if canonical in names), "unknown")
        )
        counts[group] += 1
    return counts


def complete_sum(values: Iterable[Any]) -> float | None:
    """Empty living side sums to zero; missing living values stay unknown."""
    values = list(values)
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    return float(sum(values))


def build_features(
    nodes: pl.DataFrame,
    samples: pl.DataFrame,
    *,
    feature_version: int = 1,
) -> pl.DataFrame:
    """Use only current live-player state and elapsed time known at that tick.

    Dead coordinates/equipment, Steam IDs, names, organizations, round result,
    round number, split and parser quality flags never become model inputs.
    """
    required = {"sample_id", "side", "is_alive", "x", "y", "z", *NUMERIC_SUMS}
    if feature_version == 2:
        required.update(
            {
                "primary_weapon",
                "secondary_weapon",
                "primary_slot_observed",
                "secondary_slot_observed",
            }
        )
        if missing_context := set(CONTEXT_FEATURES) - set(samples.columns):
            raise PipelineError(f"V2 缺少因果上下文：{sorted(missing_context)}；请先运行 context。")
    columns = feature_columns(feature_version)
    if missing := required - set(nodes.columns):
        raise PipelineError(f"玩家输入缺少字段：{sorted(missing)}。请重新构建数据集。")
    if not {"sample_id", "elapsed_since_freeze_end_seconds"} <= set(samples.columns):
        raise PipelineError("时点输入缺少 sample_id 或冻结结束后的经过时间。")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in nodes.select(sorted(required)).iter_rows(named=True):
        if row["side"] not in ("T", "CT") or row["is_alive"] is None:
            raise PipelineError("玩家阵营或存活状态未知；不能猜测人数。")
        grouped.setdefault(row["sample_id"], []).append(row)
    if samples["sample_id"].n_unique() != samples.height:
        raise PipelineError("时点表的 sample_id 不唯一。")
    if set(grouped) != set(samples["sample_id"].to_list()):
        raise PipelineError("玩家表与时点表的 sample_id 不一致。")
    output = []
    sample_columns = ["sample_id", "elapsed_since_freeze_end_seconds"] + (
        CONTEXT_FEATURES if feature_version == 2 else []
    )
    for sample in samples.select(sample_columns).iter_rows(named=True):
        elapsed = sample["elapsed_since_freeze_end_seconds"]
        if elapsed is None or not math.isfinite(elapsed) or elapsed < 0:
            raise PipelineError("经过时间必须是当前 tick 可知的非负秒数。")
        record: dict[str, Any] = dict(sample)
        rows = grouped[sample["sample_id"]]
        for side, prefix in (("CT", "ct"), ("T", "t")):
            side_rows = [r for r in rows if r["side"] == side]
            if len(side_rows) != 5:
                raise PipelineError(f"{sample['sample_id']} 的 {side} 未保留 5 名选手身份。")
            living = [r for r in side_rows if r["is_alive"]]
            record[f"{prefix}_alive_count"] = float(len(living))
            for column in NUMERIC_SUMS:
                record[f"{prefix}_alive_{column}_sum"] = complete_sum(r[column] for r in living)
            for coordinate in ("x", "y", "z"):
                total = complete_sum(r[coordinate] for r in living)
                record[f"{prefix}_alive_{coordinate}_mean"] = (
                    total / len(living) if living and total is not None else None
                )
        for metric in SIDE_METRICS:
            ct, t = record[f"ct_{metric}"], record[f"t_{metric}"]
            record[f"ct_minus_t_{metric}"] = ct - t if ct is not None and t is not None else None
        centers = [
            record[f"{side}_alive_{axis}_mean"] for side in ("ct", "t") for axis in ("x", "y")
        ]
        record["alive_centroid_distance_xy"] = (
            math.hypot(centers[0] - centers[2], centers[1] - centers[3])
            if all(v is not None for v in centers)
            else None
        )
        if feature_version == 2:
            for side, prefix in (("CT", "ct"), ("T", "t")):
                living = [r for r in rows if r["side"] == side and r["is_alive"]]
                for slot, groups in (("primary", PRIMARY_GROUPS), ("secondary", SECONDARY_GROUPS)):
                    for group, count in weapon_counts(living, slot, groups).items():
                        record[f"{prefix}_alive_{slot}_{group}_count"] = count
            for metric in WEAPON_METRICS:
                ct, t = record[f"ct_alive_{metric}"], record[f"t_alive_{metric}"]
                record[f"ct_minus_t_alive_{metric}"] = (
                    ct - t if ct is not None and t is not None else None
                )
        output.append(record)
    return (
        pl.DataFrame(output, schema=feature_schema(feature_version))
        .select("sample_id", *columns)
        .sort("sample_id")
    )


def validate_feature_table(features: pl.DataFrame, *, feature_version: int = 1) -> None:
    """Use an explicit allowlist: a new column never enters training by accident."""
    schema = feature_schema(feature_version)
    columns = feature_columns(feature_version)
    if set(features.columns) != set(schema):
        raise PipelineError("特征列与固定白名单不一致；禁止自动使用标签、身份或其他额外列。")
    if features.schema != pl.Schema(schema):
        raise PipelineError("特征表字段类型或顺序错误；请重新构建。")
    if features.is_empty() or features["sample_id"].null_count():
        raise PipelineError("特征表为空或 sample_id 缺失。")
    if features["sample_id"].n_unique() != features.height:
        raise PipelineError("特征表 sample_id 重复。")
    if features.filter(
        pl.any_horizontal(pl.col(c).is_not_null() & ~pl.col(c).is_finite() for c in columns)
    ).height:
        raise PipelineError("特征包含 NaN/Inf；未知值应存为 null。")

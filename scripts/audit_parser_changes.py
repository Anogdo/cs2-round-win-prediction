"""Quantify parser-version differences without using winner labels or model scores."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cs2_pipeline.dataset import write_json

FIELDS = (
    "side",
    "team",
    "is_alive",
    "health",
    "money",
    "equipment_value",
    "smoke_grenades",
    "flashbangs",
    "he_grenades",
    "fire_grenades",
    "primary_weapon",
    "secondary_weapon",
)


def main() -> None:
    plan_path = Path("configs/reparse_expanded_2026.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    total = {
        "old_rows": 0,
        "new_rows": 0,
        "removed_keys": 0,
        "added_keys": 0,
        "changed_all": dict.fromkeys(FIELDS, 0),
        "changed_both_alive": dict.fromkeys(FIELDS, 0),
    }
    details = []
    for item in plan["sources"]:
        old = pl.read_parquet(
            plan_path.parent / item["original_processed"] / "snapshots.parquet",
            columns=["tick", "steamid", *FIELDS],
        )
        new = pl.read_parquet(
            plan_path.parent / item["new_processed"] / "snapshots.parquet",
            columns=["tick", "steamid", *FIELDS],
        )
        key = ["tick", "steamid"]
        pairs = old.join(new, on=key, suffix="_new", validate="1:1")
        alive = pairs.filter(pl.col("is_alive") & pl.col("is_alive_new"))
        removed = old.join(new.select(key), on=key, how="anti").height
        added = new.join(old.select(key), on=key, how="anti").height
        all_changes, living_changes = {}, {}
        for field in FIELDS:
            expression = (~pl.col(field).eq_missing(pl.col(field + "_new"))).sum()
            all_changes[field] = pairs.select(expression).item()
            living_changes[field] = alive.select(expression).item()
            total["changed_all"][field] += all_changes[field]
            total["changed_both_alive"][field] += living_changes[field]
        total["old_rows"] += old.height
        total["new_rows"] += new.height
        total["removed_keys"] += removed
        total["added_keys"] += added
        details.append(
            {
                "match_id": item["match_id"],
                "demo_sha256": item["demo_sha256"],
                "removed_keys": removed,
                "added_keys": added,
                "changed_all": all_changes,
                "changed_both_alive": living_changes,
            }
        )
    report = {
        "scope": "Awpy 3.1.1 vs 3.3.0 on identical demos and requested ticks",
        "recordings": len(details),
        "total": total,
        "sources": details,
    }
    write_json(Path("reports/awpy_3_3_comparison.json"), report)
    print(
        "解析器对照完成：",
        len(details),
        "录制；新增状态",
        total["added_keys"],
        "；减少状态",
        total["removed_keys"],
        "；存活状态变化",
        total["changed_all"]["is_alive"],
    )


if __name__ == "__main__":
    main()

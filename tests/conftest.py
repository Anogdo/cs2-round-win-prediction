from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.dataset import write_json
from cs2_pipeline.parse import clean_snapshots, prepare_rounds, write_outputs
from cs2_pipeline.schema import STATE_SCHEMA


@pytest.fixture
def rounds() -> pl.DataFrame:
    return prepare_rounds(
        pl.DataFrame(
            {
                "round_num": [12, 13, 25],
                "start_tick": [0, 1000, 2000],
                "freeze_end_tick": [100, 1100, 2100],
                "end_tick": [900, 1900, 2900],
                "winner_side": ["terrorist", "counter-terrorist", "terrorist"],
                "is_knife_round": [False, False, False],
            }
        )
    )


@pytest.fixture
def raw_states() -> pl.DataFrame:
    records = []
    for tick in (100, 420, 1100, 1420, 2100, 2420):
        for index in range(10):
            # Sides swap after round 12 and again in overtime, organizations do not.
            t_side = index < 5 if tick < 1000 or tick >= 2000 else index >= 5
            records.append(
                {
                    "tick": tick,
                    "steamid": 76561198000000000 + index,
                    "name": f"p{index}",
                    "side": "terrorist" if t_side else "counter-terrorist",
                    "x": float(index),
                    "y": -float(index),
                    "z": 64.0,
                    "health": 100,
                    "armor": 100,
                    "money": 800,
                    "equipment_value": 3000,
                    "primary_weapon": "ak47",
                    "secondary_weapon": "glock",
                    "has_helmet": True,
                    "has_defuser": False,
                    "has_bomb": index == 0,
                    "smoke_grenades": 1,
                    "flashbangs": 2,
                    "he_grenades": 0,
                    "fire_grenades": 1,
                }
            )
    return pl.DataFrame(records, schema=STATE_SCHEMA)


@pytest.fixture
def roster(raw_states: pl.DataFrame) -> pl.DataFrame:
    return (
        raw_states.unique("steamid")
        .select("steamid", "name")
        .with_columns(
            pl.when(pl.col("steamid") < 76561198000000005)
            .then(pl.lit("Alpha"))
            .otherwise(pl.lit("Beta"))
            .alias("team_clan_name")
        )
    )


@pytest.fixture
def dataset(rounds: pl.DataFrame, raw_states: pl.DataFrame, roster: pl.DataFrame) -> tuple:
    snapshots, players, cleaning = clean_snapshots(raw_states, rounds, roster, pl.DataFrame())
    metadata = {"map_name": "de_mirage", "total_rounds": 3, "cleaning": cleaning}
    return metadata, rounds, players, snapshots


@pytest.fixture
def phase2_manifest(dataset: tuple, tmp_path: Path) -> Path:
    """Three synthetic table sources; these are never presented as .dem files."""
    original, rounds, players, snapshots = dataset
    sources = []
    for index in range(3):
        path = tmp_path / f"source{index}"
        metadata = {
            **original,
            "tick_rate": 64.0,
            "demo_sha256": hashlib.sha256(f"synthetic-table-{index}".encode()).hexdigest(),
        }
        write_outputs(path, metadata, rounds, players, snapshots)
        sources.append(
            {
                "path": path.name,
                "match_id": f"synthetic:{index}",
                "match_date": f"2026-05-0{index + 1}",
                "event": f"synthetic-event-{index}",
            }
        )
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"map_name": "de_mirage", "sources": sources})
    return manifest

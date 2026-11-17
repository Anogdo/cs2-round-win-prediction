"""Synthetic causal-clock/weapon tests plus optional real-file integration checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.context import (
    CONTEXT_FEATURES,
    CONTEXT_SCHEMA,
    PAUSE_COLUMNS,
    collapse_rules,
    derive_context,
    enrich_demo,
    read_context,
)
from cs2_pipeline.dataset import build_dataset, build_map_tables, read_dataset, write_json
from cs2_pipeline.features import (
    build_features,
    feature_columns,
    validate_feature_table,
    weapon_counts,
)
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.schema import PipelineError


@pytest.fixture
def clock_tables() -> tuple:
    moments = pl.DataFrame(
        {"tick": [100, 200, 300, 500], "round_id": [0, 0, 0, 1]},
        schema={"tick": pl.Int64, "round_id": pl.Int64},
    )
    rules = pl.DataFrame(
        {
            "tick": [100, 150, 200, 300, 500],
            "game_time": [100.0, 105.0, 110.0, 120.0, 140.0],
            "round_start_time": [90.0, 90.0, 90.0, 90.0, 130.0],
            "round_duration": [115.0] * 5,
            "engine_bomb_planted": [False, True, True, True, False],
            "freeze": [False] * 5,
            **{c: [False] * 5 for c in PAUSE_COLUMNS},
        }
    )
    bomb = pl.DataFrame(
        {"tick": [150], "round_id": [0], "event": ["finish_plant"], "bombsite": ["A"]},
        schema={"tick": pl.Int64, "round_id": pl.Int64, "event": pl.String, "bombsite": pl.String},
    )
    return moments, rules, bomb


def test_clock_and_plant_use_only_current_and_past_state(clock_tables: tuple) -> None:
    moments, rules, bomb = clock_tables
    context = derive_context(moments, rules, bomb)
    assert context["round_elapsed_game_seconds"].to_list() == [10.0, 20.0, 30.0, 10.0]
    assert context["round_time_remaining_seconds"].to_list() == [105.0, None, None, 105.0]
    assert context["bomb_planted"].to_list() == [0.0, 1.0, 1.0, 0.0]
    assert context["bomb_seconds_since_plant"].to_list() == [None, 5.0, 15.0, None]
    assert context["bomb_time_remaining_seconds"].null_count() == context.height
    later = pl.concat(
        [
            bomb,
            pl.DataFrame(
                {
                    "tick": [400],
                    "round_id": [0],
                    "event": ["defuse"],
                    "bombsite": ["A"],
                },
                schema=bomb.schema,
            ),
        ]
    )
    assert derive_context(moments, rules, later).equals(context)
    # Final round result/ticks are not consulted even if accidentally provided.
    contaminated = moments.with_columns(
        pl.lit(999999).alias("end_tick"), pl.lit("CT").alias("winner_side")
    )
    assert derive_context(contaminated, rules, bomb).equals(context)


def test_c4_duration_must_be_known_at_plant_and_is_latched(clock_tables: tuple) -> None:
    moments, rules, bomb = clock_tables
    changes = pl.DataFrame(
        {"tick": [50, 160], "cvarname": ["mp_c4timer"] * 2, "cvarvalue": ["40", "80"]}
    )
    context = derive_context(moments, rules, bomb, changes)
    assert context["bomb_time_remaining_seconds"].to_list() == [None, 35.0, 25.0, None]
    future_only = changes.filter(pl.col("tick") == 160)
    assert (
        derive_context(moments, rules, bomb, future_only)[
            "bomb_time_remaining_seconds"
        ].null_count()
        == 4
    )


@pytest.mark.parametrize("flag", [True, None])
def test_paused_or_unknown_pause_has_no_fabricated_countdown(
    clock_tables: tuple, flag: bool | None
) -> None:
    moments, rules, bomb = clock_tables
    rules = rules.with_columns(
        pl.when(pl.col("tick") == 200)
        .then(pl.lit(flag, dtype=pl.Boolean))
        .otherwise(pl.col("game_paused"))
        .alias("game_paused")
    )
    row = derive_context(moments, rules, bomb).filter(pl.col("tick") == 200).row(0, named=True)
    assert row["round_elapsed_game_seconds"] is None
    assert row["bomb_seconds_since_plant"] is None
    assert row["is_paused"] == (1.0 if flag is True else None)


def test_global_rule_conflicts_are_not_arbitrarily_deduplicated(clock_tables: tuple) -> None:
    _, rules, _ = clock_tables
    assert collapse_rules(pl.concat([rules, rules])).height == rules.height
    bad = pl.concat([rules, rules.head(1).with_columns(pl.lit(900.0).alias("game_time"))])
    with pytest.raises(PipelineError, match="不一致"):
        collapse_rules(bad)


def make_context(moments: pl.DataFrame) -> pl.DataFrame:
    return moments.with_columns(
        *[
            pl.lit(None if "bomb_time" in c or "since_plant" in c else 0.0, dtype=pl.Float64).alias(
                c
            )
            for c in CONTEXT_FEATURES
        ]
    ).select(*CONTEXT_SCHEMA)


def test_v2_keeps_v1_compatible_and_adds_masked_weapon_features(dataset: tuple) -> None:
    metadata, rounds, _, snapshots = dataset
    metadata = {**metadata, "tick_rate": 64.0, "demo_sha256": "a" * 64}
    identity = {"match_id": "synthetic:1", "event": "synthetic", "match_date": "2026-05-01"}
    moments = snapshots.select("tick", "round_id").unique()
    snapshots = snapshots.with_columns(
        pl.when(pl.col("steamid") == 76561198000000005)
        .then(pl.lit("awp"))
        .otherwise(pl.col("primary_weapon"))
        .alias("primary_weapon"),
    )
    tables, _ = build_map_tables(
        metadata, rounds, snapshots, identity, feature_version=2, context=make_context(moments)
    )
    assert len(feature_columns(1)) == 47 and len(feature_columns(2)) == 96
    assert "elapsed_since_freeze_end_seconds" not in feature_columns(2)
    validate_feature_table(tables["features"], feature_version=2)
    first = (
        tables["features"]
        .join(tables["samples"].select("sample_id", "tick"), on="sample_id")
        .filter(pl.col("tick") == 100)
        .row(0, named=True)
    )
    assert first["ct_alive_primary_awp_count"] == 1.0
    assert first["t_alive_primary_awp_count"] == 0.0
    nodes = tables["nodes"].with_columns(
        pl.when(pl.col("steamid") == 76561198000000005)
        .then(False)
        .otherwise(pl.col("primary_slot_observed"))
        .alias("primary_slot_observed"),
    )
    features = build_features(nodes, tables["samples"], feature_version=2)
    row = (
        features.join(tables["samples"].select("sample_id", "tick"), on="sample_id")
        .filter(pl.col("tick") == 100)
        .row(0, named=True)
    )
    assert row["ct_alive_primary_awp_count"] is None
    with pytest.raises(PipelineError, match="上下文"):
        build_map_tables(metadata, rounds, snapshots, identity, feature_version=2)


def test_weapon_unknown_is_distinct_from_empty_and_unobserved() -> None:
    groups = {"awp": {"awp"}, "empty": set(), "unknown": set()}
    living = [
        {"primary_slot_observed": True, "primary_weapon": "weapon_awp"},
        {"primary_slot_observed": True, "primary_weapon": None},
        {"primary_slot_observed": True, "primary_weapon": "future_weapon"},
    ]
    assert weapon_counts(living, "primary", groups) == {"awp": 1.0, "empty": 1.0, "unknown": 1.0}
    living[0]["primary_slot_observed"] = False
    assert weapon_counts(living, "primary", groups) == dict.fromkeys(groups)


def test_v2_dataset_train_and_json_inference(phase2_manifest: Path, tmp_path: Path) -> None:
    config = json.loads(phase2_manifest.read_text(encoding="utf-8"))
    config["feature_version"] = 2
    for index, entry in enumerate(config["sources"]):
        processed = phase2_manifest.parent / entry["path"]
        metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
        moments = (
            pl.read_parquet(processed / "snapshots.parquet").select("tick", "round_id").unique()
        )
        out = tmp_path / f"context{index}"
        out.mkdir()
        make_context(moments).write_parquet(out / "context.parquet")
        notes = {
            "context_schema_version": 1,
            "demo_sha256": metadata["demo_sha256"],
            "context_file_sha256": sha256_file(out / "context.parquet"),
            "phase1_file_sha256": {
                f: sha256_file(processed / f) for f in ("rounds.parquet", "snapshots.parquet")
            },
        }
        write_json(out / "context_metadata.json", notes)
        entry["context_path"] = out.name
    write_json(phase2_manifest, config)
    dataset = tmp_path / "v2_dataset"
    build_dataset(phase2_manifest, dataset)
    metadata, tables = read_dataset(dataset)
    assert metadata["feature_version"] == 2 and len(metadata["feature_columns"]) == 96
    assert metadata["sources"][0]["context_files_sha256"]
    if importlib.util.find_spec("sklearn") is None:
        pytest.skip("ML 未安装；V2 训练集已独立验证")
    from cs2_pipeline.train import train_baseline

    report = train_baseline(dataset, tmp_path / "model")
    assert report["feature_count"] == 96 and report["independent_evaluation_performed"]
    model = json.loads((tmp_path / "model/model.json").read_text(encoding="utf-8"))
    assert predict_probabilities(model, tables["features"])["p_ct_win"].is_between(0, 1).all()


@pytest.mark.integration
def test_real_context_uses_engine_clock_and_retains_unknown_fuse(tmp_path: Path) -> None:
    if importlib.util.find_spec("demoparser2") is None or not Path("data/raw/match.dem").is_file():
        pytest.skip("需要真实 Demo 和 --extra context")
    notes = enrich_demo(
        Path("data/raw/match.dem"), Path("data/processed/match"), tmp_path / "context"
    )
    metadata = json.loads(Path("data/processed/match/metadata.json").read_text(encoding="utf-8"))
    context = read_context(
        tmp_path / "context", Path("data/processed/match"), metadata["demo_sha256"]
    )
    assert context.height == 295 and notes["non_null_counts"]["bomb_seconds_since_plant"] == 26
    assert context.filter(pl.col("tick") == 1274)["round_time_remaining_seconds"].item() == 115.0
    with pytest.raises(PipelineError, match="SHA256"):
        read_context(tmp_path / "context", Path("data/processed/match"), "f" * 64)


@pytest.mark.integration
def test_real_corpus_is_chronologically_disjoint_and_v1_v2_share_labels() -> None:
    first, second = Path("data/datasets/mirage_rio_v1"), Path("data/datasets/mirage_rio_v2")
    if not (first / "dataset.json").is_file() or not (second / "dataset.json").is_file():
        pytest.skip("没有真实 Rio 语料，不能声称已完成独立比赛评估")
    m1, t1 = read_dataset(first)
    m2, t2 = read_dataset(second)
    assert m1["sample_count"] == m2["sample_count"] == 3038
    assert m1["round_count"] == m2["round_count"] == 173
    assert t1["labels"].sort("sample_id").equals(t2["labels"].sort("sample_id"))
    assert t1["splits"].sort("sample_id").equals(t2["splits"].sort("sample_id"))
    assert m2["split_policy"]["match_counts"] == {"train": 7, "validation": 1, "test": 1}

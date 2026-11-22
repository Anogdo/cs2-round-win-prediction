"""Synthetic table tests; only the marked real test reads a professional Demo dataset."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.dataset import (
    build_dataset,
    build_map_tables,
    read_dataset,
    source_identity,
    validate_dataset_tables,
    write_json,
)
from cs2_pipeline.features import FEATURE_COLUMNS, build_features, validate_feature_table
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.split import chronological_split


@pytest.fixture
def phase2_tables(dataset: tuple) -> dict[str, pl.DataFrame]:
    metadata, rounds, _, snapshots = dataset
    tables, _ = build_map_tables(
        {**metadata, "tick_rate": 64.0, "demo_sha256": "a" * 64},
        rounds,
        snapshots,
        {"match_id": "synthetic:1", "match_date": "2026-05-01", "event": "synthetic-event"},
    )
    return tables


def test_dataset_roundtrip_and_complete_keys(phase2_manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "built"
    metadata = build_dataset(phase2_manifest, out)
    reread, tables = read_dataset(out)
    assert reread == metadata
    assert metadata["sample_count"] == 18 and metadata["node_count"] == 180
    assert metadata["round_count"] == 9 and metadata["split_policy"]["evaluation_ready"]
    assert metadata["split_policy"]["match_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert len(FEATURE_COLUMNS) == 47
    assert tables["nodes"]["steamid"].dtype == pl.UInt64
    assert tables["labels"]["y_ct_win"].dtype == pl.UInt8
    assert set(tables["labels"]["y_ct_win"]) == {0, 1}
    assert "y_ct_win" not in tables["nodes"].columns
    assert "winner_side" not in tables["features"].columns
    assert sorted(tables["samples"]["elapsed_since_freeze_end_seconds"].unique()) == [0.0, 5.0]
    with pytest.raises(PipelineError, match="已存在"):
        build_dataset(phase2_manifest, out)


def test_features_ignore_labels_identity_and_final_ticks(phase2_tables: dict) -> None:
    nodes, samples = phase2_tables["nodes"], phase2_tables["samples"]
    before = build_features(nodes, samples)
    contaminated_nodes = nodes.with_columns(
        pl.lit("CT").alias("winner_side"),
        pl.lit(9999999).alias("end_tick"),
        pl.lit("completely different name").alias("name"),
        pl.lit("another organization").alias("team"),
        (pl.col("steamid") + 1).alias("steamid"),
    )
    contaminated_samples = samples.with_columns(
        pl.lit(1).alias("y_ct_win"), pl.lit(900).alias("round_num")
    )
    assert before.equals(build_features(contaminated_nodes, contaminated_samples))
    with pytest.raises(PipelineError, match="白名单"):
        validate_feature_table(before.with_columns(pl.lit(1.0).alias("winner_side")))


def test_dead_state_is_masked_and_missing_living_values_stay_null(phase2_tables: dict) -> None:
    nodes, samples = phase2_tables["nodes"], phase2_tables["samples"]
    sid = nodes["steamid"][0]
    dead = nodes.with_columns(
        pl.when(pl.col("steamid") == sid)
        .then(False)
        .otherwise(pl.col("is_alive"))
        .alias("is_alive"),
    )
    before = build_features(dead, samples)
    enormous_corpse = dead.with_columns(
        *[
            pl.when(pl.col("steamid") == sid).then(1e15).otherwise(pl.col(c)).alias(c)
            for c in ("x", "y", "z", "money", "equipment_value")
        ]
    )
    assert before.equals(build_features(enormous_corpse, samples))
    missing = nodes.with_columns(pl.lit(None, dtype=pl.Int64).alias("money"))
    features = build_features(missing, samples)
    assert features["ct_alive_money_sum"].null_count() == features.height
    assert features["ct_minus_t_alive_money_sum"].null_count() == features.height


def test_missing_player_excludes_entire_timepoint(dataset: tuple) -> None:
    metadata, rounds, _, snapshots = dataset
    incomplete = snapshots.filter(
        ~((pl.col("tick") == 100) & (pl.col("steamid") == 76561198000000000))
    )
    tables, counts = build_map_tables(
        {**metadata, "tick_rate": 64.0, "demo_sha256": "a" * 64},
        rounds,
        incomplete,
        {"match_id": "synthetic:1", "match_date": "2026-05-01", "event": "synthetic"},
    )
    assert counts["incomplete_timepoints_excluded"] == 1
    assert tables["samples"].height == 5 and tables["nodes"].height == 50


def test_tick_rate_fallback_is_not_accepted_for_training(dataset: tuple) -> None:
    metadata, rounds, _, snapshots = dataset
    with pytest.raises(PipelineError, match="tick_rate"):
        build_map_tables({**metadata, "tick_rate": None}, rounds, snapshots, {})


def test_duplicate_demo_is_deduplicated_and_alias_conflict_rejected(
    phase2_manifest: Path,
    tmp_path: Path,
) -> None:
    config = json.loads(phase2_manifest.read_text(encoding="utf-8"))
    config["sources"].append(dict(config["sources"][0]))
    write_json(phase2_manifest, config)
    metadata = build_dataset(phase2_manifest, tmp_path / "dedup")
    assert metadata["sample_count"] == 18
    assert metadata["excluded_sources"][0]["reason"] == "duplicate_demo_sha256"
    config["sources"][-1]["match_id"] = "different-match"
    write_json(phase2_manifest, config)
    with pytest.raises(PipelineError, match="SHA256"):
        build_dataset(phase2_manifest, tmp_path / "conflict")


def test_reserved_event_is_excluded_before_reading_files(
    phase2_manifest: Path, tmp_path: Path
) -> None:
    config = json.loads(phase2_manifest.read_text(encoding="utf-8"))
    config["sources"].append({"path": "does-not-exist", "event": "PGL Masters Bucharest 2026"})
    write_json(phase2_manifest, config)
    metadata = build_dataset(phase2_manifest, tmp_path / "heldout")
    assert metadata["excluded_sources"][0]["reason"] == "reserved_test_event"


@pytest.mark.parametrize("invalid", ["", "unknown", "2026-99-99", "20260501"])
def test_source_dates_are_explicit_and_canonical(invalid: str) -> None:
    with pytest.raises(PipelineError):
        source_identity(
            {"match_id": "synthetic:1", "event": "synthetic", "match_date": invalid},
            {"demo_sha256": "a" * 64},
        )


def test_external_source_prevents_map_level_match_alias() -> None:
    metadata = {
        "demo_sha256": "a" * 64,
        "external_source": {
            "sha256": "a" * 64,
            "event": "Synthetic Event",
            "match_date": "2026-05-01",
            "match_url": "https://www.hltv.org/matches/123/example",
        },
    }
    entry = {"match_id": "hltv:123-map-1", "event": "synthetic-event", "match_date": "2026-05-01"}
    with pytest.raises(PipelineError, match="不能按地图拆组"):
        source_identity(entry, metadata)
    entry["match_id"] = "hltv:123"
    assert source_identity(entry, metadata)["match_id"] == "hltv:123"


def test_split_keeps_same_match_maps_and_dates_together() -> None:
    sources = [
        {"match_id": "match-a", "match_date": "2026-05-01", "event": "event-a"},
        {"match_id": "match-a", "match_date": "2026-05-01", "event": "event-a"},
        {"match_id": "match-b", "match_date": "2026-05-01", "event": "event-a"},
        {"match_id": "match-c", "match_date": "2026-05-02", "event": "event-b"},
        {"match_id": "match-d", "match_date": "2026-05-03", "event": "event-c"},
    ]
    assignment, summary = chronological_split(sources)
    assert assignment == {
        "match-a": "train",
        "match-b": "train",
        "match-c": "validation",
        "match-d": "test",
    }
    assert summary["match_count"] == 4


def test_overlapping_event_windows_merge_before_temporal_split() -> None:
    sources = [
        {"match_id": "a", "match_date": "2026-05-01", "event": "event-a"},
        {"match_id": "b", "match_date": "2026-05-04", "event": "event-a"},
        {"match_id": "c", "match_date": "2026-05-02", "event": "event-b"},
        {"match_id": "d", "match_date": "2026-05-05", "event": "event-c"},
        {"match_id": "e", "match_date": "2026-05-06", "event": "event-d"},
    ]
    assignment, summary = chronological_split(sources, group_by="event")
    assert summary["evaluation_ready"]
    assert assignment["a"] == assignment["b"] == assignment["c"] == "train"
    assert assignment["d"] == "validation" and assignment["e"] == "test"


def test_insufficient_groups_never_fall_back_to_random_rows() -> None:
    assignment, summary = chronological_split(
        [
            {"match_id": "a", "match_date": "2026-05-01", "event": "event-a"},
            {"match_id": "b", "match_date": "2026-05-02", "event": "event-b"},
        ]
    )
    assert assignment == {"a": "train", "b": "train"}
    assert summary["evaluation_ready"] is False


def test_tampered_files_and_cross_match_split_are_rejected(
    phase2_manifest: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "dataset"
    build_dataset(phase2_manifest, out)
    metadata, tables = read_dataset(out)
    first = tables["splits"].filter(pl.col("split") == "train")["sample_id"][0]
    tables["splits"] = tables["splits"].with_columns(
        pl.when(pl.col("sample_id") == first)
        .then(pl.lit("test"))
        .otherwise(pl.col("split"))
        .alias("split"),
    )
    with pytest.raises(PipelineError, match="泄漏"):
        validate_dataset_tables(tables, metadata)
    (out / "features.parquet").write_bytes(b"intentional corruption for hash validation")
    with pytest.raises(PipelineError, match="已改变"):
        read_dataset(out)


def test_historical_development_split_cannot_include_fit_rows(
    phase2_manifest: Path, tmp_path: Path
) -> None:
    out = tmp_path / "dataset"
    build_dataset(phase2_manifest, out)
    metadata, tables = read_dataset(out)
    metadata["split_policy"] = {
        "group_by": "event",
        "method": "historical_validation_only",
        "evaluation_ready": False,
    }
    with pytest.raises(PipelineError, match="only the validation split"):
        validate_dataset_tables(tables, metadata)
    tables["splits"] = tables["splits"].with_columns(pl.lit("validation").alias("split"))
    validate_dataset_tables(tables, metadata)


@pytest.mark.integration
def test_real_phase1_dataset_builds_without_label_leakage(tmp_path: Path) -> None:
    source = Path("data/processed/match")
    if not (source / "metadata.json").is_file():
        pytest.skip("没有真实 Phase 1 输出，不能声称真实训练集构建成功")
    manifest = tmp_path / "real_sources.json"
    config = json.loads(Path("configs/phase2.json").read_text(encoding="utf-8"))
    config["sources"][0]["path"] = str(source.resolve())
    write_json(manifest, config)
    metadata = build_dataset(manifest, tmp_path / "real_dataset")
    _, tables = read_dataset(tmp_path / "real_dataset")
    assert metadata["sample_count"] == 295 and metadata["node_count"] == 2950
    assert not metadata["split_policy"]["evaluation_ready"]
    # Awpy 3.3 supplies dead-player snapshots that older parses recover from events.
    assert tables["nodes"].filter(~pl.col("is_alive")).height == 540
    assert tables["labels"]["y_ct_win"].is_in([0, 1]).all()

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.parse import (
    associate_rounds,
    clean_snapshots,
    main,
    prepare_rounds,
    sampling_ticks,
    write_outputs,
)
from cs2_pipeline.schema import PipelineError, exact_ids
from cs2_pipeline.validate import validate_directory, validate_frames


def test_combat_boundaries(rounds: pl.DataFrame) -> None:
    states = pl.DataFrame({"tick": [-1, 0, 99, 100, 899, 900, 999, 1099, 1100, 1899, 1900]})
    output = associate_rounds(states, rounds)
    assert output["tick"].to_list() == [100, 899, 1100, 1899]
    assert output["round_num"].to_list() == [12, 12, 13, 13]
    assert "winner_side" not in output.columns


def test_sampling_round_alignment(rounds: pl.DataFrame) -> None:
    assert sampling_ticks(rounds, 64, 5) == [100, 420, 740, 1100, 1420, 1740, 2100, 2420, 2740]
    assert sampling_ticks(rounds, 64, 100) == [100, 1100, 2100]


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf"), 0.001])
def test_invalid_interval(rounds: pl.DataFrame, interval: float) -> None:
    with pytest.raises(PipelineError):
        sampling_ticks(rounds, 64, interval)


def test_identity_persists_across_halftime_and_overtime(dataset: tuple) -> None:
    _, rounds, players, snapshots = dataset
    assert rounds["round_num"].to_list() == [12, 13, 25]
    player = snapshots.filter(pl.col("steamid") == 76561198000000000)
    assert player["side"].to_list() == ["T", "T", "CT", "CT", "T", "T"]
    assert player["team"].unique().to_list() == ["Alpha"]
    assert players["steamid"].dtype == snapshots["steamid"].dtype == pl.UInt64
    assert "side" not in players.columns


def test_exact_duplicates_only(
    raw_states: pl.DataFrame, rounds: pl.DataFrame, roster: pl.DataFrame
) -> None:
    duplicated = pl.concat([raw_states, raw_states.head(1)])
    clean, _, report = clean_snapshots(duplicated, rounds, roster, pl.DataFrame())
    assert clean.height == raw_states.height
    assert report["exact_duplicates_removed"] == 1
    conflict = pl.concat(
        [
            raw_states,
            raw_states.head(1).with_columns(pl.lit(1, dtype=pl.Int64).alias("health")),
        ]
    )
    with pytest.raises(PipelineError, match="冲突状态"):
        clean_snapshots(conflict, rounds, roster, pl.DataFrame())


def test_optional_fields_remain_null(
    raw_states: pl.DataFrame, rounds: pl.DataFrame, roster: pl.DataFrame
) -> None:
    clean, players, report = clean_snapshots(
        raw_states.drop("money", "has_helmet"),
        rounds,
        roster.drop("team_clan_name"),
        pl.DataFrame(),
    )
    assert clean["money"].null_count() == clean.height
    assert clean["has_helmet"].null_count() == clean.height
    assert players["team"].null_count() == players.height
    assert report["missing_snapshot_columns"] == ["has_helmet", "money"]


def test_team_conflict_kept_as_null(
    raw_states: pl.DataFrame, rounds: pl.DataFrame, roster: pl.DataFrame
) -> None:
    roster = pl.concat(
        [
            roster,
            roster.head(1).with_columns(pl.lit("Other organization").alias("team_clan_name")),
        ]
    )
    _, players, _ = clean_snapshots(raw_states, rounds, roster, pl.DataFrame())
    conflicted = players.filter(pl.col("team_identity_conflict"))
    assert conflicted.height == 1
    assert conflicted["team"].item() is None
    assert len(conflicted["team_names"].to_list()[0]) == 2


def test_death_does_not_resurrect_or_cross_rounds(
    raw_states: pl.DataFrame, rounds: pl.DataFrame, roster: pl.DataFrame
) -> None:
    kills = pl.DataFrame(
        {"tick": [300], "victim_steamid": [76561198000000000]},
        schema={"tick": pl.Int64, "victim_steamid": pl.UInt64},
    )
    clean, _, report = clean_snapshots(raw_states, rounds, roster, kills)
    player = clean.filter(pl.col("steamid") == 76561198000000000)
    assert player["is_alive"].to_list() == [True, False, True, True, True, True]
    assert report["death_health_conflicts"] == 1
    dead = raw_states.with_columns(
        pl.when(pl.col("tick") == 100).then(0).otherwise(pl.col("health")).alias("health")
    )
    clean, _, _ = clean_snapshots(dead, rounds, roster, pl.DataFrame())
    assert not clean.filter(pl.col("tick") == 100)["is_alive"].any()


def test_float_steamid_is_rejected() -> None:
    with pytest.raises(PipelineError, match="浮点"):
        exact_ids(pl.DataFrame({"steamid": [76561198000000001.0]}))
    assert exact_ids(pl.DataFrame({"steamid": ["76561198000000001"]}))["steamid"].item() == (
        76561198000000001
    )


def test_missing_dead_identity_uses_only_prior_death_evidence(
    raw_states: pl.DataFrame,
    rounds: pl.DataFrame,
    roster: pl.DataFrame,
) -> None:
    sid = 76561198000000000
    absent = raw_states.filter(~((pl.col("steamid") == sid) & (pl.col("tick") == 420)))
    kills = pl.DataFrame(
        {"tick": [300], "victim_steamid": [sid]},
        schema={"tick": pl.Int64, "victim_steamid": pl.UInt64},
    )
    clean, _, report = clean_snapshots(absent, rounds, roster, kills)
    restored = clean.filter((pl.col("steamid") == sid) & (pl.col("tick") == 420))
    assert restored.height == 1 and restored["state_source"].item() == "death_event"
    assert restored["is_alive"].item() is False and restored["death_observed"].item() is True
    assert restored["x"].item() is None and restored["money"].item() is None
    assert report["confirmed_dead_rows_restored"] == 1
    # A death in the future must not fill a missing observation in the past.
    later_kill = kills.with_columns(pl.lit(500, dtype=pl.Int64).alias("tick"))
    before, _, _ = clean_snapshots(absent, rounds, roster, later_kill)
    assert before.filter((pl.col("steamid") == sid) & (pl.col("tick") == 420)).is_empty()
    disconnected, _, _ = clean_snapshots(absent, rounds, roster, pl.DataFrame())
    assert disconnected.height == absent.height


def test_living_player_with_unknown_id_fails(
    raw_states: pl.DataFrame,
    rounds: pl.DataFrame,
    roster: pl.DataFrame,
) -> None:
    bad = raw_states.with_columns(pl.lit(None, dtype=pl.UInt64).alias("steamid"))
    with pytest.raises(PipelineError, match="缺少 Steam ID"):
        clean_snapshots(bad, rounds, roster, pl.DataFrame())


def test_abnormal_rounds_preserve_original_numbers(rounds: pl.DataFrame) -> None:
    raw = rounds.drop("is_valid_round", "invalid_reason").with_columns(
        pl.when(pl.col("round_num") == 13)
        .then(True)
        .otherwise(pl.col("is_knife_round"))
        .alias("is_knife_round")
    )
    output = prepare_rounds(raw)
    assert output["round_num"].to_list() == [12, 13, 25]
    assert output["is_valid_round"].to_list() == [True, False, True]
    assert output["invalid_reason"].to_list()[1] == "knife_round"


def test_duplicate_round_numbers_are_distinct_instances(
    rounds: pl.DataFrame, raw_states: pl.DataFrame, roster: pl.DataFrame
) -> None:
    repeated = prepare_rounds(rounds.with_columns(pl.lit(1).alias("round_num")))
    kills = pl.DataFrame(
        {"tick": [300], "victim_steamid": [76561198000000000]},
        schema={"tick": pl.Int64, "victim_steamid": pl.UInt64},
    )
    clean, _, _ = clean_snapshots(raw_states, repeated, roster, kills)
    assert repeated["round_id"].n_unique() == 3
    assert not repeated["is_valid_round"].any()
    assert clean.filter(pl.col("tick") == 1100)["is_alive"].all()


def test_overlapping_rounds_are_not_arbitrarily_assigned(rounds: pl.DataFrame) -> None:
    overlap = rounds.with_columns(
        pl.when(pl.col("round_num") == 13)
        .then(500)
        .otherwise(pl.col("start_tick"))
        .alias("start_tick")
    )
    overlap = prepare_rounds(overlap)
    assert overlap["is_valid_round"].to_list() == [False, False, True]
    assert associate_rounds(pl.DataFrame({"tick": [800, 1100]}), overlap).is_empty()


def test_missing_boundary_stays_invalid(rounds: pl.DataFrame) -> None:
    raw = rounds.with_columns(pl.lit(None, dtype=pl.Int64).alias("freeze_end_tick"))
    output = prepare_rounds(raw)
    assert not output["is_valid_round"].any()
    assert sampling_ticks(output, 64, 5) == []


def test_validation_and_parquet_roundtrip(dataset: tuple, tmp_path: Path) -> None:
    metadata, rounds, players, snapshots = dataset
    target = tmp_path / "match"
    report = write_outputs(target, metadata, rounds, players, snapshots)
    assert report["passed"] and report["files_readable"]
    assert report["valid_sample_tick_count"] == 6
    assert set(p.name for p in target.iterdir()) == {
        "metadata.json",
        "rounds.parquet",
        "players.parquet",
        "snapshots.parquet",
    }
    assert pl.read_parquet(target / "snapshots.parquet")["steamid"].to_list() == (
        snapshots["steamid"].to_list()
    )
    assert json.loads((target / "metadata.json").read_text(encoding="utf-8"))["validation"][
        "passed"
    ]
    with pytest.raises(PipelineError, match="已存在"):
        write_outputs(target, metadata, rounds, players, snapshots)


@pytest.mark.parametrize(
    ("column", "value", "code"),
    [
        ("health", -1, "health"),
        ("health", 101, "health"),
        ("health", None, "health"),
        ("x", float("nan"), "coordinates"),
        ("y", float("inf"), "coordinates"),
        ("z", None, "coordinates"),
        ("side", "Alpha", "side"),
        ("round_num", 99, "round_mapping"),
        ("tick", 900, "round_mapping"),
        ("is_alive", False, "alive"),
        ("team", "Wrong organization", "team_mapping"),
        ("steamid", 123, "steamid"),
    ],
)
def test_corrupt_data_fails(dataset: tuple, column: str, value: object, code: str) -> None:
    metadata, rounds, players, snapshots = dataset
    corrupt = snapshots.with_columns(pl.lit(value, dtype=snapshots.schema[column]).alias(column))
    report = validate_frames(metadata, rounds, players, corrupt)
    assert not report["passed"]
    assert code in {item["code"] for item in report["errors"]}


def test_empty_and_duplicate_snapshots_fail(dataset: tuple) -> None:
    metadata, rounds, players, snapshots = dataset
    assert not validate_frames(metadata, rounds, players, snapshots.head(0))["passed"]
    duplicate = pl.concat([snapshots, snapshots.head(1)])
    report = validate_frames(metadata, rounds, players, duplicate)
    assert "duplicate_snapshot" in {item["code"] for item in report["errors"]}


def test_too_many_players_and_labels_fail(dataset: tuple) -> None:
    metadata, rounds, players, snapshots = dataset
    report = validate_frames(metadata, rounds, players, pl.concat([snapshots, snapshots]))
    assert "player_count" in {item["code"] for item in report["errors"]}
    leaked = snapshots.with_columns(pl.lit("T").alias("winner_side"))
    assert not validate_frames(metadata, rounds, players, leaked)["passed"]


def test_missing_or_corrupt_output_is_actionable(tmp_path: Path, dataset: tuple) -> None:
    assert not validate_directory(tmp_path)["passed"]
    metadata, rounds, players, snapshots = dataset
    write_outputs(tmp_path / "match", metadata, rounds, players, snapshots)
    (tmp_path / "match" / "players.parquet").write_text("not parquet", encoding="utf-8")
    report = validate_directory(tmp_path / "match")
    assert not report["passed"]
    assert "重新解析" in report["errors"][0]["message"]
    (tmp_path / "match" / "metadata.json").write_text("[]", encoding="utf-8")
    assert not validate_directory(tmp_path / "match")["passed"]


def test_missing_demo_cli_has_no_traceback(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert main(["--demo", str(tmp_path / "missing.dem")]) == 1
    captured = capsys.readouterr()
    assert "data/raw/match.dem" in captured.err and "Traceback" not in captured.err


def test_invalid_demo_cli_is_actionable(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    # Deliberately invalid bytes test rejection; this is never treated as a real Demo.
    path = tmp_path / "invalid.dem"
    path.write_bytes(b"invalid fixture for error handling")
    assert main(["--demo", str(path), "--out", str(tmp_path / "out")]) == 1
    assert "Awpy" in capsys.readouterr().err


def test_validation_rejects_float_schema(dataset: tuple) -> None:
    metadata, rounds, players, snapshots = dataset
    snapshots = snapshots.with_columns(pl.col("steamid").cast(pl.Float64))
    assert not validate_frames(metadata, rounds, players, snapshots)["passed"]

"""Real-file integration test; never substitutes synthetic data for a Demo."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.parse import parse_demo, sha256_file
from cs2_pipeline.validate import validate_directory


@pytest.mark.integration
def test_real_demo_end_to_end(tmp_path: Path) -> None:
    demo = Path(os.environ.get("CS2_TEST_DEMO", "data/raw/match.dem"))
    if not demo.is_file():
        pytest.skip("没有真实 Demo；下载到 data/raw/match.dem 或设置 CS2_TEST_DEMO")
    source_file = Path("docs/demo_source.json")
    source = json.loads(source_file.read_text(encoding="utf-8"))
    pinned_fixture = sha256_file(demo) == source["sha256"]
    target = tmp_path / "real_match"
    report = parse_demo(demo, target, interval=5, source=source_file if pinned_fixture else None)
    assert report["passed"], report["errors"]
    assert validate_directory(target)["passed"]
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    states = pl.read_parquet(target / "snapshots.parquet")
    rounds = pl.read_parquet(target / "rounds.parquet")
    assert states["steamid"].dtype == pl.UInt64
    assert "winner_side" not in states.columns
    assert set(states["side"].unique()) == {"T", "CT"}
    assert metadata["sampling_interval_seconds"] == 5
    # Strong fixture-specific checks are enabled only for the pinned SHA256.
    if metadata["demo_sha256"] == source["sha256"]:
        assert metadata["external_source"]["event"] == "IEM Atlanta 2026"
        assert metadata["map_name"] == "de_mirage"
        assert report["round_count"] == 18 and report["unique_players"] == 10
        assert report["sample_tick_count"] == 295
        assert report["snapshot_count"] == 2950
        assert report["observed_snapshot_count"] + report["event_derived_snapshot_count"] == 2950
        assert report["valid_sample_tick_count"] == 295
        derived = states.filter(pl.col("state_source") == "death_event")
        assert not derived["is_alive"].any()
        assert derived["x"].null_count() == derived.height
        # Awpy 3.3 can retain dead players directly; older parsers need event-derived rows.
        observed_dead = states.filter((pl.col("state_source") == "snapshot") & ~pl.col("is_alive"))
        assert observed_dead.height + derived.height > 0
        # Independent published Mirage score, with sides resolved at each round start.
        beginning = states.sort("tick").group_by("round_id", "team").first()
        winners = beginning.join(rounds.select("round_id", "winner_side"), on="round_id")
        score = winners.filter(pl.col("side") == pl.col("winner_side")).group_by("team").len()
        assert dict(score.iter_rows()) == source["expected_map_score"]

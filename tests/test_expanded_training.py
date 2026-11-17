"""Regression checks for provenance, parser consistency and event-level isolation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from cs2_pipeline.dataset import build_dataset, read_dataset, write_json
from cs2_pipeline.holdout import check_published_score, verify_frozen
from cs2_pipeline.parse import sha256_file, write_outputs
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import train_baseline
from scripts.import_user_hltv import extract_sources
from scripts.reparse_consistent import reparse


def test_declared_event_split_is_enforced_before_dataset_is_published(
    phase2_manifest: Path, tmp_path: Path
) -> None:
    config = json.loads(phase2_manifest.read_text(encoding="utf-8"))
    config["split_expected"] = {
        "group_by": "event",
        "train": ["synthetic-event-0"],
        "validation": ["synthetic-event-1"],
        "test": ["synthetic-event-2"],
    }
    write_json(phase2_manifest, config)
    out = tmp_path / "by_event"
    build_dataset(phase2_manifest, out, group_by="event")
    metadata, _ = read_dataset(out)
    assert metadata["source_manifest_sha256"] == sha256_file(phase2_manifest)
    assert metadata["split_policy"]["group_by"] == "event"
    with pytest.raises(PipelineError, match="按赛事划分"):
        build_dataset(phase2_manifest, tmp_path / "wrong_group", group_by="match")
    config["split_expected"]["test"] = ["synthetic-event-0"]
    write_json(phase2_manifest, config)
    with pytest.raises(PipelineError, match="预先声明"):
        build_dataset(phase2_manifest, tmp_path / "wrong_event", group_by="event")
    assert not (tmp_path / "wrong_event").exists()


def test_archive_extraction_rejects_path_and_option_injection(tmp_path: Path) -> None:
    archive = tmp_path / "source.rar"
    archive.write_bytes(b"test placeholder, not a real archive")
    manifest = tmp_path / "sources.json"
    source = {
        "event": "example-event",
        "match_url": "https://www.hltv.org/matches/1/example",
        "archive": archive.name,
        "archive_sha256": sha256_file(archive),
        "entry": "../fake-mirage.dem",
        "match_id": "hltv:1",
    }
    for invalid in ("../fake-mirage.dem", "-evil-mirage.dem"):
        source["entry"] = invalid
        write_json(manifest, {"maps": [source]})
        with pytest.raises(PipelineError, match="无路径"):
            extract_sources(manifest, Path("unrar"), tmp_path / "raw", tmp_path / "local.json")


def test_reparse_stops_on_changed_round_boundaries(
    dataset: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original, rounds, players, snapshots = dataset
    demo = tmp_path / "synthetic.dem"
    demo.write_bytes(b"test only; parser is mocked")
    checksum = hashlib.sha256(demo.read_bytes()).hexdigest()
    old = tmp_path / "old"
    metadata = {
        **original,
        "map_name": "de_mirage",
        "demo_sha256": checksum,
        "tick_rate": 64.0,
        "sampling_interval_seconds": 5.0,
    }
    write_outputs(old, metadata, rounds, players, snapshots)
    output = tmp_path / "new"
    plan = tmp_path / "plan.json"
    write_json(
        plan,
        {
            "awpy_version": "3.3.0",
            "report_path": "reparse.json",
            "sources": [
                {
                    "match_id": "synthetic:1",
                    "event": "synthetic",
                    "demo_path": demo.name,
                    "demo_sha256": checksum,
                    "original_processed": old.name,
                    "original_metadata_sha256": sha256_file(old / "metadata.json"),
                    "original_rounds_sha256": sha256_file(old / "rounds.parquet"),
                    "new_processed": output.name,
                }
            ],
        },
    )
    monkeypatch.setattr("scripts.reparse_consistent.version", lambda _: "3.3.0")

    def fake_parse(raw: Path, out: Path, interval: float, source: Path | None) -> dict:
        changed = rounds.with_columns((pl.col("end_tick") + 1).alias("end_tick"))
        write_outputs(out, metadata, changed, players, snapshots)
        return {"passed": True, "round_count": 3, "valid_sample_tick_count": 6, "warnings": []}

    monkeypatch.setattr("scripts.reparse_consistent.parse_demo", fake_parse)
    with pytest.raises(PipelineError, match="回合边界或胜方不同"):
        reparse(plan)


def test_score_verification_keeps_team_identity_through_side_change() -> None:
    rounds = pl.DataFrame({"round_id": [1, 2], "winner_side": ["T", "CT"]})
    snapshots = pl.DataFrame(
        {
            "round_id": [1, 1, 2, 2],
            "tick": [100, 100, 200, 200],
            "team": ["Team Falcons", "TYLOO", "Team Falcons", "TYLOO"],
            "side": ["T", "CT", "CT", "T"],
        }
    )
    check_published_score(rounds, snapshots, {"Falcons": 2, "TYLOO": 0})
    with pytest.raises(PipelineError, match="比分与公开比分不一致"):
        check_published_score(rounds, snapshots, {"Falcons": 1, "TYLOO": 1})


@pytest.mark.parametrize("smoke_test", [False, True])
def test_holdout_is_never_accepted_as_training_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke_test: bool
) -> None:
    monkeypatch.setattr(
        "cs2_pipeline.train.read_dataset",
        lambda _: ({"split_policy": {"method": "holdout_only"}}, {}),
    )
    with pytest.raises(PipelineError, match="留出数据集不能用于训练"):
        train_baseline(tmp_path / "holdout", tmp_path / "output", smoke_test=smoke_test)
    assert not (tmp_path / "output").exists()


def test_changed_frozen_model_is_rejected(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    (tmp_path / "model.json").write_text('{"parameters": "changed"}', encoding="utf-8")
    plan = configs / "plan.json"
    write_json(plan, {"frozen_model": "model.json", "frozen_model_sha256": "0" * 64})
    with pytest.raises(PipelineError, match="冻结模型 SHA256 不符"):
        verify_frozen(plan, tmp_path / "training")

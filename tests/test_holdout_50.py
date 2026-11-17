"""Checks for batch provenance, archive boundaries and team-aware score verification."""

from pathlib import Path
from subprocess import CompletedProcess

import polars as pl
import pytest

from cs2_pipeline.schema import PipelineError
from scripts.import_holdout_50 import find_archive, list_entries, published_score, write_progress


def test_same_teams_final_and_group_archives_remain_distinct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("scripts.import_holdout_50.RAW", tmp_path)
    prefix = "thunderpick-world-championship-2026-furia-vs-betboom-"
    (tmp_path / (prefix + "bo3-group.rar")).touch()
    (tmp_path / (prefix + "bo5-final.rar")).touch()
    item = {
        "event": "thunderpick-world-championship-2026",
        "matchup": "FURIA vs BETBOOM",
        "match_id": "hltv:2399417",
    }
    assert "bo5-final" in find_archive(item).name
    item["match_id"] = "hltv:2399387"
    assert "bo3-group" in find_archive(item).name
    item["match_id"] = "hltv:other"
    with pytest.raises(PipelineError, match="无法唯一匹配"):
        find_archive(item)


def test_archive_parts_must_be_contiguous_and_path_free(monkeypatch: pytest.MonkeyPatch) -> None:
    output = "x-m3-mirage-p1.dem\nx-m3-mirage-p2.dem\nx-m1-nuke.dem\n"

    def fake_run(*args: object, **kwargs: object) -> CompletedProcess:
        return CompletedProcess([], 0, output, "")

    monkeypatch.setattr("scripts.import_holdout_50.subprocess.run", fake_run)
    assert list_entries(Path("source.rar"), 3) == ["x-m3-mirage-p1.dem", "x-m3-mirage-p2.dem"]
    output = "x-m3-mirage-p1.dem\nx-m3-mirage-p3.dem\n"
    with pytest.raises(PipelineError, match="分段缺失"):
        list_entries(Path("source.rar"), 3)
    output = "../x-m3-mirage.dem\n"
    with pytest.raises(PipelineError, match="无路径"):
        list_entries(Path("source.rar"), 3)


def test_score_audit_uses_real_teams_across_side_switch(tmp_path: Path) -> None:
    pl.DataFrame({"round_id": [1, 2, 3], "winner_side": ["T", "CT", "T"]}).write_parquet(
        tmp_path / "rounds.parquet"
    )
    pl.DataFrame(
        {
            "round_id": [1, 1, 2, 2, 3, 3],
            "tick": [100, 100, 200, 200, 300, 300],
            "team": ["9zGlobant", "MIBR"] * 3,
            "side": ["T", "CT", "CT", "T", "CT", "T"],
        }
    ).write_parquet(tmp_path / "snapshots.parquet")
    assert published_score([tmp_path], {"9z": 2, "MIBR": 1}) == {"9z": 2, "MIBR": 1}
    with pytest.raises(PipelineError, match="不一致"):
        published_score([tmp_path], {"9z": 1, "MIBR": 2})


def test_progress_retries_a_transient_windows_file_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    progress = tmp_path / "progress.json"
    monkeypatch.setattr("scripts.import_holdout_50.PROGRESS", progress)
    monkeypatch.setattr("scripts.import_holdout_50.time.sleep", lambda _: None)
    original = Path.replace
    calls = 0

    def sometimes_locked(path: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise PermissionError("test sharing violation")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", sometimes_locked)
    write_progress({"status": "running", "total": 50})
    assert calls == 3
    assert '"total": 50' in progress.read_text(encoding="utf-8")

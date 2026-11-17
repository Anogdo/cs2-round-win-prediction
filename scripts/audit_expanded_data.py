"""Independently verify the expanded corpus, event split and published map scores."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory


def audit(dataset: Path, manifest: Path) -> dict:
    metadata, tables = read_dataset(dataset)
    config = json.loads(manifest.read_text(encoding="utf-8"))
    if metadata.get("source_manifest_sha256") != sha256_file(manifest):
        raise PipelineError("数据集与冻结清单 SHA256 不一致。")
    if metadata["split_policy"]["group_by"] != "event":
        raise PipelineError("新实验必须按赛事隔离。")
    replay = json.loads(Path("data/processed/awpy330_expanded/reparse_report.json").read_text())
    if not replay["complete"] or len(replay["sources"]) != len(config["sources"]):
        raise PipelineError("统一解析未全部完成。")
    warnings, parser_versions = [], set()
    for entry in metadata["sources"]:
        path = (manifest.resolve().parent / entry["path"]).resolve()
        info = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        quality = validate_directory(path)
        if not quality["passed"] or info["demo_sha256"] != entry["demo_sha256"]:
            raise PipelineError(f"解析结果校验失败：{path}")
        parser_versions.add(info["parser_version"])
        warnings.extend(
            {"match_id": entry["match_id"], **warning} for warning in quality["warnings"]
        )
    if parser_versions != {"3.3.0"}:
        raise PipelineError(f"解析版本不一致：{parser_versions}")
    user_sources = json.loads(Path("configs/user_hltv_sources.json").read_text(encoding="utf-8"))
    scores = []
    for entry in user_sources["maps"]:
        name = f"{entry['match_id'].replace(':', '-')}-m{entry['map_index']}-p1"
        folder = Path("data/processed/awpy330_expanded") / name
        rounds = pl.read_parquet(folder / "rounds.parquet")
        states = pl.read_parquet(folder / "snapshots.parquet")
        beginning = states.sort("tick").group_by("round_id", "team").first()
        counts = (
            beginning.join(rounds.select("round_id", "winner_side"), on="round_id")
            .filter(pl.col("side") == pl.col("winner_side"))
            .group_by("team")
            .len()
        )
        score = {
            "PARIVISION" if team == "PVISION" else team: count for team, count in counts.iter_rows()
        }
        if rounds.height != entry["expected_rounds"] or score != entry["expected_score"]:
            raise PipelineError(f"回合数或战队比分与 HLTV 不符：{entry['match_id']} {score}")
        scores.append(
            {
                "match_id": entry["match_id"],
                "score": score,
                "rounds": rounds.height,
                "match_url": entry["match_url"],
            }
        )
    joined = tables["samples"].join(tables["splits"], on="sample_id", validate="1:1")
    counts = {}
    for split in ("train", "validation", "test"):
        part = joined.filter(pl.col("split") == split)
        counts[split] = {
            "matches": part["match_id"].n_unique(),
            "rounds": part.select("demo_sha256", "round_id").unique().height,
            "samples": part.height,
            "events": sorted(part["event"].unique().to_list()),
        }
    return {
        "passed": True,
        "parser_version": "3.3.0",
        "recordings": len(metadata["sources"]),
        "matches": joined["match_id"].n_unique(),
        "rounds": metadata["round_count"],
        "samples": metadata["sample_count"],
        "nodes": metadata["node_count"],
        "steamids": tables["nodes"]["steamid"].n_unique(),
        "counts": counts,
        "warnings": warnings,
        "verified_hltv_map_scores": scores,
        "dataset_manifest_sha256": sha256_file(dataset / "dataset.json"),
        "source_manifest_sha256": sha256_file(manifest),
    }


def main() -> None:
    report = audit(
        Path("data/datasets/mirage_expanded_2026_v1"), Path("configs/phase2_expanded_2026.json")
    )
    out = Path("reports/expanded_2026_data_audit.json")
    write_json(out, report)
    print(
        f"审计通过：{report['recordings']} 录制 / {report['matches']} 比赛 / "
        f"{report['rounds']} 回合 / {report['samples']} 时点；{out}"
    )


if __name__ == "__main__":
    main()

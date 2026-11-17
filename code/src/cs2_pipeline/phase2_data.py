"""Shared, provenance-checked inputs for the fixed Phase 2 exploration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.context import CONTEXT_FEATURES, read_context
from cs2_pipeline.dataset import held_out_event
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import PipelineError

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ("mirage_expanded_2026_v1", "mirage_holdout_50_v1")


def sources(root: Path) -> list[dict[str, Any]]:
    """Resolve only pre-existing datasets; never discover new test events."""
    entries = []
    for name in DATASETS:
        meta = json.loads((root / "data/datasets" / name / "dataset.json").read_text("utf-8"))
        for source in meta["sources"]:
            if held_out_event(source["event"]):
                raise PipelineError("发现 Bucharest 保留赛事，拒绝读取。")
            path = Path(source["path"])
            base = root / "configs" if str(path).startswith("..") else root
            path = (base / path).resolve()
            if not path.is_relative_to(root.resolve()):
                raise PipelineError(f"来源超出工作目录：{path}")
            record = json.loads((path / "metadata.json").read_text("utf-8"))
            if record["demo_sha256"] != source["demo_sha256"]:
                raise PipelineError(f"Demo SHA256 与数据清单不一致：{path}")
            entries.append({"dataset": name, "source": source, "path": path, "metadata": record})
    hashes = [item["metadata"]["demo_sha256"] for item in entries]
    if len(set(hashes)) != len(hashes):
        raise PipelineError("同一录制出现在多个来源，拒绝重复处理。")
    return entries


def context_for_samples(
    root: Path, dataset: str, items: list[dict[str, Any]], samples: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, Any]]:
    frames = []
    fallback_matches = []
    fingerprint = []
    for item in items:
        if item["dataset"] != dataset:
            continue
        checksum = item["metadata"]["demo_sha256"]
        directory = root / "data/context/phase2_audit" / checksum
        context = read_context(directory, item["path"], checksum)
        notes = json.loads((directory / "context_metadata.json").read_text("utf-8"))
        if notes.get("game_rules_status") == "event_only":
            fallback_matches.append(item["source"]["match_id"])
        fingerprint.append(
            {
                "demo_sha256": checksum,
                "context_sha256": notes["context_file_sha256"],
                "metadata_sha256": sha256_file(directory / "context_metadata.json"),
            }
        )
        frames.append(context.with_columns(pl.lit(checksum).alias("demo_sha256")))
    all_context = pl.concat(frames)
    aligned = samples.join(
        all_context, on=["demo_sha256", "tick", "round_id"], how="left", validate="1:1"
    )
    if aligned.height != samples.height or aligned["bomb_planted"].null_count():
        raise PipelineError(f"{dataset} 的上下文缺行或下包状态未知。")
    bad_binary = aligned.filter(
        ~pl.col("bomb_planted").is_in([0.0, 1.0])
        | (pl.col("bomb_site_a").fill_null(0) + pl.col("bomb_site_b").fill_null(0) > 1)
        | (
            (pl.col("bomb_planted") == 1)
            & pl.col("bomb_site_a").is_not_null()
            & pl.col("bomb_site_b").is_not_null()
            & (pl.col("bomb_site_a") + pl.col("bomb_site_b") != 1)
        )
        | (
            (pl.col("bomb_planted") == 0)
            & (pl.col("bomb_site_a").fill_null(0) + pl.col("bomb_site_b").fill_null(0) != 0)
        )
        | ((pl.col("bomb_planted") == 1) & pl.col("round_time_remaining_seconds").is_not_null())
    ).height
    if bad_binary:
        raise PipelineError(f"上下文中存在 {bad_binary} 个矛盾的炸弹/时钟状态。")
    return aligned, {
        "samples": aligned.height,
        "event_only_matches": sorted(set(fallback_matches)),
        "event_only_samples": aligned.filter(pl.col("match_id").is_in(fallback_matches)).height,
        "bomb_planted_samples": aligned.filter(pl.col("bomb_planted") == 1).height,
        "non_null_counts": {
            name: aligned.height - aligned[name].null_count() for name in CONTEXT_FEATURES
        },
        "planted_unknown_site": aligned.filter(
            (pl.col("bomb_planted") == 1)
            & (pl.col("bomb_site_a").is_null() | pl.col("bomb_site_b").is_null())
        ).height,
        "source_hashes": sorted(fingerprint, key=lambda x: x["demo_sha256"]),
    }

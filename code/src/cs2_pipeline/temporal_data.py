"""Build trailing V1 windows from audited samples and their recording-local round ticks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.dataset import held_out_event, read_dataset
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import PipelineError

ROUND_KEY = ("match_id", "map_id", "demo_sha256", "round_id")
RECORDING_KEY = ("match_id", "map_id", "part_index", "match_date", "event", "demo_sha256")
ELAPSED = "elapsed_since_freeze_end_seconds"


def join_samples(tables: dict[str, pl.DataFrame]) -> pl.DataFrame:
    samples = tables["samples"].rename({ELAPSED: "elapsed_seconds"})
    return samples.join(
        tables["features"], on="sample_id", validate="1:1", maintain_order="left"
    ).join(tables["labels"], on="sample_id", validate="1:1", maintain_order="left")


def recording_rounds(root: Path, sources: list[dict[str, Any]]) -> pl.DataFrame:
    frames = []
    for source in sources:
        if held_out_event(source["event"]):
            raise PipelineError("The Bucharest reserve must not be read for this experiment.")
        relative = Path(source["path"])
        base = root / "configs" if relative.parts[0] == ".." else root
        directory = (base / relative).resolve()
        if not directory.is_relative_to(root.resolve()):
            raise PipelineError(f"Recording path is outside the repository: {directory}")
        for name in ("metadata.json", "rounds.parquet"):
            if sha256_file(directory / name) != source["source_file_sha256"][name]:
                raise PipelineError(f"Recording round source has changed: {directory / name}")
        metadata = json.loads((directory / "metadata.json").read_text("utf-8"))
        rate = metadata.get("tick_rate")
        if (
            metadata["demo_sha256"] != source["demo_sha256"]
            or metadata["map_name"] != "de_mirage"
            or not isinstance(rate, (float, int))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise PipelineError(f"Invalid Mirage recording identity or tick rate: {directory}")
        frames.append(
            pl.read_parquet(
                directory / "rounds.parquet",
                columns=[
                    "round_id",
                    "round_num",
                    "freeze_end_tick",
                    "end_tick",
                    "is_valid_round",
                    "winner_side",
                ],
            )
            .rename({"round_num": "recorded_round_num"})
            .with_columns(
                *(pl.lit(source[key]).alias(key) for key in RECORDING_KEY),
                pl.lit(float(rate)).alias("tick_rate"),
            )
        )
    return pl.concat(frames)


def validate_timeline(frame: pl.DataFrame, rounds: pl.DataFrame) -> None:
    """Verify membership and label against verified rounds without returning future fields."""
    index = frame.join(rounds, on=[*RECORDING_KEY, "round_id"], how="left", validate="m:1")
    expected_id = pl.concat_str("demo_sha256", "round_id", "tick", separator=":")
    expected_elapsed = (pl.col("tick") - pl.col("freeze_end_tick")) / pl.col("tick_rate")
    valid = (
        (pl.col("sample_id") == expected_id)
        & (pl.col("map_name") == "de_mirage")
        & pl.col("is_valid_round")
        & (pl.col("round_num") == pl.col("recorded_round_num"))
        & (pl.col("tick") >= pl.col("freeze_end_tick"))
        & (pl.col("tick") < pl.col("end_tick"))
        & ((pl.col("elapsed_seconds") - expected_elapsed).abs() <= 1e-9)
        & (pl.col(ELAPSED) == pl.col("elapsed_seconds"))
        & pl.col("winner_side").is_in(["CT", "T"])
        & (pl.col("y_ct_win") == (pl.col("winner_side") == "CT").cast(pl.UInt8))
    ).fill_null(False)
    failed = index.filter(~valid)
    if failed.height:
        raise PipelineError(
            f"Misaligned sample identity, round label or combat tick ({failed.height} rows); "
            f"first sample: {failed['sample_id'][0]}"
        )


def load_v1(root: Path, path: Path, expected_hash: str) -> tuple[dict[str, Any], pl.DataFrame]:
    if sha256_file(path / "dataset.json") != expected_hash:
        raise PipelineError(f"Dataset differs from the frozen experiment input: {path}")
    metadata, tables = read_dataset(path)
    if metadata.get("feature_version") != 1 or metadata["feature_columns"] != feature_columns(1):
        raise PipelineError("This comparison requires the frozen 47-feature V1 datasets.")
    frame = join_samples(tables).sort("sample_id")
    if frame.is_empty() or frame.height != tables["samples"].height:
        raise PipelineError("Sample/feature/label alignment lost rows.")
    validate_timeline(frame, recording_rounds(root, metadata["sources"]))
    return metadata, frame


def verify_partition_order(parts: dict[str, pl.DataFrame]) -> None:
    seen: dict[str, set[str]] = {
        key: set() for key in ("event", "match_id", "map_id", "demo_sha256")
    }
    previous_end = ""
    for name, frame in parts.items():
        if frame.is_empty() or frame["match_date"].min() <= previous_end:
            raise PipelineError(f"Missing data or nonchronological matches in {name}.")
        for key, prior in seen.items():
            current = set(frame[key].unique().to_list())
            if None in current or prior & current:
                raise PipelineError(f"Invalid or overlapping {key} in {name}.")
            if key == "event" and any(held_out_event(event) for event in current):
                raise PipelineError("The Bucharest reserve must not participate.")
            prior.update(current)
        previous_end = frame["match_date"].max()


def window_indices(frame: pl.DataFrame, length: int) -> np.ndarray:
    """Left-pad with -1; put each target last and use only its own strictly earlier ticks."""
    if length < 1:
        raise PipelineError("Window length must be positive.")
    indices = np.full((frame.height, length), -1, dtype=np.int64)
    timeline = frame.select(*ROUND_KEY, "tick").with_row_index("row").sort(*ROUND_KEY, "tick")
    for group in timeline.partition_by(ROUND_KEY, maintain_order=True):
        ticks = group["tick"].to_numpy()
        if np.any(np.diff(ticks) <= 0):
            raise PipelineError("Duplicate tick within a recording-local round.")
        rows = group["row"].to_numpy()
        offsets = np.arange(len(rows))[:, None] + np.arange(1 - length, 1)
        indices[rows] = np.where(offsets < 0, -1, rows[offsets.clip(min=0)].astype(np.int64))
    if not np.array_equal(indices[:, -1], np.arange(frame.height)):
        raise PipelineError("A window did not end at its target sample.")
    return indices

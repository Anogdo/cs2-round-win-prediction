"""Compare V1 logistic with V1 plus the observed planted-bomb state."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.context import verify_event_only_coverage
from cs2_pipeline.dataset import write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import associate_rounds, sha256_file
from cs2_pipeline.phase2_compare import PLAN as SPLIT_PLAN
from cs2_pipeline.phase2_compare import counts, fit_logistic, predict_logistic
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import PLAN as TEMPORAL_PLAN
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import load_v1, verify_partition_order
from cs2_pipeline.train import round_weights

NEW_EVENTS = {"blast-bounty-2026-season-2-finals", "esports-world-cup-2026"}
BOMB = "bomb_planted"
SOURCE_PLAN = ROOT / "configs/phase2_validation_more_20261118.json"


def bomb_feature(samples: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """A plant takes effect at its recorded tick; never carry state between rounds."""
    transitions = events.filter(pl.col("event").is_in(["finish_plant", "defuse"]))
    if transitions.select(pl.struct("round_id", "tick").is_duplicated().any()).item():
        raise PipelineError("Multiple bomb transitions at one round tick; verify the recording.")
    transitions = transitions.select(
        "round_id", "tick", (pl.col("event") == "finish_plant").cast(pl.Float64).alias(BOMB)
    )
    return (
        samples.select("sample_id", "round_id", "tick")
        .sort("tick")
        .join_asof(transitions.sort("tick"), on="tick", by="round_id", check_sortedness=False)
        .select("sample_id", pl.col(BOMB).fill_null(0.0))
    )


def locate_demo(index: dict[str, list[Path]], filename: str, checksum: str) -> Path:
    for path in index.get(filename, []):
        if sha256_file(path) == checksum:
            return path
    raise PipelineError(f"Missing DEM with recorded SHA-256: {filename} ({checksum})")


def collect_bomb(
    frame: pl.DataFrame,
    metadata: dict[str, Any],
    demo_index: dict[str, list[Path]],
    reporter: ProgressReporter | None,
    *,
    stage: str = "load",
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    from awpy import Demo

    frames = []
    evidence = []
    by_recording = frame.partition_by("demo_sha256", as_dict=True)
    for source in metadata["sources"]:
        checksum = source["demo_sha256"]
        if (checksum,) not in by_recording:
            continue
        part = by_recording[(checksum,)]
        relative = Path(source["path"])
        processed = ((ROOT / "configs" if relative.parts[0] == ".." else ROOT) / relative).resolve()
        record = json.loads((processed / "metadata.json").read_text("utf-8"))
        demo = locate_demo(demo_index, record["demo_filename"], checksum)
        rounds = pl.read_parquet(processed / "rounds.parquet")
        emit(
            reporter,
            stage,
            "active",
            f"Reading bomb transitions: {source['match_id']} · {demo.name}",
        )
        raw = (
            Demo(demo)
            .bomb.select(pl.col("tick").cast(pl.Int64), "event")
            .filter(pl.col("event").is_in(["finish_plant", "defuse"]))
        )
        events = associate_rounds(raw, rounds).filter(pl.col("is_valid_round"))
        verify_event_only_coverage(rounds, events)
        feature = bomb_feature(part, events)
        frames.append(feature)
        evidence.append(
            {
                "demo_sha256": checksum,
                "match_id": source["match_id"],
                "recording_path": demo.relative_to(ROOT).as_posix(),
                "processed_rounds_sha256": source["source_file_sha256"]["rounds.parquet"],
                "plant_events_in_valid_rounds": events.filter(
                    pl.col("event") == "finish_plant"
                ).height,
                "valid_bomb_ending_rounds_checked": rounds.filter(
                    pl.col("is_valid_round")
                    & pl.col("reason_name").is_in(["target_bombed", "bomb_defused"])
                ).height,
                "samples": feature.height,
                "planted_samples": int(feature[BOMB].sum()),
            }
        )
    aligned = frame.join(
        pl.concat(frames), on="sample_id", how="left", validate="1:1", maintain_order="left"
    )
    if not aligned["sample_id"].equals(frame["sample_id"]) or aligned[BOMB].null_count():
        raise PipelineError("Bomb state does not cover precisely the requested V1 targets.")
    return aligned, evidence


def evaluate(frame: pl.DataFrame, bootstrap: dict) -> dict[str, Any]:
    control, augmented = (frame[name].to_list() for name in ("v1", "v1_bomb"))
    raw = {"v1": metrics(frame, control), "v1_bomb": metrics(frame, augmented)}
    result = {
        **counts(frame),
        "planted_samples": int(frame[BOMB].sum()),
        "metrics": raw,
        "v1_bomb_minus_v1": {
            key: raw["v1_bomb"][key] - raw["v1"][key]
            for key in ("log_loss", "brier_score", "ece_10_bins")
        },
    }
    if result["matches"] > 1:
        boot = cluster_bootstrap(
            frame, control, augmented, draws=bootstrap["draws"], seed=bootstrap["seed"]
        )
        result["paired_ci95"] = boot["percentile_95_intervals"]["calibrated_minus_base"]
    return result


def run(
    development: Path, expected_sha256: str, out: Path, reporter: ProgressReporter | None = None
) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"Choose a fresh output path: {out}")
    temporal_plan = json.loads(TEMPORAL_PLAN.read_text("utf-8"))
    if sha256_file(SPLIT_PLAN) != temporal_plan["comparison_plan_sha256"]:
        raise PipelineError("The frozen comparison plan changed.")
    policy = json.loads(SPLIT_PLAN.read_text("utf-8"))
    original_hash = temporal_plan["dataset_sha256"]
    original_meta, original = load_v1(
        ROOT, ROOT / policy["original_training_dataset"], original_hash
    )
    emit(reporter, "load", "active", "Verifying development data and event separation.")
    dev_meta, new = load_v1(ROOT, development, expected_sha256)
    source_plan_hash = sha256_file(SOURCE_PLAN)
    if (
        set(new["event"].unique()) != NEW_EVENTS
        or dev_meta["split_policy"]["method"] != "historical_validation_only"
        or dev_meta["source_manifest_sha256"] != source_plan_hash
        or set(pl.read_parquet(development / "splits.parquet", columns=["split"])["split"])
        != {"validation"}
    ):
        raise PipelineError("Both new events must be marked historical_validation_only.")
    if any(set(new[key]) & set(original[key]) for key in ("match_id", "map_id", "demo_sha256")):
        raise PipelineError("The new events overlap the historical corpus.")
    fit = original.filter(pl.col("event").is_in(policy["candidate_fit_events"]))
    cal = original.filter(pl.col("event").is_in(policy["candidate_calibration_events"]))
    porto = original.filter(pl.col("event").is_in(policy["candidate_validation_events"]))
    verify_partition_order({"fit": fit, "calibration_reserved": cal, "dev": new, "porto": porto})
    if new["match_date"].max() >= "2026-09-08":
        raise PipelineError("Development event dates must predate the separately observed 50.")

    demo_index = defaultdict(list)
    for directory in (ROOT / "data/raw", ROOT / "data/datasets/mirage demo"):
        for demo in directory.rglob("*.dem"):
            demo_index[demo.name].append(demo)
    fit, fit_evidence = collect_bomb(fit, original_meta, demo_index, reporter)
    emit(reporter, "load", "completed", "Training bomb events and V1 targets verified.")
    emit(
        reporter, "fit", "active", "Fitting paired round-weighted logistic models on Rio + Atlanta."
    )
    v1 = feature_columns(1)
    models = {"v1": fit_logistic(fit, v1), "v1_bomb": fit_logistic(fit, [*v1, BOMB])}
    if models["v1"]["indicator_indices"] != models["v1_bomb"]["indicator_indices"]:
        raise PipelineError("The extra feature must not introduce a missingness indicator.")

    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bomb-ablation-", dir=out.parent) as temporary:
        stage = Path(temporary) / "out"
        stage.mkdir()
        write_json(stage / "models.json", models)
        model_hash = sha256_file(stage / "models.json")
        emit(reporter, "fit", "completed", "Both model weights frozen before development scoring.")
        emit(reporter, "evaluate", "active", "Reading events and scoring paired predictions.")
        dev, dev_evidence = collect_bomb(new, dev_meta, demo_index, reporter, stage="evaluate")
        porto, porto_evidence = collect_bomb(
            porto, original_meta, demo_index, reporter, stage="evaluate"
        )
        predictions = {}
        for label, frame in (("development", dev), ("porto", porto)):
            rows = frame.select(
                "sample_id",
                "match_id",
                "event",
                "match_date",
                "demo_sha256",
                "round_id",
                "tick",
                "y_ct_win",
                BOMB,
            )
            predictions[label] = rows.with_columns(
                pl.Series("round_weight", round_weights(frame)),
                *(
                    pl.Series(name, predict_logistic(model, frame))
                    for name, model in models.items()
                ),
            )
            path = stage / f"{label}_predictions.parquet"
            predictions[label].write_parquet(path)
            if not pl.read_parquet(path).equals(predictions[label]):
                raise PipelineError(f"Saved predictions changed rows for {label}.")
        dev_rows = predictions["development"]
        pooled = evaluate(dev_rows, policy["bootstrap"])
        by_event = {
            name: evaluate(dev_rows.filter(pl.col("event") == name), policy["bootstrap"])
            for name in sorted(dev_rows["event"].unique())
        }
        by_bomb_state = {
            name: evaluate(subset, policy["bootstrap"])
            for value, name in ((0.0, "unplanted"), (1.0, "planted"))
            if not (subset := dev_rows.filter(pl.col(BOMB) == value)).is_empty()
        }
        result = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "scope": "Historical development validation only; no prospective blind-test claim.",
            "bomb_definition": "Latest completed plant or defuse at or before the target tick "
            "in the same recorded combat round; reset each round.",
            "bomb_quality_note": "Known bomb-ended rounds require observed plant events. End "
            "reason is used only to audit event coverage, not to infer per-tick state.",
            "model_fit": {**counts(fit), "planted_samples": int(fit[BOMB].sum())},
            "calibration_reserved_unused": counts(cal),
            "comparison_plan_sha256": sha256_file(SPLIT_PLAN),
            "original_dataset_sha256": original_hash,
            "development_dataset_sha256": expected_sha256,
            "development_source_plan_sha256": source_plan_hash,
            "lock_sha256": sha256_file(ROOT / "uv.lock"),
            "code_sha256": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in (
                    "phase2_bomb.py",
                    "phase2_compare.py",
                    "temporal_data.py",
                    "context.py",
                    "calibrate.py",
                    "train.py",
                    "parse.py",
                )
            },
            "versions": {
                name: version(name) for name in ("awpy", "numpy", "polars", "scikit-learn")
            },
            "features": {name: model["feature_columns"] for name, model in models.items()},
            "calibration": "None; ECE and reliability bins describe raw probability estimates.",
            "bootstrap": {
                **policy["bootstrap"],
                "interpretation": "Within these matches, not between-event uncertainty.",
            },
            "portfolio": {
                "development_pooled": pooled,
                "development_by_event": by_event,
                "development_by_bomb_state": by_bomb_state,
                "historical_porto": evaluate(predictions["porto"], policy["bootstrap"]),
            },
            "recording_audit": {
                "fit": fit_evidence,
                "development": dev_evidence,
                "porto": porto_evidence,
            },
            "model_sha256_before_scoring": model_hash,
            "artifact_sha256": {p.name: sha256_file(p) for p in stage.iterdir()},
        }
        emit(
            reporter,
            "evaluate",
            "completed",
            "Aligned prediction metrics and paired intervals complete.",
        )
        emit(reporter, "save", "active", "Writing reproducibility artifacts to a fresh directory.")
        write_json(stage / "report.json", result)
        stage.rename(out)
    emit(reporter, "save", "completed", f"Bomb-only ablation written to {out}")
    emit(reporter, "run", "completed", "Bomb-only ablation complete.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True, type=Path)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        result = run(args.development, args.dataset_sha256, args.out, reporter)
    except (
        OSError,
        KeyError,
        ValueError,
        RuntimeError,
        PipelineError,
        pl.exceptions.PolarsError,
    ) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(f"Bomb-only comparison failed: {exc}", file=sys.stderr)
        return 1
    delta = result["portfolio"]["development_pooled"]["v1_bomb_minus_v1"]["log_loss"]
    print(
        f"Development delta (V1 + bomb minus V1) log loss: {delta:+.6f}. "
        f"Report: {args.out / 'report.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

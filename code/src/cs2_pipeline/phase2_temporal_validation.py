"""Score unchanged Porto-selected checkpoints on the added development events."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.dataset import write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import PLAN as COMPARISON_PLAN
from cs2_pipeline.phase2_compare import counts, sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import (
    MODELS,
    driver_version,
    history_comparison,
    logistic_predict,
    network_arrays,
    prediction_table,
    score,
    transform,
)
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import load_v1, verify_partition_order, window_indices

PLAN = ROOT / "configs/phase2_temporal_validation_plan.json"
FROZEN_FILES = {
    "plan.json",
    "preprocessing.json",
    "logistic.json",
    "snapshot_transformer.pt",
    "history8_transformer.pt",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_frozen_run(path: Path, selection_hash: str) -> dict[str, Any]:
    if sha256_file(path / "selection.json") != selection_hash:
        raise PipelineError("The Porto selection receipt differs from the frozen hash.")
    selection = read_json(path / "selection.json")
    files = selection["pre_exploration_sha256"]
    if files.keys() != FROZEN_FILES:
        raise PipelineError("The Porto selection receipt does not list the five frozen inputs.")
    for filename, digest in files.items():
        if sha256_file(path / filename) != digest:
            raise PipelineError(f"Frozen experiment input has changed: {filename}")
    return selection


def audit_scope(dataset: Path, temporal: dict, comparison: dict, plan: dict) -> dict:
    """Check dates and identity against recorded manifests without reading older labels."""
    identities = ("event", "match_id", "map_id", "demo_sha256", "match_date")
    manifests = []
    for path, hash_key in (
        (ROOT / comparison["original_training_dataset"], "dataset_sha256"),
        (ROOT / comparison["exploratory_dataset"], "exploratory_dataset_sha256"),
    ):
        metadata = path / "dataset.json"
        if sha256_file(metadata) != temporal[hash_key]:
            raise PipelineError(f"Frozen manifest has changed: {metadata}")
        sources = read_json(metadata)["sources"]
        manifests.append(
            pl.DataFrame([{key: record[key] for key in identities} for record in sources])
        )
    earlier, observed = manifests
    digest = sha256_file(dataset / "dataset.json")
    meta, frame = load_v1(ROOT, dataset, digest)
    if meta["split_policy"]["method"] != "historical_validation_only":
        raise PipelineError("This dataset must be explicitly reserved for historical validation.")
    if set(frame["event"].unique()) != set(plan["historical_events"]):
        raise PipelineError("Historical dataset events differ from this plan.")
    source_plan = ROOT / plan["historical_source_plan"]
    source_hash = plan["historical_source_sha256"]
    if sha256_file(source_plan) != source_hash or meta["source_manifest_sha256"] != source_hash:
        raise PipelineError("Historical validation intake manifest differs from the pinned plan.")
    planned = {
        entry["match_id"]: entry["archive_sha256"] for entry in read_json(source_plan)["archives"]
    }
    present = {entry["match_id"]: entry["archive_sha256"] for entry in meta["sources"]}
    if planned != present or set(frame["match_id"]) != set(planned):
        raise PipelineError("Historical match or archive roster differs from the pinned intake.")
    splits = pl.read_parquet(dataset / "splits.parquet", columns=["split"])
    if set(splits["split"]) != {"validation"}:
        raise PipelineError("Historical samples must all be assigned to validation.")
    dev = frame.select(*identities).unique()
    for known in manifests:
        for key in identities[:-1]:
            if set(dev[key]) & set(known[key]):
                raise PipelineError(f"Historical batch overlaps previously used {key} provenance.")
    parts = {
        name: earlier.filter(pl.col("event").is_in(comparison[f"candidate_{name}_events"]))
        for name in ("fit", "calibration", "validation")
    }
    verify_partition_order(
        {
            "fit": parts["fit"],
            "reserved_calibration": parts["calibration"],
            "historical_development": dev,
            "porto_selection": parts["validation"],
            "observed_50": observed,
        }
    )
    return {"frame": frame, "dataset_sha256": digest}


def run(dataset: Path, out: Path, reporter: ProgressReporter | None = None) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"Use a fresh output directory; already exists: {out}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, cuda_environment
        from cs2_pipeline.temporal_net import predict as network_predict
    except (ImportError, OSError) as exc:
        raise PipelineError(
            "Install the locked CUDA dependencies with uv sync --extra temporal."
        ) from exc

    emit(
        reporter, "load", "active", "Verifying frozen checkpoints and historical development data."
    )
    plan = read_json(PLAN)
    frozen = ROOT / plan["frozen_run"]
    selection = verify_frozen_run(frozen, plan["selection_sha256"])
    temporal = read_json(frozen / "plan.json")
    if sha256_file(COMPARISON_PLAN) != temporal["comparison_plan_sha256"]:
        raise PipelineError("Original comparison plan has changed.")
    comparison = read_json(COMPARISON_PLAN)
    historical = audit_scope(dataset, temporal, comparison, plan)
    frame = historical["frame"]
    values = transform(frame, read_json(frozen / "preprocessing.json"))
    rows = window_indices(frame, max(temporal["window_lengths"]))
    emit(reporter, "load", "completed", "Verified identical targets and legal trailing windows.")
    emit(
        reporter, "fit", "completed", "Reusing the frozen Porto models; no fitting or calibration."
    )
    emit(reporter, "evaluate", "active", "Scoring frozen logistic regression and CUDA checkpoints.")
    device, hardware = cuda_environment(temporal["training"]["device"])
    hardware["nvidia_driver"] = driver_version()
    predictions = {"logistic": logistic_predict(read_json(frozen / "logistic.json"), values)}
    for name, window in zip(MODELS[1:], temporal["window_lengths"], strict=True):
        model = WindowTransformer(values.shape[1], rows.shape[1], temporal["transformer"]).to(
            device
        )
        checkpoint = torch.load(frozen / f"{name}.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint)
        data = DeviceWindows.from_arrays(*network_arrays(frame, values, rows, window), device)
        predictions[name] = sigmoid(
            network_predict(model, data, temporal["training"]["batch_size"])
        )

    results = {}
    for label in ["pooled", *plan["historical_events"]]:
        mask = (
            np.ones(frame.height, dtype=bool)
            if label == "pooled"
            else frame["event"].to_numpy() == label
        )
        part = frame.filter(pl.Series(mask))
        probabilities = {name: prob[mask] for name, prob in predictions.items()}
        results[label] = {
            **counts(part),
            "scores": score(part, probabilities),
            **history_comparison(part, rows[mask], probabilities, plan["bootstrap"]),
        }
    selected = selection["selected_by_porto_log_loss"]
    emit(
        reporter,
        "evaluate",
        "completed",
        "Historical development scores computed.",
        split="validation",
        metrics=results["pooled"]["scores"][selected],
    )
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "temporal_historical_validation",
        "evaluation_scope": plan["scope"],
        "fit_or_calibration_performed": False,
        "frozen_selection": selection,
        "frozen_selection_sha256": plan["selection_sha256"],
        "dataset_sha256": historical["dataset_sha256"],
        "historical_source_sha256": plan["historical_source_sha256"],
        "code_sha256": sha256_file(Path(__file__)),
        "plan_sha256": sha256_file(PLAN),
        "device": hardware,
        "results": results,
        "metrics": {"validation": {"model": results["pooled"]["scores"][selected]}},
    }
    emit(reporter, "save", "active", "Saving probabilities for the common target IDs.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-validation-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        shutil.copyfile(PLAN, stage / "plan.json")
        path = stage / "predictions.parquet"
        table = prediction_table(frame, rows, predictions)
        table.write_parquet(path)
        if not pl.read_parquet(path).equals(table):
            raise PipelineError("Saved Parquet differs from aligned predictions.")
        report["predictions_sha256"] = sha256_file(path)
        write_json(stage / "report.json", report)
        if out.exists():
            raise PipelineError(f"Output directory was created during the run: {out}")
        stage.rename(out)
    emit(reporter, "save", "completed", "Historical validation artifacts saved.", output=str(out))
    emit(reporter, "run", "completed", "Frozen checkpoint validation completed.")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    progress = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        run(args.dataset, args.out, progress)
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        pl.exceptions.PolarsError,
    ) as exc:
        if progress:
            progress.emit("run", "failed", str(exc))
        else:
            print(f"Historical validation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

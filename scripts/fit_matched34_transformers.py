"""Fit a logistic and paired temporal candidates on the exact archived 34-map split."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import counts
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import (
    COMPARISON_PLAN,
    PLAN,
    driver_version,
    fit_logistic,
    fit_preprocessing,
    network_arrays,
    transform,
)
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import ROUND_KEY, load_v1, window_indices
from cs2_pipeline.train import round_weights

SEEDS = (42, 7, 2026)
LENGTHS = (1, 8)
TEMPORAL_PLAN_SHA256 = "aa68625bc55b5b12053601b899612657aff018c7f54019b77c2a61d32b5c4c8a"
PORTO_SELECTION = ROOT / "models/monitor_runs/20261118T051803-7248d43d/selection.json"
PORTO_SELECTION_SHA256 = "47c7fb00b428361b8a75b4674755b741aa469cacee1212ada315e520bb86e1ba"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fit_inputs() -> tuple[dict[str, Any], pl.DataFrame]:
    if sha256_file(PLAN) != TEMPORAL_PLAN_SHA256:
        raise PipelineError("The original temporal training plan has changed.")
    if sha256_file(PORTO_SELECTION) != PORTO_SELECTION_SHA256:
        raise PipelineError("The prior 18-map Porto checkpoint selection has changed.")
    if read_json(PORTO_SELECTION)["checkpoint_epochs"] != {
        "snapshot_transformer": 1,
        "history8_transformer": 1,
    }:
        raise PipelineError("Both windows must use the prior selected epoch of one.")
    plan = read_json(PLAN)
    if sha256_file(COMPARISON_PLAN) != plan["comparison_plan_sha256"]:
        raise PipelineError("The original Phase 2 split plan has changed.")
    comparison = read_json(COMPARISON_PLAN)
    dataset = ROOT / comparison["original_training_dataset"]
    _, table = load_v1(ROOT, dataset, plan["dataset_sha256"])
    fit_events = comparison["full_baseline_fit_events"]
    fit_frame = table.filter(pl.col("event").is_in(fit_events))
    archived = pl.read_parquet(dataset / "splits.parquet").filter(pl.col("split") == "train")
    if (
        not fit_frame["sample_id"].equals(archived["sample_id"].sort())
        or fit_frame.height != 13_208
        or fit_frame["match_id"].n_unique() != 34
        or fit_frame.select("demo_sha256", "round_id").n_unique() != 730
        or set(fit_frame["event"].unique()) != set(fit_events)
    ):
        raise PipelineError("The candidate fit IDs differ from the exact frozen 34-map exposure.")
    return plan, fit_frame


def fit(out: Path, reporter: ProgressReporter | None) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"Use a fresh output path; already exists: {out}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        from cs2_pipeline.temporal_net import (
            DeviceWindows,
            WindowTransformer,
            cuda_environment,
            train_epoch,
        )
    except (ImportError, OSError) as exc:
        raise PipelineError(
            "Install the locked CUDA dependencies with uv sync --extra temporal."
        ) from exc

    emit(reporter, "load", "active", "Verifying the exact 34 archived training maps.")
    plan, frame = fit_inputs()
    training, architecture = plan["training"], plan["transformer"]
    device, device_evidence = cuda_environment(training["device"])
    device_evidence["nvidia_driver"] = driver_version()
    preprocessing = fit_preprocessing(frame)
    values = transform(frame, preprocessing)
    rows = window_indices(frame, max(LENGTHS))
    emit(reporter, "load", "completed", "Verified 34 maps, 730 rounds and 13,208 fit snapshots.")
    emit(reporter, "fit", "active", "Fitting a matched logistic and three one-epoch CUDA pairs.")
    telemetry = {}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".matched34-fit-", dir=out.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_json(stage / "preprocessing.json", preprocessing)
        write_json(stage / "controlled_logistic.json", fit_logistic(frame, values, plan))
        fit_sample_ids = frame.select("sample_id", "event", *ROUND_KEY).with_columns(
            pl.Series("round_weight", round_weights(frame))
        )
        fit_sample_ids.write_parquet(stage / "fit_sample_ids.parquet")
        if not fit_sample_ids.equals(pl.read_parquet(stage / "fit_sample_ids.parquet")):
            raise PipelineError("Fit sample IDs or weights changed after serialization.")
        windows = {
            length: DeviceWindows.from_arrays(*network_arrays(frame, values, rows, length), device)
            for length in LENGTHS
        }
        for seed in SEEDS:
            for length in LENGTHS:
                name = f"seed_{seed}_window_{length}"
                torch.manual_seed(seed)
                shuffle = torch.Generator().manual_seed(seed)
                model = WindowTransformer(values.shape[1], max(LENGTHS), architecture).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=training["learning_rate"],
                    weight_decay=training["weight_decay"],
                    foreach=False,
                    fused=False,
                )
                telemetry[name] = train_epoch(
                    model,
                    optimizer,
                    windows[length],
                    training["batch_size"],
                    shuffle,
                    verify_step=True,
                )
                torch.save(model.state_dict(), stage / f"{name}.pt")
                emit(
                    reporter,
                    "fit",
                    "active",
                    f"Frozen {name} after one CUDA epoch.",
                    model=name,
                    completed=len(telemetry),
                    total=len(SEEDS) * len(LENGTHS),
                )
        emit(reporter, "fit", "completed", "All matched candidates frozen; no scoring performed.")
        emit(reporter, "save", "active", "Hashing model checkpoints and fit sample IDs.")
        report = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "scope": (
                "Fit only on the archived full_baseline_fit_events. Prior Porto-only 18-map "
                "study fixed one epoch; Porto and Cologne are part of the fit here. "
                "No validation-based selection, calibration, or evaluation in this run."
            ),
            "fit": counts(frame),
            "fit_dataset_sha256": plan["dataset_sha256"],
            "comparison_plan_sha256": plan["comparison_plan_sha256"],
            "temporal_plan_sha256": TEMPORAL_PLAN_SHA256,
            "porto_selection_sha256": PORTO_SELECTION_SHA256,
            "code_sha256": sha256_file(Path(__file__)),
            "architecture": architecture,
            "training_config": {**training, "max_epochs": 1},
            "seeds": list(SEEDS),
            "primary_seed": SEEDS[0],
            "lengths": list(LENGTHS),
            "epochs": 1,
            "device": device_evidence,
            "training": telemetry,
            "artifacts": {path.name: sha256_file(path) for path in sorted(stage.iterdir())},
        }
        write_json(stage / "fit.json", report)
        if out.exists():
            raise PipelineError(f"Output path appeared during fit: {out}")
        stage.rename(out)
    emit(reporter, "save", "completed", f"Saved frozen training outputs in {out}.")
    emit(reporter, "run", "completed", "Matched 34-map fit complete.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        fit(args.out, reporter)
    except (
        PipelineError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as exc:
        emit(reporter, "run", "failed", str(exc))
        parser.exit(1, f"Matched fit failed: {exc}\n")


if __name__ == "__main__":
    main()

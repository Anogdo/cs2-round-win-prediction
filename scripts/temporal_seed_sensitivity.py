"""Freeze two additional one-epoch Transformer pairs, then score them without tuning."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import counts, sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import (
    COMPARISON_PLAN,
    PLAN,
    fit_preprocessing,
    network_arrays,
    transform,
)
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import ROUND_KEY, load_v1, verify_partition_order, window_indices
from cs2_pipeline.train import round_weights

SEEDS = (7, 2026)
LENGTHS = (1, 8)
DEVELOPMENT_EVENTS = {"blast-bounty-2026-season-2-finals", "esports-world-cup-2026"}
PORTO_SELECTION = ROOT / "models/monitor_runs/20261118T051803-7248d43d/selection.json"
PORTO_SELECTION_SHA256 = "47c7fb00b428361b8a75b4674755b741aa469cacee1212ada315e520bb86e1ba"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def model_name(seed: int, length: int) -> str:
    return f"seed_{seed}_window_{length}"


def original_frames() -> tuple[dict, dict, dict[str, pl.DataFrame]]:
    plan = read_json(PLAN)
    if sha256_file(COMPARISON_PLAN) != plan["comparison_plan_sha256"]:
        raise PipelineError("The original evaluation plan has changed.")
    comparison = read_json(COMPARISON_PLAN)
    _, original = load_v1(
        ROOT, ROOT / comparison["original_training_dataset"], plan["dataset_sha256"]
    )
    parts = {
        split: original.filter(pl.col("event").is_in(comparison[f"candidate_{split}_events"]))
        for split in ("fit", "calibration", "validation")
    }
    verify_partition_order(parts)
    if counts(parts["fit"])["matches"] != 18:
        raise PipelineError("Expected the original 18 Rio/Atlanta fit matches.")
    return plan, comparison, parts


def fit(out: Path, reporter: ProgressReporter | None) -> None:
    import torch

    from cs2_pipeline.temporal_net import (
        DeviceWindows,
        WindowTransformer,
        cuda_environment,
        train_epoch,
    )

    if sha256_file(PORTO_SELECTION) != PORTO_SELECTION_SHA256:
        raise PipelineError("The original Porto checkpoint selection has changed.")
    if set(read_json(PORTO_SELECTION)["checkpoint_epochs"].values()) != {1}:
        raise PipelineError("This sensitivity check fixes both models at the selected first epoch.")
    emit(reporter, "load", "active", "Reading only the original Rio and Atlanta fit cohort.")
    plan, _, parts = original_frames()
    training, architecture = plan["training"], plan["transformer"]
    device, device_evidence = cuda_environment(training["device"])
    frame = parts["fit"]
    preprocessing = fit_preprocessing(frame)
    values, rows = transform(frame, preprocessing), window_indices(frame, max(LENGTHS))
    emit(reporter, "load", "completed", "Frozen inputs verified.")
    emit(reporter, "fit", "active", "Fitting two fixed seed pairs, one epoch each.")
    telemetry = {}
    artifacts = ["preprocessing.json"]
    write_json(out / artifacts[0], preprocessing)
    for seed in SEEDS:
        for length in LENGTHS:
            torch.manual_seed(seed)
            shuffle = torch.Generator().manual_seed(seed)
            model = WindowTransformer(len(feature_columns(1)), max(LENGTHS), architecture).to(
                device
            )
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training["learning_rate"],
                weight_decay=training["weight_decay"],
                foreach=False,
                fused=False,
            )
            data = DeviceWindows.from_arrays(*network_arrays(frame, values, rows, length), device)
            name = model_name(seed, length)
            telemetry[name] = train_epoch(
                model, optimizer, data, training["batch_size"], shuffle, verify_step=True
            )
            checkpoint = f"{name}.pt"
            torch.save(model.state_dict(), out / checkpoint)
            artifacts.append(checkpoint)
            emit(reporter, "fit", "active", f"Frozen {name} after one CUDA epoch.", model=name)
    emit(reporter, "fit", "completed", "All four checkpoints frozen before development scoring.")
    write_json(
        out / "fit.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "fit": counts(frame),
            "fit_dataset_sha256": plan["dataset_sha256"],
            "temporal_plan_sha256": sha256_file(PLAN),
            "porto_selection_sha256": PORTO_SELECTION_SHA256,
            "code_sha256": sha256_file(Path(__file__)),
            "seeds": list(SEEDS),
            "lengths": list(LENGTHS),
            "epochs": 1,
            "device": device_evidence,
            "training": telemetry,
            "artifacts": {name: sha256_file(out / name) for name in artifacts},
        },
    )


def score(frozen: Path, dataset: Path, out: Path, reporter: ProgressReporter | None) -> None:
    import torch

    from cs2_pipeline.temporal_net import (
        DeviceWindows,
        WindowTransformer,
        cuda_environment,
        predict,
    )

    fit_record = read_json(frozen / "fit.json")
    needed = {"preprocessing.json"} | {
        f"{model_name(seed, length)}.pt" for seed in SEEDS for length in LENGTHS
    }
    if (
        fit_record["temporal_plan_sha256"] != sha256_file(PLAN)
        or fit_record["porto_selection_sha256"] != PORTO_SELECTION_SHA256
        or fit_record["code_sha256"] != sha256_file(Path(__file__))
        or fit_record["epochs"] != 1
        or set(fit_record["artifacts"]) != needed
        or any(sha256_file(frozen / name) != fit_record["artifacts"][name] for name in needed)
    ):
        raise PipelineError("Frozen seed artifacts are missing or differ from the recorded run.")
    emit(reporter, "load", "active", "Verifying split isolation and frozen seed checkpoints.")
    plan, comparison, parts = original_frames()
    digest = sha256_file(dataset / "dataset.json")
    metadata, frame = load_v1(ROOT, dataset, digest)
    if (
        metadata["split_policy"]["method"] != "historical_validation_only"
        or set(frame["event"]) != DEVELOPMENT_EVENTS
    ):
        raise PipelineError("This score requires the held-out historical Bounty/EWC cohort.")
    parts = {
        "fit": parts["fit"],
        "calibration": parts["calibration"],
        "historical": frame,
        "validation": parts["validation"],
    }
    verify_partition_order(parts)
    emit(reporter, "load", "completed", "Historical samples verified.")
    emit(reporter, "fit", "completed", "Using frozen checkpoints; no new training.")
    device, device_evidence = cuda_environment(plan["training"]["device"])
    rows = window_indices(frame, max(LENGTHS))
    values = transform(frame, read_json(frozen / "preprocessing.json"))
    predictions = frame.select("sample_id", *ROUND_KEY, "event", "tick", "y_ct_win").with_columns(
        pl.Series("round_weight", round_weights(frame)),
        pl.Series("available_snapshots", (rows >= 0).sum(axis=1)),
    )
    emit(reporter, "evaluate", "active", "Scoring the matched historical targets.")
    for seed in SEEDS:
        for length in LENGTHS:
            name = model_name(seed, length)
            model = WindowTransformer(
                len(feature_columns(1)), max(LENGTHS), plan["transformer"]
            ).to(device)
            state = torch.load(frozen / f"{name}.pt", weights_only=True, map_location="cpu")
            model.load_state_dict(state)
            data = DeviceWindows.from_arrays(*network_arrays(frame, values, rows, length), device)
            probability = sigmoid(predict(model, data, plan["training"]["batch_size"]))
            predictions = predictions.with_columns(pl.Series(name, probability))
            emit(reporter, "evaluate", "active", f"Completed {name}.", model=name)
    groups = {"pooled": predictions}
    groups.update(
        {
            event: predictions.filter(pl.col("event") == event)
            for event in sorted(DEVELOPMENT_EVENTS)
        }
    )
    results = {}
    for label, group in groups.items():
        results[label] = {"matches": group["match_id"].n_unique(), "seeds": {}}
        for seed in SEEDS:
            current, history = (group[model_name(seed, length)].to_list() for length in LENGTHS)
            current_metrics, history_metrics = metrics(group, current), metrics(group, history)
            bootstrap = comparison["bootstrap"]
            paired = cluster_bootstrap(
                group, current, history, draws=bootstrap["draws"], seed=bootstrap["seed"]
            )
            results[label]["seeds"][str(seed)] = {
                "current": current_metrics,
                "history": history_metrics,
                "history_minus_current": {
                    key: history_metrics[key] - current_metrics[key]
                    for key in ("log_loss", "brier_score", "roc_auc")
                },
                "paired_match_bootstrap_95": paired["percentile_95_intervals"][
                    "calibrated_minus_base"
                ],
            }
    emit(reporter, "evaluate", "completed", "Historical sensitivity scores calculated.")
    predictions.write_parquet(out / "predictions.parquet")
    if not predictions.equals(pl.read_parquet(out / "predictions.parquet")):
        raise PipelineError("Saved prediction rows do not match the scored observations.")
    write_json(
        out / "report.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "run_mode": "historical_validation_only",
            "scope": "Historical development only, no refitting, epoch selection, or calibration.",
            "dataset_sha256": digest,
            "frozen_fit_sha256": sha256_file(frozen / "fit.json"),
            "predictions_sha256": sha256_file(out / "predictions.parquet"),
            "evaluation": counts(frame),
            "fit": fit_record["fit"],
            "device": device_evidence,
            "results": results,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit", "score"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Use a fresh --out directory.")
    if args.mode == "score" and (args.dataset is None or args.frozen is None):
        parser.error("score requires --dataset and --frozen.")
    if args.mode == "fit" and (args.dataset is not None or args.frozen is not None):
        parser.error("fit does not accept development inputs.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    reporter = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".seed-sensitivity-", dir=args.out.parent) as temp:
            staged = Path(temp) / "artifacts"
            staged.mkdir()
            if args.mode == "fit":
                fit(staged, reporter)
            else:
                score(args.frozen, args.dataset, staged, reporter)
            emit(reporter, "save", "active", "Saving reproducible seed artifacts.")
            staged.rename(args.out)
            emit(reporter, "save", "completed", f"Saved {args.out}.")
        emit(reporter, "run", "completed", "Temporal seed sensitivity stage complete.")
    except (
        PipelineError,
        OSError,
        ValueError,
        RuntimeError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as exc:
        emit(reporter, "run", "failed", str(exc))
        parser.exit(1, f"Seed sensitivity failed: {exc}\n")


if __name__ == "__main__":
    main()

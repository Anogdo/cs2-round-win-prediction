"""Fit matched Mirage models on two nested sequences of 16, 24, and 32 matches."""

from __future__ import annotations

import argparse
import math
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import log_loss

from cs2_pipeline.dataset import write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import counts, sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import (
    fit_logistic,
    fit_preprocessing,
    logistic_predict,
    network_arrays,
    transform,
)
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import load_v1, verify_partition_order, window_indices
from scripts.fit_matched34_transformers import fit_inputs, read_json
from scripts.fit_mirage_snapshot_epochs import (
    check_scope,
    cuda_device,
    hashes,
    locked_selection,
    target_index,
    write_parquet,
)

PLAN = ROOT / "configs/mirage_train_size.json"


def make_splits(
    full: pl.DataFrame, plan: dict[str, Any]
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    events = plan["event_order"]
    pool = [
        sorted(full.filter(pl.col("event") == event)["match_id"].unique().to_list())
        for event in events
    ]
    if [len(group) for group in pool] != plan["event_counts_by_size"]["32"]:
        raise PipelineError("The event-specific counts differ from the frozen 32-match fit set.")
    trials, manifest = {}, {}
    for seed in plan["subset_seeds"]:
        rng = np.random.default_rng(seed)
        ranked = [rng.permutation(group).tolist() for group in pool]
        for size in ("16", "24"):
            prefix = [
                items[:n]
                for items, n in zip(ranked, plan["event_counts_by_size"][size], strict=True)
            ]
            name = f"draw_{seed}_n{size}"
            match_ids = [match for event_matches in prefix for match in event_matches]
            trials[name] = full.filter(pl.col("match_id").is_in(match_ids))
            manifest[name] = {
                "subset_seed": seed,
                "size": int(size),
                "match_ids_by_event": dict(zip(events, prefix, strict=True)),
                "fit": counts(trials[name]),
            }
            if manifest[name]["fit"]["matches"] != int(size):
                raise PipelineError(f"Match selection differs from the requested size: {name}")
    trials["full_n32"] = full
    manifest["full_n32"] = {
        "subset_seeds": plan["subset_seeds"],
        "size": 32,
        "match_ids_by_event": dict(zip(events, pool, strict=True)),
        "fit": counts(full),
    }
    return trials, manifest


def fit_prefixes(
    directory: Path,
    trial: str,
    fit_frame: pl.DataFrame,
    fit_values: np.ndarray,
    dev_frame: pl.DataFrame,
    dev_values: np.ndarray,
    temporal: dict[str, Any],
    run_seeds: list[int],
    epochs: int,
    device: Any,
    reporter: ProgressReporter,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, predict, train_epoch

    fit = DeviceWindows.from_arrays(
        *network_arrays(fit_frame, fit_values, window_indices(fit_frame, 1), 1), device
    )
    dev = DeviceWindows.from_arrays(
        *network_arrays(dev_frame, dev_values, window_indices(dev_frame, 1), 1), device
    )
    fy, dy = fit_frame["y_ct_win"].to_numpy(), dev_frame["y_ct_win"].to_numpy()
    # Use the original Float64 scoring weights, including for the fit curve.
    fw = target_index(fit_frame)["round_weight"].to_numpy()
    dw = target_index(dev_frame)["round_weight"].to_numpy()
    config = temporal["training"]
    batch = config["batch_size"]
    rows, epoch_one, runs = [], {}, {}
    for seed in run_seeds:
        torch.manual_seed(seed)
        shuffle = torch.Generator().manual_seed(seed)
        model = WindowTransformer(
            fit_values.shape[1], max(temporal["window_lengths"]), temporal["transformer"]
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            foreach=False,
            fused=False,
        )
        telemetry = []
        cumulative = 0
        for epoch in range(1, epochs + 1):
            step = train_epoch(model, optimizer, fit, batch, shuffle, verify_step=epoch == 1)
            telemetry.append(step)
            cumulative += step["steps"]
            p = sigmoid(predict(model, dev, batch))
            fit_loss = float(
                log_loss(fy, sigmoid(predict(model, fit, batch)), labels=[0, 1], sample_weight=fw)
            )
            dev_loss = float(log_loss(dy, p, labels=[0, 1], sample_weight=dw))
            rows.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "epoch": epoch,
                    "optimizer_steps": cumulative,
                    "optimizer_log_loss": step["train_log_loss"],
                    "train_log_loss": fit_loss,
                    "development_log_loss": dev_loss,
                }
            )
            if epoch in (1, epochs):
                torch.save(model.state_dict(), directory / f"seed_{seed}_epoch_{epoch}.pt")
            if epoch == 1:
                epoch_one[probability_column(trial, seed)] = p
            emit(
                reporter,
                "fit",
                "active",
                f"{trial} · seed {seed} · epoch {epoch}/{epochs} · dev Log Loss {dev_loss:.5f}",
                trial=trial,
                seed=seed,
                iterations=epoch,
                optimizer_steps=cumulative,
                train_log_loss=fit_loss,
                development_log_loss=dev_loss,
            )
        runs[str(seed)] = {
            "epochs": epochs,
            "optimizer_steps": cumulative,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "first_step": telemetry[0]["first_step"],
            "seconds": sum(step["seconds"] for step in telemetry),
            "peak_allocated_mib": max(step["peak_allocated_mib"] for step in telemetry),
        }
    return rows, epoch_one, runs


def probability_column(trial: str, seed: int) -> str:
    return f"{trial}_seed_{seed}_epoch_1"


def run(out: Path, reporter: ProgressReporter) -> None:
    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_bytes(PLAN.read_bytes())
    plan = read_json(out / "plan.json")
    run_seeds = [plan["primary_training_seed"], *plan["sensitivity_training_seeds"]]
    epochs = plan["learning_curve_epochs"]
    emit(reporter, "load", "active", "Verifying the previous 32-map development inputs.")
    prior = ROOT / plan["source_run"]
    if sha256_file(prior / "selection.json") != plan["source_selection_sha256"]:
        raise PipelineError("The previous development-run receipt differs from the plan.")
    source_plan, source_receipt = locked_selection(prior)
    temporal, archived_fit = fit_inputs()
    fit32 = archived_fit.filter(pl.col("event").is_in(plan["event_order"]))
    spec = source_plan["development"]
    _, development = load_v1(ROOT, ROOT / spec["dataset"], spec["dataset_sha256"])
    check_scope(development, spec)
    check_scope(fit32, source_plan["development_fit"])
    verify_partition_order({"fit": fit32, "development": development})
    trials, manifest = make_splits(fit32, plan)
    write_json(
        out / "split_manifest.json",
        {
            "frozen_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256_file(out / "plan.json"),
            "trials": manifest,
            "development": counts(development),
        },
    )

    prior_directory = prior / "development"
    prior_predictions = pl.read_parquet(prior_directory / "predictions.parquet")
    prior_curves = pl.read_csv(prior_directory / "curves.csv").filter(pl.col("epoch") <= epochs)
    index = target_index(development)
    if (
        not index.equals(prior_predictions.select(index.columns))
        or not target_index(fit32).equals(pl.read_parquet(prior_directory / "fit_targets.parquet"))
        or fit_preprocessing(fit32) != read_json(prior_directory / "preprocessing.json")
    ):
        raise PipelineError("The archived 32-match fit or the 10-match target index changed.")
    device, evidence = cuda_device(temporal)
    emit(reporter, "load", "completed", "Locked the nested match IDs and verified archival rows.")

    all_curves, predictions, receipts = [], {}, {}
    for trial, frame in trials.items():
        directory = out / trial
        directory.mkdir()
        preproc = fit_preprocessing(frame)
        fit_values, dev_values = transform(frame, preproc), transform(development, preproc)
        logistic = fit_logistic(frame, fit_values, temporal)
        write_json(directory / "preprocessing.json", preproc)
        write_json(directory / "logistic.json", logistic)
        write_parquet(target_index(frame), directory / "fit_targets.parquet")
        predictions[f"{trial}_logistic"] = logistic_predict(logistic, dev_values)
        if trial == "full_n32":
            steps = math.ceil(frame.height / temporal["training"]["batch_size"])
            for seed in run_seeds:
                record = source_receipt["seeds"][str(seed)]
                if record["optimizer_steps"] != record["epochs"] * steps:
                    raise PipelineError(f"The archived step count differs for seed {seed}.")
                shutil.copyfile(
                    prior_directory / f"seed_{seed}" / "epoch_1.pt",
                    directory / f"seed_{seed}_epoch_1.pt",
                )
                predictions[probability_column(trial, seed)] = prior_predictions[
                    f"seed_{seed}_epoch_1"
                ].to_numpy()
            all_curves.extend(
                prior_curves.select(
                    pl.lit(trial).alias("trial"),
                    "seed",
                    "epoch",
                    (pl.col("epoch") * steps).alias("optimizer_steps"),
                    "optimizer_log_loss",
                    "train_log_loss",
                    "development_log_loss",
                ).to_dicts()
            )
            receipts[trial] = {
                "source": plan["source_run"],
                "source_selection_sha256": plan["source_selection_sha256"],
                "reused_epochs": epochs,
                "optimizer_steps_per_seed_through_epoch_20": epochs * steps,
            }
            emit(reporter, "fit", "active", "Reused verified 32-match CUDA curves and predictions.")
            continue
        curves, per_seed_probabilities, receipt = fit_prefixes(
            directory,
            trial,
            frame,
            fit_values,
            development,
            dev_values,
            temporal,
            run_seeds,
            epochs,
            device,
            reporter,
        )
        all_curves.extend(curves)
        predictions.update(per_seed_probabilities)
        receipts[trial] = {"seeds": receipt}
    pl.DataFrame(all_curves).write_csv(out / "curves.csv")
    write_parquet(
        index.with_columns(*(pl.Series(name, p) for name, p in predictions.items())),
        out / "development_predictions.parquet",
    )
    write_json(
        out / "fit.json",
        {
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "device": evidence,
            "wall_seconds": time.perf_counter() - started,
            "trials": receipts,
            "artifacts": hashes(out),
        },
    )
    emit(
        reporter, "fit", "completed", "Frozen all matched fits and aligned development predictions."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file)
    try:
        run(args.out, reporter)
    except (PipelineError, OSError, ValueError, ImportError, RuntimeError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

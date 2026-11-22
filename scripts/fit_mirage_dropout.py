"""Train the 0.20 Mirage candidate and recover the matched 0.10 checkpoints."""

from __future__ import annotations

import argparse
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
from cs2_pipeline.train import round_weights
from scripts.fit_matched34_transformers import fit_inputs, read_json
from scripts.fit_mirage_snapshot_epochs import (
    PLAN as EPOCH_PLAN,
)
from scripts.fit_mirage_snapshot_epochs import (
    check_scope,
    cuda_device,
    hashes,
    seed_order,
    target_index,
    verify_hashes,
    write_parquet,
)
from scripts.fit_mirage_train_expansion import extend_fit

PLAN = ROOT / "configs/mirage_dropout.json"


def model_config(temporal: dict[str, Any], dropout: float) -> dict[str, Any]:
    return {**temporal, "transformer": {**temporal["transformer"], "dropout": dropout}}


def load_inputs(
    plan: dict[str, Any],
) -> tuple[dict[str, Any], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    source = ROOT / plan["source_run"]
    verify_hashes(source, plan["source_hashes"])
    release = read_json(source / "prospective_freeze.json")
    verify_hashes(
        source,
        {
            "plan.json": release["plan_sha256"],
            "selection.json": release["selection_sha256"],
            "split_manifest.json": release["split_manifest_sha256"],
            "curves.csv": read_json(source / "selection.json")["curves_sha256"],
        },
    )
    verify_hashes(source / "final_fit", release["artifacts"])
    split = read_json(source / "split_manifest.json")
    intake_path = ROOT / plan["source_intake"]
    if sha256_file(intake_path) != split["intake_sha256"]:
        raise PipelineError("The expansion intake receipt changed.")
    intake = read_json(intake_path)
    temporal, fit34 = fit_inputs()
    if (
        temporal["transformer"] != release["transformer"]
        or temporal["transformer"]["dropout"] != plan["dropout"]["d010"]
        or temporal["logistic_c"] != plan["logistic_c"]
    ):
        raise PipelineError("The original model settings differ from the frozen expansion.")
    spec = read_json(EPOCH_PLAN)
    fit32 = fit34.filter(pl.col("event").is_in(spec["development_fit"]["events"]))
    _, extra = load_v1(ROOT, ROOT / intake["dataset"], intake["dataset_sha256"])
    dev_spec = spec["development"]
    _, dev = load_v1(ROOT, ROOT / dev_spec["dataset"], dev_spec["dataset_sha256"])
    check_scope(dev, dev_spec)
    fit, final = extend_fit(fit32, extra), extend_fit(fit34, extra)
    for role, frame, expected in (
        ("development_fit", fit, plan["fit_matches"]),
        ("development", dev, plan["development_matches"]),
        ("final_fit", final, plan["final_fit_matches"]),
    ):
        match_ids = sorted(frame["match_id"].unique().to_list())
        if match_ids != split[role] or len(match_ids) != expected:
            raise PipelineError(f"Unexpected expansion match roster: {role}.")
    porto = final.join(fit.select("sample_id"), on="sample_id", how="anti")
    verify_partition_order({"fit": fit, "development": dev, "porto": porto})
    previous_dev = pl.read_parquet(source / "development_predictions.parquet")
    dev_index = target_index(dev)
    if (
        not dev_index.equals(previous_dev.select(dev_index.columns))
        or not target_index(fit).equals(pl.read_parquet(source / "development/fit_targets.parquet"))
        or not target_index(final).equals(pl.read_parquet(source / "final_fit/fit_targets.parquet"))
    ):
        raise PipelineError("An exact archived target or round weight differs.")
    return temporal, fit, dev, final


def select_shared_epoch(curve: pl.DataFrame, seeds: list[int], epochs: int) -> int:
    for seed in seeds:
        rows = curve.filter(pl.col("seed") == seed).sort("epoch")
        if rows["epoch"].to_list() != list(range(1, epochs + 1)):
            raise PipelineError(f"Incomplete per-epoch losses for seed {seed}.")
    if curve.height != len(seeds) * epochs or not curve["development_log_loss"].is_finite().all():
        raise PipelineError("Unexpected seed or nonfinite development loss.")
    mean_curve = curve.group_by("epoch").agg(pl.col("development_log_loss").mean()).sort("epoch")
    return mean_curve["epoch"][mean_curve["development_log_loss"].arg_min()]


def fit_curves(
    arm: str,
    temporal: dict[str, Any],
    fit: pl.DataFrame,
    dev: pl.DataFrame,
    preprocessing: dict[str, Any],
    seeds: list[int],
    epochs: int,
    device: Any,
    reporter: ProgressReporter,
) -> tuple[pl.DataFrame, dict, dict]:
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, predict, train_epoch

    fit_values, dev_values = transform(fit, preprocessing), transform(dev, preprocessing)
    fit_data = DeviceWindows.from_arrays(
        *network_arrays(fit, fit_values, window_indices(fit, 1), 1), device
    )
    dev_data = DeviceWindows.from_arrays(
        *network_arrays(dev, dev_values, window_indices(dev, 1), 1), device
    )
    training = temporal["training"]
    batch = training["batch_size"]
    fit_y, dev_y = fit["y_ct_win"].to_numpy(), dev["y_ct_win"].to_numpy()
    fit_w, dev_w = round_weights(fit), round_weights(dev)
    rows, checkpoints, runs = [], {}, {}
    for seed in seeds:
        torch.manual_seed(seed)
        shuffle = torch.Generator().manual_seed(seed)
        model = WindowTransformer(
            fit_values.shape[1], max(temporal["window_lengths"]), temporal["transformer"]
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            foreach=False,
            fused=False,
        )
        telemetry = []
        steps = 0
        for epoch in range(1, epochs + 1):
            step = train_epoch(model, optimizer, fit_data, batch, shuffle, verify_step=epoch == 1)
            telemetry.append(step)
            steps += step["steps"]
            probabilities = sigmoid(predict(model, dev_data, batch))
            dev_loss = float(log_loss(dev_y, probabilities, labels=[0, 1], sample_weight=dev_w))
            train_loss = float(
                log_loss(
                    fit_y,
                    sigmoid(predict(model, fit_data, batch)),
                    labels=[0, 1],
                    sample_weight=fit_w,
                )
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "train_log_loss": train_loss,
                    "development_log_loss": dev_loss,
                }
            )
            checkpoints[seed, epoch] = (
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                probabilities,
            )
            emit(
                reporter,
                "fit",
                "active",
                f"{arm} · seed {seed} · epoch {epoch}/{epochs} · dev Log Loss {dev_loss:.5f}",
                arm=arm,
                seed=seed,
                iterations=epoch,
                optimizer_steps=steps,
                train_log_loss=train_loss,
                development_log_loss=dev_loss,
            )
        runs[str(seed)] = {
            "optimizer_steps": steps,
            "epochs": epochs,
            "first_step": telemetry[0]["first_step"],
            "seconds": sum(row["seconds"] for row in telemetry),
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in telemetry),
        }
    return pl.DataFrame(rows), checkpoints, runs


def probability_column(arm: str, role: str, seed: int) -> str:
    return f"{arm}_{role}_seed_{seed}"


def run(out: Path, reporter: ProgressReporter) -> None:
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, predict

    started = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_bytes(PLAN.read_bytes())
    plan = read_json(out / "plan.json")
    emit(reporter, "load", "active", "Verifying the exact archived 47/10/49 match and sample IDs.")
    temporal, fit, dev, final = load_inputs(plan)
    source = ROOT / plan["source_run"]
    shutil.copyfile(source / "split_manifest.json", out / "split_manifest.json")
    index = target_index(dev)
    previous = pl.read_parquet(source / "development_predictions.parquet")
    preprocessing = fit_preprocessing(fit)
    if preprocessing != read_json(source / "development/preprocessing.json"):
        raise PipelineError("Fit-only preprocessing differs from the archived 47-match fit.")
    values = transform(dev, preprocessing)
    logistic = fit_logistic(fit, transform(fit, preprocessing), temporal)
    log_p = logistic_predict(logistic, values)
    if not np.array_equal(log_p, previous["expanded_logistic"].to_numpy()):
        raise PipelineError("Refitted logistic probabilities differ from the archived baseline.")
    write_json(out / "preprocessing.json", preprocessing)
    write_json(out / "logistic.json", logistic)
    write_parquet(target_index(fit), out / "fit_targets.parquet")
    seeds, epochs = seed_order(plan), plan["epochs"]
    device, hardware = cuda_device(temporal)
    emit(reporter, "load", "completed", "Frozen the comparison plan and exact shared samples.")
    reference = pl.read_csv(source / "curves.csv").select(
        pl.lit("d010").alias("arm"),
        "seed",
        "epoch",
        "optimizer_steps",
        "train_log_loss",
        "development_log_loss",
    )
    new, candidate, telemetry = fit_curves(
        "d020",
        model_config(temporal, plan["dropout"]["d020"]),
        fit,
        dev,
        preprocessing,
        seeds,
        epochs,
        device,
        reporter,
    )
    chosen = {
        "d010": select_shared_epoch(reference, seeds, epochs),
        "d020": select_shared_epoch(new, seeds, epochs),
    }
    originals, original_telemetry = {}, {}
    if chosen["d010"] != 1:
        replay, originals, original_telemetry = fit_curves(
            "d010",
            temporal,
            fit,
            dev,
            preprocessing,
            seeds,
            chosen["d010"],
            device,
            reporter,
        )
        archived_prefix = reference.filter(pl.col("epoch") <= chosen["d010"])
        keys = ("arm", "seed", "epoch", "optimizer_steps")
        losses = ("train_log_loss", "development_log_loss")
        if not archived_prefix.select(*keys).equals(replay.select(*keys)) or not np.allclose(
            archived_prefix.select(*losses).to_numpy(),
            replay.select(*losses).to_numpy(),
            rtol=0,
            atol=1e-12,
        ):
            raise PipelineError("The 0.10 replay did not reproduce the archived curve prefix.")
        for seed in seeds:
            if not np.array_equal(
                originals[seed, 1][1], previous[f"expanded_seed_{seed}_epoch_1"].to_numpy()
            ):
                raise PipelineError(f"The 0.10 replay changed epoch-1 predictions for seed {seed}.")
    comparison = out / "development"
    comparison.mkdir()
    dev_data = DeviceWindows.from_arrays(
        *network_arrays(dev, values, window_indices(dev, 1), 1), device
    )
    # Dropout is disabled for inference by predict(), so the same module can check both arms.
    model = WindowTransformer(
        values.shape[1], max(temporal["window_lengths"]), temporal["transformer"]
    ).to(device)
    predictions = {"logistic": log_p}
    for arm in ("d010", "d020"):
        directory = comparison / arm
        directory.mkdir()
        for seed in seeds:
            for role, epoch in (("epoch_1", 1), ("selected", chosen[arm])):
                path = directory / f"seed_{seed}_epoch_{epoch}.pt"
                if arm == "d010" and epoch == 1:
                    archived = source / "development" / path.name
                    if not path.exists():
                        shutil.copyfile(archived, path)
                    expected = previous[f"expanded_seed_{seed}_epoch_1"].to_numpy()
                else:
                    state, expected = (originals if arm == "d010" else candidate)[seed, epoch]
                    if not path.exists():
                        torch.save(state, path)
                model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                p = sigmoid(predict(model, dev_data, temporal["training"]["batch_size"]))
                if not np.array_equal(p, expected):
                    raise PipelineError(f"Inference changed on reload: {arm}/{seed}/{epoch}.")
                predictions[probability_column(arm, role, seed)] = p
    pl.concat([reference, new]).write_csv(out / "curves.csv")
    write_parquet(
        index.with_columns(*(pl.Series(name, p) for name, p in predictions.items())),
        out / "development_predictions.parquet",
    )
    write_json(
        out / "development_fit.json",
        {
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "time_seconds": time.perf_counter() - started,
            "device": hardware,
            "fit": counts(fit),
            "development": counts(dev),
            "planned_final_fit": counts(final),
            "shared_epochs": chosen,
            "fresh_candidate_telemetry": telemetry,
            "baseline_replay_telemetry": original_telemetry,
            "artifacts": hashes(out),
        },
    )
    emit(reporter, "fit", "completed", "Saved both arms and verified all scored checkpoints.")


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

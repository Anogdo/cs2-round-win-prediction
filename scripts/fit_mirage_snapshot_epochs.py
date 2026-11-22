"""Select an epoch on 32-to-10 Mirage matches, then refit on the original 34."""

from __future__ import annotations

import argparse
import os
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
    PLAN as TEMPORAL_PLAN,
)
from cs2_pipeline.phase2_temporal import (
    driver_version,
    fit_logistic,
    fit_preprocessing,
    network_arrays,
    transform,
)
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import ROUND_KEY, load_v1, verify_partition_order, window_indices
from cs2_pipeline.train import round_weights
from scripts.fit_matched34_transformers import fit_inputs, read_json

PLAN = ROOT / "configs/mirage_snapshot_epochs.json"
CHECKPOINTS = ("epoch_1", "selected", "epoch_200")


def seed_order(plan: dict[str, Any]) -> list[int]:
    return [plan["primary_seed"], *plan["sensitivity_seeds"]]


def check_scope(frame: pl.DataFrame, spec: dict[str, Any]) -> None:
    summary = counts(frame)
    if any(summary[key] != value for key, value in spec["counts"].items()) or (
        "events" in spec and set(frame["event"]) != set(spec["events"])
    ):
        raise PipelineError("The split differs from the epoch experiment plan.")


def target_index(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        "sample_id", "event", *ROUND_KEY, "match_date", "tick", "y_ct_win"
    ).with_columns(pl.Series("round_weight", round_weights(frame)))


def write_parquet(table: pl.DataFrame, path: Path) -> None:
    table.write_parquet(path)
    if not table.equals(pl.read_parquet(path)):
        raise PipelineError(f"Parquet round-trip differs: {path}")


def hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def verify_hashes(directory: Path, artifacts: dict[str, str]) -> None:
    for name, digest in artifacts.items():
        if sha256_file(directory / name) != digest:
            raise PipelineError(f"Frozen artifact has changed: {directory / name}")


def locked_selection(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = read_json(out / "plan.json")
    selection = read_json(out / "selection.json")
    if selection["plan_sha256"] != sha256_file(out / "plan.json"):
        raise PipelineError("The saved experiment plan changed after selection.")
    verify_hashes(out / "development", selection["artifacts"])
    curves = pl.read_csv(out / "development/curves.csv")
    for seed in seed_order(plan):
        rows = curves.filter(pl.col("seed") == seed).sort("epoch")
        earliest = rows["development_log_loss"].arg_min()
        if rows["epoch"].to_list() != list(range(1, plan["epochs"] + 1)) or (
            rows["epoch"][earliest] != selection["seeds"][str(seed)]["selected_epoch"]
        ):
            raise PipelineError(
                f"The locked epoch disagrees with the development curve for {seed}."
            )
    return plan, selection


def cuda_device(temporal: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from cs2_pipeline.temporal_net import cuda_environment

    device, evidence = cuda_environment(temporal["training"]["device"])
    if "RTX 4070 Ti" not in evidence["gpu_name"]:
        raise PipelineError("The required RTX 4070 Ti is unavailable.")
    evidence["nvidia_driver"] = driver_version()
    return device, evidence


def fit_trajectories(
    out: Path,
    frame: pl.DataFrame,
    plan: dict[str, Any],
    temporal: dict[str, Any],
    reporter: ProgressReporter | None,
    *,
    development: pl.DataFrame | None = None,
    chosen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, predict, train_epoch

    started = time.perf_counter()
    device, evidence = cuda_device(temporal)
    training = temporal["training"]
    batch_size = training["batch_size"]
    max_length = max(temporal["window_lengths"])
    preprocessing = fit_preprocessing(frame)
    values = transform(frame, preprocessing)
    write_json(out / "preprocessing.json", preprocessing)
    train_data = DeviceWindows.from_arrays(
        *network_arrays(frame, values, window_indices(frame, 1), 1), device
    )
    train_targets = target_index(frame)
    write_parquet(train_targets, out / "fit_targets.parquet")
    if development is None:
        write_json(out / "logistic.json", fit_logistic(frame, values, temporal))
        dev_data, dev_predictions = None, None
    else:
        dev_data = DeviceWindows.from_arrays(
            *network_arrays(
                development,
                transform(development, preprocessing),
                window_indices(development, 1),
                1,
            ),
            device,
        )
        dev_predictions = target_index(development)

    def held_loss(inputs: DeviceWindows, targets: pl.DataFrame) -> tuple[float, np.ndarray]:
        p = sigmoid(predict(model, inputs, batch_size))
        return (
            float(
                log_loss(
                    targets["y_ct_win"], p, labels=[0, 1], sample_weight=targets["round_weight"]
                )
            ),
            p,
        )

    rows, runs = [], {}
    stage = "Development" if development is not None else "Final fit"
    for seed in seed_order(plan):
        directory = out / f"seed_{seed}"
        directory.mkdir()
        torch.manual_seed(seed)
        shuffle = torch.Generator().manual_seed(seed)
        model = WindowTransformer(values.shape[1], max_length, temporal["transformer"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            foreach=False,
            fused=False,
        )
        best_epoch = chosen[str(seed)]["selected_epoch"] if chosen else 0
        best_loss = float("inf")
        saved = {}
        telemetry = []
        for epoch in range(1, plan["epochs"] + 1):
            step = train_epoch(
                model, optimizer, train_data, batch_size, shuffle, verify_step=epoch == 1
            )
            telemetry.append(step)
            row = {"seed": seed, "epoch": epoch, "optimizer_log_loss": step["train_log_loss"]}
            if dev_data is not None:
                fit_loss, _ = held_loss(train_data, train_targets)
                dev_loss, dev_p = held_loss(dev_data, dev_predictions)
                row.update(train_log_loss=fit_loss, development_log_loss=dev_loss)
                if dev_loss < best_loss:
                    best_epoch, best_loss = epoch, dev_loss
                    torch.save(model.state_dict(), directory / "selected.pt")
                    saved["selected"] = dev_p
                if epoch in (1, plan["epochs"]):
                    saved[f"epoch_{epoch}"] = dev_p
            elif epoch == best_epoch:
                torch.save(model.state_dict(), directory / "selected.pt")
            if epoch in (1, plan["epochs"]):
                torch.save(model.state_dict(), directory / f"epoch_{epoch}.pt")
            rows.append(row)
            suffix = f" · dev Log Loss {row['development_log_loss']:.5f}" if dev_data else ""
            emit(
                reporter,
                "fit",
                "active",
                f"{stage} · seed {seed} · epoch {epoch}/{plan['epochs']}{suffix}",
                iterations=epoch,
                seed=seed,
                total=plan["epochs"],
                selected_epoch=best_epoch,
                **{key: value for key, value in row.items() if key not in ("seed", "epoch")},
            )
        summary = {
            "selected_epoch": best_epoch,
            "epochs": len(telemetry),
            "optimizer_steps": sum(step["steps"] for step in telemetry),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "first_step": telemetry[0]["first_step"],
            "training_seconds": sum(step["seconds"] for step in telemetry),
            "peak_allocated_mib": max(step["peak_allocated_mib"] for step in telemetry),
        }
        if dev_data is not None:
            for name in CHECKPOINTS:
                model.load_state_dict(
                    torch.load(directory / f"{name}.pt", map_location="cpu", weights_only=True)
                )
                reloaded = sigmoid(predict(model, dev_data, batch_size))
                if not np.array_equal(reloaded, saved[name]):
                    raise PipelineError(f"Development checkpoint changed on reload: {seed}/{name}")
                dev_predictions = dev_predictions.with_columns(
                    pl.Series(f"seed_{seed}_{name}", reloaded)
                )
            summary["selected_development_log_loss"] = best_loss
        runs[str(seed)] = summary
    pl.DataFrame(rows).write_csv(out / "curves.csv")
    if dev_predictions is not None:
        write_parquet(dev_predictions, out / "predictions.parquet")
    return {
        "fit": counts(frame),
        "device": evidence,
        "seeds": runs,
        "wall_seconds": time.perf_counter() - started,
        "artifacts": hashes(out),
    }


def develop(out: Path, reporter: ProgressReporter | None) -> None:
    out.mkdir(parents=True, exist_ok=False)
    plan = read_json(PLAN)
    (out / "plan.json").write_bytes(PLAN.read_bytes())
    emit(reporter, "load", "active", "Verifying frozen development and 34-map fit manifests.")
    temporal, full = fit_inputs()
    if sha256_file(TEMPORAL_PLAN) != plan["temporal_plan_sha256"]:
        raise PipelineError("The matched training settings changed.")
    fit_spec, dev_spec = plan["development_fit"], plan["development"]
    fit = full.filter(pl.col("event").is_in(fit_spec["events"]))
    meta, dev = load_v1(ROOT, ROOT / dev_spec["dataset"], dev_spec["dataset_sha256"])
    check_scope(full, plan["final_fit"])
    check_scope(fit, fit_spec)
    check_scope(dev, dev_spec)
    if meta["split_policy"]["method"] != "historical_validation_only":
        raise PipelineError("The 10 matches must be reserved for development.")
    # The 10 development matches fall between Cologne and Porto, but overlap neither.
    verify_partition_order(
        {
            "fit": fit,
            "development": dev,
            "porto": full.filter(~pl.col("event").is_in(fit_spec["events"])),
        }
    )
    emit(reporter, "load", "completed", "Verified 32 fit matches and 10 development matches.")
    directory = out / "development"
    directory.mkdir()
    result = fit_trajectories(directory, fit, plan, temporal, reporter, development=dev)
    write_json(
        out / "selection.json",
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256_file(out / "plan.json"),
            "development": counts(dev),
            **result,
        },
    )
    emit(
        reporter,
        "fit",
        "active",
        "All development choices locked before refitting.",
        selected={seed: data["selected_epoch"] for seed, data in result["seeds"].items()},
    )


def refit(out: Path, reporter: ProgressReporter | None) -> None:
    plan, selection = locked_selection(out)
    temporal, full = fit_inputs()
    check_scope(full, plan["final_fit"])
    directory = out / "final_fit"
    directory.mkdir(exist_ok=False)
    result = fit_trajectories(directory, full, plan, temporal, reporter, chosen=selection["seeds"])
    write_json(
        out / "final_fit.json",
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "selection_sha256": sha256_file(out / "selection.json"),
            **result,
        },
    )
    emit(reporter, "fit", "completed", "Frozen 34-match checkpoints and matched logistic model.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("develop", "refit", "both"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file)
    try:
        if args.stage in ("develop", "both"):
            develop(args.out, reporter)
        if args.stage in ("refit", "both"):
            refit(args.out, reporter)
    except (PipelineError, OSError, ValueError, ImportError, RuntimeError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

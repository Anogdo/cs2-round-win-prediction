"""Fit matched Mirage models with new training events and freeze Singapore candidates."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.calibrate import metrics
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
    seed_order,
    target_index,
    verify_hashes,
    write_parquet,
)
from scripts.fit_mirage_train_size import fit_prefixes, probability_column
from scripts.import_mirage_train_expansion import PLAN
from scripts.score_mirage_train_size import interval, match_bootstrap_scores


def extend_fit(base: pl.DataFrame, extra: pl.DataFrame) -> pl.DataFrame:
    for key in ("match_id", "event", "map_id", "demo_sha256", "sample_id"):
        if set(base[key]) & set(extra[key]):
            raise PipelineError(f"New replay data duplicates existing {key}.")
    return pl.concat([base, extra.select(base.columns)]).sort("sample_id")


def fit_final(
    out: Path,
    frame: pl.DataFrame,
    temporal: dict[str, Any],
    selected_epochs: dict[int, int],
    device: Any,
    reporter: ProgressReporter,
) -> dict[str, list[dict[str, Any]]]:
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, train_epoch

    out.mkdir()
    preprocessing = fit_preprocessing(frame)
    values = transform(frame, preprocessing)
    write_json(out / "preprocessing.json", preprocessing)
    write_parquet(target_index(frame), out / "fit_targets.parquet")
    write_json(out / "logistic.json", fit_logistic(frame, values, temporal))
    data = DeviceWindows.from_arrays(
        *network_arrays(frame, values, window_indices(frame, 1), 1), device
    )
    training = temporal["training"]
    runs = {}
    for seed, stop in selected_epochs.items():
        torch.manual_seed(seed)
        shuffle = torch.Generator().manual_seed(seed)
        model = WindowTransformer(
            values.shape[1], max(temporal["window_lengths"]), temporal["transformer"]
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
            foreach=False,
            fused=False,
        )
        runs[str(seed)] = []
        for epoch in range(1, stop + 1):
            step = train_epoch(
                model, optimizer, data, training["batch_size"], shuffle, verify_step=epoch == 1
            )
            runs[str(seed)].append(step)
            emit(
                reporter,
                "fit",
                "active",
                f"Final fit · seed {seed} · epoch {epoch}/{stop}",
                seed=seed,
                iterations=epoch,
                optimizer_steps=sum(s["steps"] for s in runs[str(seed)]),
            )
            if epoch in (1, stop):
                checkpoint = out / f"seed_{seed}_epoch_{epoch}.pt"
                torch.save(model.state_dict(), checkpoint)
                restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
                if any(
                    not torch.equal(value.cpu(), restored[key])
                    for key, value in model.state_dict().items()
                ):
                    raise PipelineError(f"Reloaded weights differ from {checkpoint.name}.")
    return runs


def verify_epoch_one(
    directory: Path,
    frame: pl.DataFrame,
    values: np.ndarray,
    probabilities: dict[str, np.ndarray],
    temporal: dict[str, Any],
    device: Any,
    seeds: list[int],
) -> None:
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, predict

    data = DeviceWindows.from_arrays(
        *network_arrays(frame, values, window_indices(frame, 1), 1), device
    )
    model = WindowTransformer(
        values.shape[1], max(temporal["window_lengths"]), temporal["transformer"]
    ).to(device)
    for seed in seeds:
        model.load_state_dict(
            torch.load(directory / f"seed_{seed}_epoch_1.pt", map_location="cpu", weights_only=True)
        )
        restored = sigmoid(predict(model, data, temporal["training"]["batch_size"]))
        if not np.array_equal(restored, probabilities[probability_column("expanded", seed)]):
            raise PipelineError(f"Expanded development probabilities changed on reload for {seed}.")


def score_development(
    predictions: pl.DataFrame, seeds: list[int], bootstrap: dict[str, Any]
) -> list[dict[str, Any]]:
    columns = [
        name
        for trial in ("full_n32", "expanded")
        for name in (f"{trial}_logistic", *(probability_column(trial, seed) for seed in seeds))
    ]
    scores, draws = match_bootstrap_scores(predictions, columns, bootstrap)
    rows = []
    for trial in ("full_n32", "expanded"):
        baseline = f"{trial}_logistic"
        evaluated = metrics(predictions, predictions[baseline].to_list())
        row = {
            "fit": trial,
            "model": "logistic",
            "seed": None,
            **{k: evaluated[k] for k in ("log_loss", "brier_score", "roc_auc")},
        }
        rows.append(row)
        for seed in seeds:
            candidate = probability_column(trial, seed)
            evaluated = metrics(predictions, predictions[candidate].to_list())
            gap = draws[candidate] - draws[baseline]
            row = {
                "fit": trial,
                "model": "snapshot_epoch_1",
                "seed": seed,
                **{k: evaluated[k] for k in ("log_loss", "brier_score", "roc_auc")},
                "tf_minus_logistic": scores[candidate] - scores[baseline],
                **interval(gap),
            }
            if trial == "expanded":
                old = probability_column("full_n32", seed)
                row.update(
                    expansion_change_in_gap=row["tf_minus_logistic"]
                    - (scores[old] - scores["full_n32_logistic"]),
                    **{
                        f"change_{k}": v
                        for k, v in interval(
                            gap - (draws[old] - draws["full_n32_logistic"])
                        ).items()
                    },
                )
            rows.append(row)
    return rows


def report_text(report: dict[str, Any]) -> str:
    lines = [
        "# Mirage training expansion",
        "",
        f"The fit grew from 32 to {report['development_fit']['matches']} matches for development.",
        f"The locked final fit uses {report['final_fit']['matches']} matches.",
        "Both cohorts are scored on the same 10 already observed Bounty / EWC matches.",
        "The additional recordings change both training volume and event mix.",
        "",
        "| Fit | Candidate | Log Loss | Brier | AUC | Δ to logistic [95% interval] |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["development_scores"]:
        name = "Logistic" if row["seed"] is None else f"Transformer epoch 1, seed {row['seed']}"
        fit = 32 if row["fit"] == "full_n32" else report["development_fit"]["matches"]
        gap = (
            "—"
            if row["seed"] is None
            else (
                f"{row['tf_minus_logistic']:+.5f} [{row['ci95_low']:+.5f}, {row['ci95_high']:+.5f}]"
            )
        )
        lines.append(
            f"| {fit} | {name} | {row['log_loss']:.5f} | "
            f"{row['brier_score']:.5f} | {row['roc_auc']:.5f} | {gap} |"
        )
    lines.extend(
        [
            "",
            "Expansion change is the expanded-fit gap minus the 32-match gap."
            " Both gaps use the same matches in each of 2,000 paired bootstrap draws.",
            "",
        ]
    )
    for row in report["development_scores"]:
        if "expansion_change_in_gap" in row:
            lines.append(
                f"- Seed {row['seed']} {row['expansion_change_in_gap']:+.5f} "
                f"[{row['change_ci95_low']:+.5f}, {row['change_ci95_high']:+.5f}]"
            )
    lines.extend(
        [
            "",
            f"Development-selected epochs by seed are {report['selected_epochs']}.",
            "The epoch-1 checkpoints remain the primary prospective Transformer comparators.",
            "See [curves](curves.csv), [aligned probabilities](development_predictions.parquet), "
            "and the [prospective freeze](prospective_freeze.json).",
            "",
        ]
    )
    if report["missing_match_ids"]:
        lines.append("Missing planned archives " + ", ".join(report["missing_match_ids"]) + ".")
    return "\n".join(lines) + "\n"


def run(out: Path, intake_path: Path, reporter: ProgressReporter) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_bytes(PLAN.read_bytes())
    plan = read_json(out / "plan.json")
    intake = read_json(intake_path)
    if intake["plan_sha256"] != sha256_file(out / "plan.json") or intake["failed"]:
        raise PipelineError("The supplemental dataset lacks a valid intake receipt.")
    emit(reporter, "load", "active", "Checking V1 source tables and freezing fit membership.")
    temporal, fit34 = fit_inputs()
    epoch_plan = read_json(ROOT / "configs/mirage_snapshot_epochs.json")
    fit32 = fit34.filter(pl.col("event").is_in(epoch_plan["development_fit"]["events"]))
    check_scope(fit32, epoch_plan["development_fit"])
    spec = epoch_plan["development"]
    _, development = load_v1(ROOT, ROOT / spec["dataset"], spec["dataset_sha256"])
    check_scope(development, spec)
    _, extra = load_v1(ROOT, ROOT / intake["dataset"], intake["dataset_sha256"])
    if counts(extra)["matches"] != intake["map_count"] or set(extra["match_id"]) != {
        item["match_id"] for item in intake["matches"]
    }:
        raise PipelineError("The supplemental sample IDs differ from the verified roster.")
    expanded_dev, expanded_final = extend_fit(fit32, extra), extend_fit(fit34, extra)
    porto = fit34.filter(~pl.col("match_id").is_in(fit32["match_id"].unique().to_list()))
    verify_partition_order({"fit": expanded_dev, "development": development, "porto": porto})

    previous = ROOT / plan["source_run"]
    if sha256_file(previous / "fit.json") != plan["source_fit_sha256"]:
        raise PipelineError("The previous 32-match comparison differs from the plan.")
    verify_hashes(previous, read_json(previous / "fit.json")["artifacts"])
    reference = pl.read_parquet(previous / "development_predictions.parquet")
    index = target_index(development)
    if not index.equals(reference.select(index.columns)) or not target_index(fit32).equals(
        pl.read_parquet(previous / "full_n32/fit_targets.parquet")
    ):
        raise PipelineError("The 32-match fit IDs or the development target IDs changed.")
    write_json(
        out / "split_manifest.json",
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "intake_sha256": sha256_file(intake_path),
            "development_fit": sorted(set(expanded_dev["match_id"])),
            "development": sorted(set(development["match_id"])),
            "final_fit": sorted(set(expanded_final["match_id"])),
            "missing_planned_matches": intake["missing"],
        },
    )
    emit(
        reporter,
        "load",
        "completed",
        f"Locked {counts(expanded_dev)['matches']} development-fit matches and "
        f"{counts(expanded_final)['matches']} final-fit matches.",
    )

    seeds = seed_order(plan)
    device, hardware = cuda_device(temporal)
    directory = out / "development"
    directory.mkdir()
    preprocessing = fit_preprocessing(expanded_dev)
    values = transform(expanded_dev, preprocessing)
    validation_values = transform(development, preprocessing)
    logistic = fit_logistic(expanded_dev, values, temporal)
    write_json(directory / "preprocessing.json", preprocessing)
    write_json(directory / "logistic.json", logistic)
    write_parquet(target_index(expanded_dev), directory / "fit_targets.parquet")
    curves, probabilities, telemetry = fit_prefixes(
        directory,
        "expanded",
        expanded_dev,
        values,
        development,
        validation_values,
        temporal,
        seeds,
        plan["epochs"],
        device,
        reporter,
    )
    verify_epoch_one(
        directory, development, validation_values, probabilities, temporal, device, seeds
    )
    curve = pl.DataFrame(curves)
    curve.write_csv(out / "curves.csv")
    selected = {}
    minima = {}
    for seed in seeds:
        seed_curve = curve.filter(pl.col("seed") == seed).sort("epoch")
        best = seed_curve.row(seed_curve["development_log_loss"].arg_min(), named=True)
        selected[seed] = best["epoch"]
        minima[str(seed)] = best["development_log_loss"]
    write_json(
        out / "selection.json",
        {
            "locked_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256_file(out / "plan.json"),
            "split_manifest_sha256": sha256_file(out / "split_manifest.json"),
            "selected_epochs": selected,
            "minimum_development_log_loss": minima,
            "primary_epoch": 1,
            "curves_sha256": sha256_file(out / "curves.csv"),
        },
    )
    final_directory = out / "final_fit"
    final_runs = fit_final(final_directory, expanded_final, temporal, selected, device, reporter)
    frozen = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_file(out / "plan.json"),
        "selection_sha256": sha256_file(out / "selection.json"),
        "split_manifest_sha256": sha256_file(out / "split_manifest.json"),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "scope": plan["roles"]["prospective"],
        "primary_seed": plan["primary_seed"],
        "sensitivity_seeds": plan["sensitivity_seeds"],
        "primary_epoch": 1,
        "secondary_development_selected_epoch": selected,
        "primary_metric": "round-weighted Log Loss, Transformer minus matched logistic regression",
        "secondary_metrics": ["Brier", "ROC AUC"],
        "bootstrap": plan["bootstrap"],
        "transformer": temporal["transformer"],
        "single_snapshot_window": 1,
        "relative_position_max_length": max(temporal["window_lengths"]),
        "inference_batch_size": temporal["training"]["batch_size"],
        "device": hardware,
        "artifacts": hashes(final_directory),
    }
    write_json(out / "prospective_freeze.json", frozen)
    emit(reporter, "fit", "completed", "Saved and verified all prospective model artifacts.")

    emit(reporter, "evaluate", "active", "Scoring on the fixed 10-match development cohort.")
    predictions = reference.select(
        *index.columns,
        "full_n32_logistic",
        *(probability_column("full_n32", seed) for seed in seeds),
    ).with_columns(
        pl.Series("expanded_logistic", logistic_predict(logistic, validation_values)),
        *(pl.Series(name, values) for name, values in probabilities.items()),
    )
    rows = score_development(predictions, seeds, plan["bootstrap"])
    write_parquet(predictions, out / "development_predictions.parquet")
    pl.from_dicts(rows, infer_schema_length=None).write_csv(out / "development_comparison.csv")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "mirage_train_expansion_historical_development",
        "reference_fit": counts(fit32),
        "supplement": counts(extra),
        "development_fit": counts(expanded_dev),
        "development": counts(development),
        "final_fit": counts(expanded_final),
        "missing_match_ids": intake["missing"],
        "device": hardware,
        "development_training": telemetry,
        "final_training": final_runs,
        "selected_epochs": selected,
        "development_scores": rows,
        "predictions_sha256": sha256_file(out / "development_predictions.parquet"),
        "prospective_freeze_sha256": sha256_file(out / "prospective_freeze.json"),
    }
    write_json(out / "report.json", report)
    (out / "report.md").write_text(report_text(report), encoding="utf-8")
    emit(reporter, "evaluate", "completed", "Saved the matched development scores and intervals.")
    emit(reporter, "run", "completed", "Mirage training expansion finished.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--intake", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file)
    try:
        run(args.out, args.intake, reporter)
    except (OSError, ValueError, ImportError, PipelineError, RuntimeError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the fixed epoch-1 contrasts with the same development matches in every draw."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.dataset import write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from scripts.fit_matched34_transformers import read_json
from scripts.fit_mirage_snapshot_epochs import verify_hashes
from scripts.fit_mirage_train_size import probability_column


def match_bootstrap_scores(
    predictions: pl.DataFrame, columns: list[str], config: dict[str, Any]
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Compute pooled round-weighted scores and common whole-match replicates."""
    probabilities = predictions.select(columns).to_numpy()
    if not np.all(np.isfinite(probabilities) & (probabilities >= 0) & (probabilities <= 1)):
        raise PipelineError("A saved probability is outside [0, 1] or nonfinite.")
    epsilon = np.finfo(np.float64).eps
    p = np.clip(probabilities, epsilon, 1 - epsilon)
    y = predictions["y_ct_win"].to_numpy()
    w = predictions["round_weight"].to_numpy()
    tick_loss = np.where(y[:, None] == 1, -np.log(p), -np.log1p(-p))
    match_ids, row_group = np.unique(predictions["match_id"].to_numpy(), return_inverse=True)
    weighted_loss = tick_loss * w[:, None]
    per_match = np.stack([weighted_loss[row_group == i].sum(axis=0) for i in range(len(match_ids))])
    mass = np.bincount(row_group, weights=w)
    choices = np.random.default_rng(config["seed"]).integers(
        0, len(match_ids), size=(config["draws"], len(match_ids))
    )
    replicated_scores = per_match[choices].sum(axis=1) / mass[choices].sum(axis=1)[:, None]
    points = dict(zip(columns, (weighted_loss.sum(axis=0) / w.sum()).tolist(), strict=True))
    replicated = dict(zip(columns, replicated_scores.T, strict=True))
    return points, replicated


def interval(bootstrap: np.ndarray) -> dict[str, float]:
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {"ci95_low": float(low), "ci95_high": float(high)}


def score(out: Path, reporter: ProgressReporter) -> None:
    outputs = ("epoch1_gaps.csv", "gap_changes.csv", "curve_summary.csv", "report.json")
    if any((out / name).exists() for name in outputs):
        raise PipelineError("This experiment has already been scored.")
    fit = read_json(out / "fit.json")
    verify_hashes(out, fit["artifacts"])
    plan = read_json(out / "plan.json")
    trials = read_json(out / "split_manifest.json")["trials"]
    seeds = [plan["primary_training_seed"], *plan["sensitivity_training_seeds"]]
    epochs = plan["learning_curve_epochs"]
    predictions = pl.read_parquet(out / "development_predictions.parquet")
    curves = pl.read_csv(out / "curves.csv")
    if curves.height != len(trials) * len(seeds) * epochs:
        raise PipelineError("Incomplete or duplicated curve rows.")
    columns = [
        name
        for trial in trials
        for name in (f"{trial}_logistic", *(probability_column(trial, seed) for seed in seeds))
    ]
    emit(reporter, "evaluate", "active", "Scoring every size on common development ticks.")
    points, replicated = match_bootstrap_scores(predictions, columns, plan["bootstrap"])
    rows, summaries, by_trial = [], [], {}
    for trial, info in trials.items():
        logistic = f"{trial}_logistic"
        for seed in seeds:
            candidate = probability_column(trial, seed)
            curve = curves.filter(pl.col("trial") == trial, pl.col("seed") == seed).sort("epoch")
            if curve["epoch"].to_list() != list(range(1, epochs + 1)):
                raise PipelineError(f"Unexpected epoch sequence for {trial}, seed {seed}.")
            start, end = curve.row(0, named=True), curve.row(-1, named=True)
            if not np.isclose(points[candidate], start["development_log_loss"], rtol=0, atol=1e-12):
                raise PipelineError(
                    f"Saved epoch-1 probabilities disagree with the curve: {candidate}"
                )
            delta = points[candidate] - points[logistic]
            gaps = replicated[candidate] - replicated[logistic]
            by_trial[trial, seed] = (delta, gaps)
            common = {
                "trial": trial,
                "subset_seed": info.get("subset_seed"),
                "training_seed": seed,
                "fit_matches": info["fit"]["matches"],
                "fit_rounds": info["fit"]["rounds"],
                "fit_ticks": info["fit"]["samples"],
                "optimizer_steps_epoch_1": start["optimizer_steps"],
                "optimizer_steps_epoch_20": end["optimizer_steps"],
            }
            rows.append(
                {
                    **common,
                    "logistic_log_loss": points[logistic],
                    "transformer_epoch_1_log_loss": points[candidate],
                    "epoch_1_gap": delta,
                    **interval(gaps),
                }
            )
            best = curve.row(curve["development_log_loss"].arg_min(), named=True)
            summaries.append(
                {
                    **common,
                    "minimum_development_epoch": best["epoch"],
                    "optimizer_steps_at_minimum": best["optimizer_steps"],
                    "development_log_loss_epoch_1": start["development_log_loss"],
                    "minimum_development_log_loss": best["development_log_loss"],
                    "development_log_loss_epoch_20": end["development_log_loss"],
                    "development_log_loss_rise_from_minimum": (
                        end["development_log_loss"] - best["development_log_loss"]
                    ),
                    "training_log_loss_epoch_1": start["train_log_loss"],
                    "training_log_loss_epoch_20": end["train_log_loss"],
                }
            )

    changes = []
    for draw in plan["subset_seeds"]:
        for seed in seeds:
            gap16, replicates16 = by_trial[f"draw_{draw}_n16", seed]
            gap32, replicates32 = by_trial["full_n32", seed]
            changes.append(
                {
                    "subset_seed": draw,
                    "training_seed": seed,
                    "gap_16": gap16,
                    "gap_32": gap32,
                    "gap_change_32_minus_16": gap32 - gap16,
                    **interval(replicates32 - replicates16),
                }
            )
    pl.DataFrame(rows).write_csv(out / "epoch1_gaps.csv")
    pl.DataFrame(changes).write_csv(out / "gap_changes.csv")
    pl.DataFrame(summaries).write_csv(out / "curve_summary.csv")
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "mirage_train_size_historical_development",
        "scope": plan["development_scope"],
        "primary_training_seed": plan["primary_training_seed"],
        "fit_sha256": sha256_file(out / "fit.json"),
        "plan_sha256": sha256_file(out / "plan.json"),
        "split_manifest_sha256": sha256_file(out / "split_manifest.json"),
        "bootstrap": plan["bootstrap"],
        "epoch_1_gaps": rows,
        "gap_changes": changes,
        "curve_summary": summaries,
        "metrics": {
            "validation": {
                "model": {
                    "log_loss": points[
                        probability_column("full_n32", plan["primary_training_seed"])
                    ]
                }
            }
        },
    }
    write_json(out / "report.json", result)
    emit(reporter, "evaluate", "completed", "Computed paired size contrasts and curve summaries.")
    emit(reporter, "run", "completed", "Saved the historical development training-size results.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file)
    try:
        score(args.out, reporter)
    except (PipelineError, OSError, ValueError, RuntimeError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

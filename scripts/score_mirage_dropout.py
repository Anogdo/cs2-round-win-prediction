"""Score both dropout strategies and freeze the 49-match model only on adoption."""

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
from cs2_pipeline.phase2_compare import counts
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from scripts.fit_matched34_transformers import read_json
from scripts.fit_mirage_dropout import (
    load_inputs,
    model_config,
    probability_column,
    select_shared_epoch,
)
from scripts.fit_mirage_snapshot_epochs import cuda_device, hashes, seed_order, verify_hashes
from scripts.fit_mirage_train_expansion import fit_final
from scripts.score_mirage_train_size import interval, match_bootstrap_scores


def prefixed_interval(name: str, draws: np.ndarray) -> dict[str, float]:
    return {f"{name}_{key}": value for key, value in interval(draws).items()}


def score_cohort(
    cohort: str, frame: pl.DataFrame, plan: dict[str, Any], shared: dict[str, int]
) -> list[dict[str, Any]]:
    seeds = seed_order(plan)
    keys = {
        (arm, role): [probability_column(arm, role, seed) for seed in seeds]
        for role in ("epoch_1", "selected")
        for arm in ("d010", "d020")
    }
    columns = ["logistic", *(name for columns in keys.values() for name in columns)]
    point, replicates = match_bootstrap_scores(frame, columns, plan["bootstrap"])
    summary = counts(frame)
    rows = []
    for role in ("epoch_1", "selected"):
        first, second = keys["d010", role], keys["d020", role]
        for position, seed in enumerate([*seeds, None]):
            positions = slice(None) if seed is None else slice(position, position + 1)
            base_names, candidate_names = first[positions], second[positions]
            base = float(np.mean([point[name] for name in base_names]))
            candidate = float(np.mean([point[name] for name in candidate_names]))
            base_draws = np.mean([replicates[name] for name in base_names], axis=0)
            cand_draws = np.mean([replicates[name] for name in candidate_names], axis=0)
            rows.append(
                {
                    "cohort": cohort,
                    "matches": summary["matches"],
                    "rounds": summary["rounds"],
                    "samples": summary["samples"],
                    "comparison": role,
                    "seed": seed,
                    "epoch_d010": 1 if role == "epoch_1" else shared["d010"],
                    "epoch_d020": 1 if role == "epoch_1" else shared["d020"],
                    "logistic_log_loss": point["logistic"],
                    "d010_log_loss": base,
                    "d020_log_loss": candidate,
                    "d020_minus_d010": candidate - base,
                    **prefixed_interval("d020_minus_d010", cand_draws - base_draws),
                    "d010_minus_logistic": base - point["logistic"],
                    **prefixed_interval("d010_minus_logistic", base_draws - replicates["logistic"]),
                    "d020_minus_logistic": candidate - point["logistic"],
                    **prefixed_interval("d020_minus_logistic", cand_draws - replicates["logistic"]),
                }
            )
    return rows


def freeze_candidate(
    out: Path, plan: dict[str, Any], chosen_epoch: int, reporter: ProgressReporter
) -> dict[str, Any]:
    temporal, _, _, final = load_inputs(plan)
    new_config = model_config(temporal, plan["dropout"]["d020"])
    directory = out / "final_fit"
    device, evidence = cuda_device(temporal)
    seeds = seed_order(plan)
    training = fit_final(
        directory, final, new_config, dict.fromkeys(seeds, chosen_epoch), device, reporter
    )
    source = ROOT / plan["source_run"]
    original = read_json(source / "prospective_freeze.json")
    previous_final = source / "final_fit"
    if not pl.read_parquet(directory / "fit_targets.parquet").equals(
        pl.read_parquet(previous_final / "fit_targets.parquet")
    ) or any(
        read_json(directory / name) != read_json(previous_final / name)
        for name in ("preprocessing.json", "logistic.json")
    ):
        raise PipelineError(
            "The 49-match refit differs from the incumbent in rows or preprocessing."
        )
    release = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_file(out / "plan.json"),
        "selection_sha256": sha256_file(out / "selection.json"),
        "split_manifest_sha256": sha256_file(out / "split_manifest.json"),
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "scope": original["scope"],
        "primary_seed": plan["primary_seed"],
        "sensitivity_seeds": plan["sensitivity_seeds"],
        "primary_epoch": chosen_epoch,
        "reference_epoch": 1,
        "primary_metric": original["primary_metric"],
        "secondary_metrics": original["secondary_metrics"],
        "bootstrap": original["bootstrap"],
        "transformer": new_config["transformer"],
        "feature_version": 1,
        "feature_count": 47,
        "single_snapshot_window": 1,
        "relative_position_max_length": original["relative_position_max_length"],
        "inference_batch_size": original["inference_batch_size"],
        "preprocessing": "final_fit/preprocessing.json",
        "logistic_model": "final_fit/logistic.json",
        "selected_checkpoints": {
            str(seed): f"final_fit/seed_{seed}_epoch_{chosen_epoch}.pt" for seed in seeds
        },
        "calibration": "none",
        "device": evidence,
        "fit": counts(final),
        "training": training,
        "baseline_freeze": f"{plan['source_run']}/prospective_freeze.json",
        "baseline_freeze_sha256": plan["source_hashes"]["prospective_freeze.json"],
        "artifacts": hashes(directory),
    }
    write_json(out / "prospective_freeze.json", release)
    emit(
        reporter, "fit", "completed", "Saved the adopted 49-match candidate and its scoring freeze."
    )
    return release


def run(out: Path, reporter: ProgressReporter) -> None:
    for name in ("development_scores.csv", "selection.json", "report.json", "final_fit"):
        if (out / name).exists():
            raise PipelineError(f"Use a fresh output path for scoring: {name} already exists.")
    plan = read_json(out / "plan.json")
    receipt = read_json(out / "development_fit.json")
    verify_hashes(out, receipt["artifacts"])
    curve = pl.read_csv(out / "curves.csv")
    shared = {
        arm: select_shared_epoch(
            curve.filter(pl.col("arm") == arm), seed_order(plan), plan["epochs"]
        )
        for arm in plan["dropout"]
    }
    if shared != receipt["shared_epochs"]:
        raise PipelineError("The stored epochs disagree with the three-seed development curves.")
    predictions = pl.read_parquet(out / "development_predictions.parquet")
    gate_scopes = plan["required_cohorts_for_adoption"]
    if set(gate_scopes) != {"pooled", *predictions["event"].unique().to_list()}:
        raise PipelineError("Adoption cohorts differ from the frozen development data.")
    emit(reporter, "evaluate", "active", "Scoring epoch 1 and the shared-epoch strategies.")
    rows = [
        row
        for cohort in gate_scopes
        for row in score_cohort(
            cohort,
            predictions if cohort == "pooled" else predictions.filter(pl.col("event") == cohort),
            plan,
            shared,
        )
    ]
    scores = pl.from_dicts(rows)
    for arm in plan["dropout"]:
        for row in scores.filter(
            pl.col("cohort") == "pooled", pl.col("seed").is_not_null()
        ).iter_rows(named=True):
            expected = curve.filter(
                pl.col("arm") == arm,
                pl.col("seed") == row["seed"],
                pl.col("epoch") == row[f"epoch_{arm}"],
            )["development_log_loss"].item()
            if not np.isclose(row[f"{arm}_log_loss"], expected, rtol=0, atol=1e-12):
                raise PipelineError("Frozen checkpoint probabilities do not match the curve.")
    scores.write_csv(out / "development_scores.csv")
    gate = [
        {
            "cohort": row["cohort"],
            "d010_mean_log_loss": row["d010_log_loss"],
            "d020_mean_log_loss": row["d020_log_loss"],
            "delta": row["d020_minus_d010"],
            "improved": row["d020_log_loss"] < row["d010_log_loss"],
        }
        for row in rows
        if row["comparison"] == "selected" and row["seed"] is None
    ]
    if len(gate) != len(gate_scopes):
        raise PipelineError("Adoption rule is missing a required cohort.")
    adopt = all(row["improved"] for row in gate)
    decision = {
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_file(out / "plan.json"),
        "development_fit_sha256": sha256_file(out / "development_fit.json"),
        "development_scores_sha256": sha256_file(out / "development_scores.csv"),
        "development_predictions_sha256": sha256_file(out / "development_predictions.parquet"),
        "shared_epochs": shared,
        "adoption_rule": plan["adoption"],
        "adopt_d020": adopt,
        "gate": gate,
    }
    write_json(out / "selection.json", decision)
    emit(reporter, "evaluate", "completed", f"Locked the 0.20 adoption decision: {adopt}.")
    if adopt:
        freeze_candidate(out, plan, shared["d020"], reporter)
        release = out / "prospective_freeze.json"
    else:
        release = ROOT / plan["source_run"] / "prospective_freeze.json"
    baseline_freeze = ROOT / plan["source_run"] / "prospective_freeze.json"
    prior_sha = plan["source_hashes"]["prospective_freeze.json"]
    if sha256_file(baseline_freeze) != prior_sha:
        raise PipelineError("The prior prospective freeze changed.")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "mirage_dropout_historical_development",
        "scope": plan["scope"],
        "plan_sha256": decision["plan_sha256"],
        "selection_sha256": sha256_file(out / "selection.json"),
        "development_fit": receipt["fit"],
        "development": receipt["development"],
        "planned_final_fit": receipt["planned_final_fit"],
        "bootstrap": plan["bootstrap"],
        "chosen_prospective_freeze": release.resolve().relative_to(ROOT).as_posix(),
        "chosen_prospective_freeze_sha256": sha256_file(release),
        "adopt_d020": adopt,
        "shared_epochs": shared,
        "gate": gate,
        "scores": rows,
    }
    write_json(out / "report.json", report)
    emit(reporter, "run", "completed", "Saved the dropout comparison and prospective decision.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file)
    try:
        run(args.out, reporter)
    except (PipelineError, OSError, ValueError, RuntimeError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Score the frozen Mirage epoch experiment on the previously observed 50 matches."""

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
from cs2_pipeline.matched34_score import difference
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import counts, sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import logistic_predict, network_arrays, transform
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import load_v1, verify_partition_order, window_indices
from scripts.fit_matched34_transformers import fit_inputs, read_json
from scripts.fit_mirage_snapshot_epochs import (
    CHECKPOINTS,
    check_scope,
    cuda_device,
    locked_selection,
    seed_order,
    target_index,
    verify_hashes,
    write_parquet,
)

REFERENCE = ROOT / "models/matched34_observed50_v1"


def score_predictions(
    predictions: pl.DataFrame, plan: dict[str, Any], selection: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scores = {"logistic": metrics(predictions, predictions["logistic"].to_list())}
    table = [{"model": "logistic", "seed": None, "epoch": None, **scores["logistic"]}]
    pairs = []
    for seed in seed_order(plan):
        seed_scores = {}
        for role in CHECKPOINTS:
            column = f"seed_{seed}_{role}"
            seed_scores[role] = metrics(predictions, predictions[column].to_list())
            epoch = {
                "epoch_1": 1,
                "selected": selection["seeds"][str(seed)]["selected_epoch"],
                "epoch_200": plan["epochs"],
            }[role]
            table.append({"model": role, "seed": seed, "epoch": epoch, **seed_scores[role]})
        scores[str(seed)] = seed_scores
        for base, candidate in plan["comparisons"]:
            reference = base if base == "logistic" else f"seed_{seed}_{base}"
            paired = difference(
                predictions, reference, f"seed_{seed}_{candidate}", plan["bootstrap"]
            )
            pairs.append({"seed": seed, "comparison": f"{candidate} minus {base}", **paired})
    return scores, table, pairs


def verify_observed_targets(predictions: pl.DataFrame) -> pl.DataFrame:
    archived = REFERENCE / "predictions.parquet"
    expected = read_json(REFERENCE / "report.json")["predictions_sha256"]
    if sha256_file(archived) != expected:
        raise PipelineError("The archived matched 34-to-50 predictions have changed.")
    prior = pl.read_parquet(archived).sort("sample_id")
    keys = (
        "sample_id",
        "event",
        "match_id",
        "map_id",
        "demo_sha256",
        "round_id",
        "tick",
        "y_ct_win",
        "round_weight",
    )
    if not predictions.select(*keys).equals(prior.select(*keys)):
        raise PipelineError(
            "Retrospective targets, labels or weights differ from the prior scoring."
        )
    return prior


def score(out: Path, reporter: ProgressReporter | None) -> dict[str, Any]:
    if any((out / name).exists() for name in ("report.json", "results.csv", "predictions.parquet")):
        raise PipelineError("Choose a fresh experiment directory to avoid overwriting scores.")
    emit(reporter, "evaluate", "active", "Verifying the development lock and final-fit hashes.")
    plan, selection = locked_selection(out)
    receipt = read_json(out / "final_fit.json")
    lock_hash = sha256_file(out / "selection.json")
    if receipt["selection_sha256"] != lock_hash or any(
        receipt["seeds"][str(seed)]["selected_epoch"]
        != selection["seeds"][str(seed)]["selected_epoch"]
        or receipt["seeds"][str(seed)]["epochs"] != plan["epochs"]
        for seed in seed_order(plan)
    ):
        raise PipelineError("The final-fit checkpoint epochs differ from the locked selection.")
    directory = out / "final_fit"
    verify_hashes(directory, receipt["artifacts"])
    temporal, fit_frame = fit_inputs()
    if not target_index(fit_frame).equals(pl.read_parquet(directory / "fit_targets.parquet")):
        raise PipelineError("The final fit used different targets than the exact archived split.")

    emit(
        reporter, "evaluate", "active", "Opening the observed 50 matches for retrospective scoring."
    )
    opened_at = datetime.now(UTC).isoformat()
    spec = plan["retrospective"]
    meta, observed = load_v1(ROOT, ROOT / spec["dataset"], temporal["exploratory_dataset_sha256"])
    check_scope(observed, spec)
    verify_partition_order({"final_fit": fit_frame, "retrospective": observed})
    development = pl.read_parquet(out / "development/predictions.parquet")
    verify_partition_order({"development": development, "retrospective": observed})
    if meta["split_policy"]["method"] != "holdout_only":
        raise PipelineError("The retrospective dataset must be holdout_only.")

    predictions = target_index(observed)
    prior = verify_observed_targets(predictions)
    device, evidence = cuda_device(temporal)
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer, predict

    values = transform(observed, read_json(directory / "preprocessing.json"))
    inputs = DeviceWindows.from_arrays(
        *network_arrays(observed, values, window_indices(observed, 1), 1), device
    )
    predictions = predictions.with_columns(
        pl.Series("logistic", logistic_predict(read_json(directory / "logistic.json"), values))
    )
    for seed in seed_order(plan):
        model = WindowTransformer(
            values.shape[1], max(temporal["window_lengths"]), temporal["transformer"]
        ).to(device)
        for role in CHECKPOINTS:
            model.load_state_dict(
                torch.load(
                    directory / f"seed_{seed}" / f"{role}.pt", map_location="cpu", weights_only=True
                )
            )
            p = sigmoid(predict(model, inputs, temporal["training"]["batch_size"]))
            if not np.isfinite(p).all():
                raise PipelineError(f"Invalid checkpoint scores: seed {seed}, {role}")
            predictions = predictions.with_columns(pl.Series(f"seed_{seed}_{role}", p))
        emit(reporter, "evaluate", "active", f"Scored all three checkpoints for seed {seed}.")

    scores, table, pairs = score_predictions(predictions, plan, selection)
    repeat_columns = {"logistic": "controlled_logistic"}
    repeat_columns.update(
        {f"seed_{seed}_epoch_1": f"seed_{seed}_window_1" for seed in seed_order(plan)}
    )
    repeat_delta = {
        new: float(np.max(np.abs(predictions[new].to_numpy() - prior[old].to_numpy())))
        for new, old in repeat_columns.items()
    }
    primary = scores[str(plan["primary_seed"])]["selected"]
    emit(
        reporter,
        "evaluate",
        "completed",
        "Finished aligned retrospective scoring.",
        split="test",
        metrics=primary,
    )
    emit(reporter, "save", "active", "Saving probabilities and paired match-bootstrap intervals.")
    prediction_path = out / "predictions.parquet"
    write_parquet(predictions, prediction_path)
    pair_lookup = {(pair["candidate"], pair["reference"]): pair for pair in pairs}
    result_rows = []
    for row in table:
        seed, role = row["seed"], row["model"]
        result = {key: row[key] for key in ("model", "seed", "epoch", *plan["metrics"])}
        for base in ("logistic", "epoch_1"):
            pair = pair_lookup.get(
                (f"seed_{seed}_{role}", base if base == "logistic" else f"seed_{seed}_{base}")
            )
            delta, low, high = (
                (pair["delta"]["log_loss"], *pair["match_bootstrap_95_ci"]["log_loss"])
                if pair
                else (None, None, None)
            )
            result.update(
                {
                    f"log_loss_delta_vs_{base}": delta,
                    f"log_loss_ci95_low_vs_{base}": low,
                    f"log_loss_ci95_high_vs_{base}": high,
                }
            )
        result_rows.append(result)
    pl.DataFrame(result_rows).write_csv(out / "results.csv")
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "mirage_snapshot_epoch_retrospective",
        "scope": spec["policy"],
        "selection_sha256": lock_hash,
        "final_fit_sha256": sha256_file(out / "final_fit.json"),
        "first_observed_data_access_utc": opened_at,
        "plan_sha256": selection["plan_sha256"],
        "uv_lock_sha256": sha256_file(ROOT / "uv.lock"),
        "code_sha256": {
            path: sha256_file(ROOT / path)
            for path in (
                "scripts/fit_matched34_transformers.py",
                "scripts/fit_mirage_snapshot_epochs.py",
                "scripts/score_mirage_snapshot_epochs.py",
                "code/src/cs2_pipeline/temporal_net.py",
            )
        },
        "observed_dataset_sha256": temporal["exploratory_dataset_sha256"],
        "previous_predictions_sha256": sha256_file(REFERENCE / "predictions.parquet"),
        "predictions_sha256": sha256_file(prediction_path),
        "primary_seed": plan["primary_seed"],
        "development_fit": selection["fit"],
        "development": selection["development"],
        "final_fit": receipt["fit"],
        "retrospective": counts(observed),
        "device": evidence,
        "results": scores,
        "paired": pairs,
        "bootstrap": plan["bootstrap"],
        "archived_epoch_1_max_abs_probability_delta": repeat_delta,
        "metrics": {"test": {"model": primary}},
    }
    write_json(out / "report.json", result)
    emit(reporter, "save", "completed", "Saved and re-read the aligned retrospective predictions.")
    emit(reporter, "run", "completed", "Mirage epoch experiment scored.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file)
    try:
        score(args.out, reporter)
    except (PipelineError, OSError, ValueError, ImportError, RuntimeError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

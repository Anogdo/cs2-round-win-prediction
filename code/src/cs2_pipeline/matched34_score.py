"""Score the same-fit-set traditional models and Transformers on the observed cohort."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import PLAN as COMPARISON_PLAN
from cs2_pipeline.phase2_compare import counts, sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.phase2_temporal import PLAN as TEMPORAL_PLAN
from cs2_pipeline.phase2_temporal import driver_version, logistic_predict, network_arrays, transform
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import ROUND_KEY, load_v1, verify_partition_order, window_indices
from cs2_pipeline.train import round_weights

PLAN = ROOT / "configs/matched34_comparison.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reference_probabilities(path: Path, digest: str, name: str, ids: pl.Series) -> pl.Series:
    if sha256_file(path) != digest:
        raise PipelineError(f"Frozen reference predictions changed: {path}")
    table = pl.read_parquet(path, columns=["sample_id", name]).sort("sample_id")
    if not table["sample_id"].equals(ids):
        raise PipelineError(f"Frozen reference has different target IDs: {name}")
    return table[name]


def difference(frame: pl.DataFrame, first: str, second: str, bootstrap: dict) -> dict:
    base, candidate = frame[first].to_list(), frame[second].to_list()
    before, after = metrics(frame, base), metrics(frame, candidate)
    paired = cluster_bootstrap(
        frame, base, candidate, draws=bootstrap["draws"], seed=bootstrap["seed"]
    )
    return {
        "candidate": second,
        "reference": first,
        "delta": {key: after[key] - before[key] for key in ("log_loss", "brier_score")},
        "match_bootstrap_95_ci": paired["percentile_95_intervals"]["calibrated_minus_base"],
    }


def run(
    dataset: Path,
    out: Path,
    reporter: ProgressReporter | None = None,
    *,
    fit_directory: Path | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"Choose a fresh output directory: {out}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        from cs2_pipeline.temporal_net import (
            DeviceWindows,
            WindowTransformer,
            cuda_environment,
            predict,
        )
    except (ImportError, OSError) as exc:
        raise PipelineError(
            "Install the optional CUDA dependencies with uv sync --extra temporal."
        ) from exc

    plan = read_json(PLAN)
    if sha256_file(COMPARISON_PLAN) != plan["comparison_plan_sha256"]:
        raise PipelineError("The Phase 2 comparison plan changed.")
    original = read_json(COMPARISON_PLAN)
    temporal = read_json(TEMPORAL_PLAN)
    if dataset.resolve() != (ROOT / original["exploratory_dataset"]).resolve():
        raise PipelineError("Use the frozen observed 50-match dataset for this comparison.")
    emit(reporter, "load", "active", "Checking the 34-map fit receipt and comparison target IDs.")
    frozen = fit_directory or ROOT / plan["fit_receipt_directory"]
    receipt = read_json(frozen / "fit.json")
    fit_hash = sha256_file(frozen / "fit.json")
    seeds = [plan["primary_seed"], *plan["sensitivity_seeds"]]
    lengths = plan["windows"]
    names = [f"seed_{seed}_window_{length}" for seed in seeds for length in lengths]
    required = {
        "preprocessing.json",
        "fit_sample_ids.parquet",
        "controlled_logistic.json",
        *(f"{name}.pt" for name in names),
    }
    artifacts = receipt["artifacts"]
    if (
        not required.issubset(artifacts)
        or any(sha256_file(frozen / name) != artifacts[name] for name in required)
        or receipt["fit_dataset_sha256"] != temporal["dataset_sha256"]
        or receipt["temporal_plan_sha256"] != sha256_file(TEMPORAL_PLAN)
        or receipt["seeds"] != seeds
        or receipt["lengths"] != lengths
        or receipt["epochs"] != plan["epochs"]
    ):
        raise PipelineError("The frozen matched-fit artifacts do not satisfy the scoring plan.")
    training_path = ROOT / original["original_training_dataset"]
    _, training = load_v1(ROOT, training_path, temporal["dataset_sha256"])
    fit = training.filter(pl.col("event").is_in(original["full_baseline_fit_events"]))
    fit_ids = pl.read_parquet(frozen / "fit_sample_ids.parquet", columns=["sample_id"])
    train_split = pl.read_parquet(training_path / "splits.parquet").filter(
        pl.col("split") == "train"
    )
    if (
        not all(
            item["sample_id"].sort().equals(fit["sample_id"]) for item in (fit_ids, train_split)
        )
        or counts(fit) != receipt["fit"]
    ):
        raise PipelineError("The Transformer fit IDs do not equal the original V1 training split.")
    base = ROOT / original["frozen_v1"]
    if sha256_file(base) != original["frozen_v1_sha256"]:
        raise PipelineError("The frozen traditional V1 model changed.")
    baseline = read_json(base)
    if baseline["feature_columns"] != feature_columns(1) or any(
        baseline["training"][f"{metric}_count"] != receipt["fit"][plural]
        for metric, plural in (("match", "matches"), ("round", "rounds"), ("sample", "samples"))
    ):
        raise PipelineError("Frozen V1 metadata does not match the fit scope.")
    metadata, observed = load_v1(ROOT, dataset, temporal["exploratory_dataset_sha256"])
    verify_partition_order({"fit": fit, "observed": observed})
    if metadata["split_policy"]["method"] != "holdout_only":
        raise PipelineError("The observed evaluation must remain holdout_only.")
    emit(reporter, "load", "completed", "Exact training rows and out-of-fit evaluation verified.")
    emit(reporter, "fit", "completed", "Models already fitted and saved; no fitting on evaluation.")
    emit(reporter, "evaluate", "active", "Running CUDA inference on the common target rows.")
    device, hardware = cuda_environment(temporal["training"]["device"])
    hardware["nvidia_driver"] = driver_version()
    values = transform(observed, read_json(frozen / "preprocessing.json"))
    rows = window_indices(observed, max(lengths))
    columns = ["controlled_logistic", *plan["traditional_references"], *names]
    predictions = observed.select(
        "sample_id", "event", *ROUND_KEY, "tick", "y_ct_win"
    ).with_columns(
        pl.Series("round_weight", round_weights(observed)),
        pl.Series("available_snapshots", (rows >= 0).sum(axis=1)),
        pl.Series(
            "controlled_logistic",
            logistic_predict(read_json(frozen / "controlled_logistic.json"), values),
        ),
        *(
            reference_probabilities(
                ROOT / spec["predictions"], spec["sha256"], name, observed["sample_id"]
            )
            for name, spec in plan["traditional_references"].items()
        ),
    )
    inputs = {
        length: DeviceWindows.from_arrays(*network_arrays(observed, values, rows, length), device)
        for length in lengths
    }
    for seed in seeds:
        for length in lengths:
            name = f"seed_{seed}_window_{length}"
            model = WindowTransformer(values.shape[1], rows.shape[1], temporal["transformer"]).to(
                device
            )
            state = torch.load(frozen / f"{name}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            probability = sigmoid(
                predict(model, inputs[length], temporal["training"]["batch_size"])
            )
            if not np.isfinite(probability).all():
                raise PipelineError(f"Nonfinite inference probabilities: {name}")
            predictions = predictions.with_columns(pl.Series(name, probability))
            emit(reporter, "evaluate", "active", f"Scored {name}.", model=name)
    groups = {
        "pooled": predictions,
        **{
            event: predictions.filter(pl.col("event") == event)
            for event in sorted(set(predictions["event"]))
        },
    }
    scores = {
        name: {column: metrics(part, part[column].to_list()) for column in columns}
        for name, part in groups.items()
    }
    boot = plan["bootstrap"]
    pairs = [(f"seed_{seed}_window_1", f"seed_{seed}_window_8") for seed in seeds] + [
        (base_name, f"seed_{plan['primary_seed']}_window_{length}")
        for base_name in ("controlled_logistic", "frozen_v1", "gpu_full34")
        for length in lengths
    ]
    emit(
        reporter,
        "evaluate",
        "completed",
        "Retrospective comparison scored on the same rows.",
        split="validation",
        metrics=scores["pooled"]["controlled_logistic"],
    )
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "matched34_retrospective",
        "scope": plan["purpose"],
        "plan_sha256": sha256_file(PLAN),
        "code_sha256": sha256_file(Path(__file__)),
        "fit_receipt_sha256": fit_hash,
        "dataset_sha256": temporal["exploratory_dataset_sha256"],
        "fit": counts(fit),
        "evaluation": counts(observed),
        "by_event_counts": {
            event: counts(observed.filter(pl.col("event") == event))
            for event in groups
            if event != "pooled"
        },
        "device": hardware,
        "results": scores,
        "paired": [difference(predictions, first, second, boot) for first, second in pairs],
        "metrics": {
            "validation": {"model": scores["pooled"]["controlled_logistic"]},
            "test": {"model": scores["pooled"]["controlled_logistic"]},
        },
    }
    emit(reporter, "save", "active", "Writing aligned scored rows and their metrics.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".matched34-score-", dir=out.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        path = stage / "predictions.parquet"
        predictions.write_parquet(path)
        if not predictions.equals(pl.read_parquet(path)):
            raise PipelineError("The prediction table did not round-trip.")
        result["predictions_sha256"] = sha256_file(path)
        write_json(stage / "report.json", result)
        if out.exists():
            raise PipelineError(f"Output was created while scoring: {out}")
        stage.rename(out)
    emit(reporter, "save", "completed", "Saved matched comparison.", output=str(out))
    emit(reporter, "run", "completed", "Matched 34-map comparison completed.")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fit-dir", type=Path, help="A separately completed fit directory.")
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    reporter = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        report = run(args.dataset, args.out, reporter, fit_directory=args.fit_dir)
    except (
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        PipelineError,
        pl.exceptions.PolarsError,
    ) as exc:
        message = f"Matched comparison failed: {exc}"
        if reporter:
            reporter.emit("run", "failed", message)
        else:
            print(message, file=sys.stderr)
        return 1
    print({name: score["log_loss"] for name, score in report["results"]["pooled"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

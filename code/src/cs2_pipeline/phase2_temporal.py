"""Compare logistic, current-snapshot, and trailing-history models on frozen Mirage splits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from cs2_pipeline.calibrate import cluster_bootstrap, metrics
from cs2_pipeline.dataset import write_json
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_compare import PLAN as COMPARISON_PLAN
from cs2_pipeline.phase2_compare import counts, sigmoid
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import ROUND_KEY, load_v1, verify_partition_order, window_indices
from cs2_pipeline.train import round_weights

if TYPE_CHECKING:
    from cs2_pipeline.temporal_net import DeviceWindows

PLAN = ROOT / "configs/phase2_temporal_plan.json"
MODELS = ("logistic", "snapshot_transformer", "history8_transformer")


def fit_preprocessing(fit: pl.DataFrame) -> dict[str, Any]:
    columns = feature_columns(1)
    raw = fit.select(columns).to_numpy()
    if np.isinf(raw).any():
        raise PipelineError("Infinite values are not legal V1 features.")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    filled = imputer.fit_transform(raw)
    weights = round_weights(fit)
    mean = np.average(filled, axis=0, weights=weights)
    scale = np.sqrt(np.average((filled - mean) ** 2, axis=0, weights=weights))
    scale[(np.ptp(filled, axis=0) == 0) | (scale == 0)] = 1.0
    return {
        "columns": columns,
        "medians": imputer.statistics_.tolist(),
        "round_weighted_mean": mean.tolist(),
        "round_weighted_scale": scale.tolist(),
        "missing_indicators": False,
    }


def transform(frame: pl.DataFrame, preprocessing: dict[str, Any]) -> np.ndarray:
    values = frame.select(preprocessing["columns"]).to_numpy()
    filled = np.where(np.isnan(values), preprocessing["medians"], values)
    normalized = (filled - preprocessing["round_weighted_mean"]) / preprocessing[
        "round_weighted_scale"
    ]
    if not np.isfinite(normalized).all():
        raise PipelineError("Nonfinite feature after fit-only preprocessing.")
    return normalized


def fit_logistic(frame: pl.DataFrame, values: np.ndarray, plan: dict[str, Any]) -> dict[str, Any]:
    model = LogisticRegression(
        C=plan["logistic_c"], max_iter=2000, solver="lbfgs", random_state=plan["training"]["seed"]
    )
    model.fit(values, frame["y_ct_win"].to_numpy(), sample_weight=round_weights(frame))
    saved = {
        "coefficients": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
        "C": plan["logistic_c"],
    }
    if saved["iterations"] >= model.max_iter or not np.allclose(
        logistic_predict(saved, values), model.predict_proba(values)[:, 1], atol=1e-12, rtol=1e-12
    ):
        raise PipelineError("Logistic regression failed to converge or export.")
    return saved


def logistic_predict(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    return sigmoid(values @ model["coefficients"] + model["intercept"])


def fit_network(
    name: str,
    fit: DeviceWindows,
    validation: DeviceWindows,
    validation_frame: pl.DataFrame,
    plan: dict[str, Any],
    checkpoint: Path,
    reporter: ProgressReporter | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    from cs2_pipeline.temporal_net import WindowTransformer, predict, train_epoch

    training = plan["training"]
    seed, batch_size = training["seed"], training["batch_size"]
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    model = WindowTransformer(len(feature_columns(1)), 8, plan["transformer"]).to(fit.values.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
        foreach=False,
        fused=False,
    )
    curve = []
    best_epoch = 0
    best_loss = float("inf")
    best_probabilities = None
    for epoch in range(1, training["max_epochs"] + 1):
        telemetry = train_epoch(model, optimizer, fit, batch_size, shuffle, verify_step=epoch == 1)
        probabilities = sigmoid(predict(model, validation, batch_size))
        validation_loss = metrics(validation_frame, probabilities.tolist())["log_loss"]
        curve.append({"epoch": epoch, "validation_log_loss": validation_loss, **telemetry})
        if validation_loss < best_loss:
            best_epoch, best_loss, best_probabilities = epoch, validation_loss, probabilities
            torch.save(model.state_dict(), checkpoint)
        emit(
            reporter,
            "fit",
            "active",
            f"{name} · epoch {epoch}/{training['max_epochs']} · "
            f"Porto Log Loss {validation_loss:.4f}",
            iterations=epoch,
            model=name,
            best_epoch=best_epoch,
            log_loss=validation_loss,
        )
    if best_probabilities is None:
        raise PipelineError(f"No valid validation checkpoint for {name}.")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    reloaded = sigmoid(predict(model, validation, batch_size))
    delta = float(np.max(np.abs(best_probabilities - reloaded)))
    if delta > 1e-7:
        raise PipelineError(f"Validation predictions changed after reloading {name}: {delta}.")
    return reloaded, {
        "window_length": fit.rows.shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch,
        "selected_validation_log_loss": best_loss,
        "checkpoint_roundtrip_max_abs_probability_delta": delta,
        "epochs": curve,
    }


def network_arrays(
    frame: pl.DataFrame, values: np.ndarray, rows: np.ndarray, length: int
) -> tuple[np.ndarray, ...]:
    return values, rows[:, -length:], frame["y_ct_win"].to_numpy(), np.asarray(round_weights(frame))


def prediction_table(
    frame: pl.DataFrame, rows: np.ndarray, predictions: dict[str, np.ndarray]
) -> pl.DataFrame:
    return frame.select(
        "sample_id", "event", *ROUND_KEY, "tick", "elapsed_seconds", "y_ct_win"
    ).with_columns(
        pl.Series("round_weight", round_weights(frame)),
        pl.Series("available_snapshots", (rows >= 0).sum(axis=1)),
        *(pl.Series(name, predictions[name]) for name in MODELS),
    )


def score(frame: pl.DataFrame, predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    if frame["y_ct_win"].n_unique() != 2:
        raise PipelineError("Full comparison partitions must contain both outcomes.")
    return {name: metrics(frame, predictions[name].tolist()) for name in MODELS}


def history_comparison(
    frame: pl.DataFrame, rows: np.ndarray, predictions: dict[str, np.ndarray], bootstrap: dict
) -> dict[str, Any]:
    paired = {}
    for reference in ("snapshot_transformer", "logistic"):
        estimate = cluster_bootstrap(
            frame,
            predictions[reference].tolist(),
            predictions["history8_transformer"].tolist(),
            draws=bootstrap["draws"],
            seed=bootstrap["seed"],
        )
        paired[f"history_minus_{reference}"] = {
            "matches": estimate["match_count"],
            "draws": estimate["draws"],
            "seed": estimate["seed"],
            "percentile_95_ci": estimate["percentile_95_intervals"]["calibrated_minus_base"],
        }
    available = (rows >= 0).sum(axis=1)
    strata = {}
    for label, mask in (
        ("one_snapshot", available == 1),
        ("two_to_seven_snapshots", (available > 1) & (available < 8)),
        ("eight_snapshots", available == 8),
    ):
        subset = frame.filter(pl.Series(mask))
        if not subset.is_empty():
            strata[label] = {
                **counts(subset),
                "scores": {
                    name: metrics(subset, probabilities[mask].tolist())
                    for name, probabilities in predictions.items()
                },
            }
    return {"paired_match_bootstrap": paired, "by_available_snapshots": strata}


def driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        versions = sorted(set(result.stdout.split()))
        return ", ".join(versions) or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(
    dataset: Path, out: Path, reporter: ProgressReporter | None = None, *, smoke_test: bool = False
) -> dict[str, Any]:
    if out.exists():
        raise PipelineError(f"Use a fresh output directory; already exists: {out}")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        import torch

        from cs2_pipeline.temporal_net import (
            DeviceWindows,
            WindowTransformer,
            cuda_environment,
            predict,
            train_epoch,
        )
    except (ImportError, OSError) as exc:
        raise PipelineError(
            "Install the locked CUDA dependencies with uv sync --extra temporal."
        ) from exc

    plan = json.loads(PLAN.read_text("utf-8"))
    if sha256_file(COMPARISON_PLAN) != plan["comparison_plan_sha256"]:
        raise PipelineError("The frozen Phase 2 comparison plan has changed.")
    comparison = json.loads(COMPARISON_PLAN.read_text("utf-8"))
    if dataset.resolve() != (ROOT / comparison["original_training_dataset"]).resolve():
        raise PipelineError("Use the frozen mirage_expanded_2026_v1 feature dataset.")
    if (
        plan["schema_version"] != 1
        or plan["feature_version"] != 1
        or plan["window_lengths"] != [1, 8]
    ):
        raise PipelineError("Unsupported temporal experiment schema, feature version or windows.")
    training = plan["training"]
    if training["dtype"] != "float32" or training["max_epochs"] < 1:
        raise PipelineError("The temporal plan requires FP32 and at least one epoch.")
    emit(
        reporter, "load", "active", "Verifying frozen V1 features and recording-local round ticks."
    )
    device, hardware = cuda_environment(training["device"])
    hardware["nvidia_driver"] = driver_version()
    hardware["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    _, original = load_v1(ROOT, dataset, plan["dataset_sha256"])
    parts = {}
    for name in ("fit", "calibration", "validation"):
        events = set(comparison[f"candidate_{name}_events"])
        part = original.filter(pl.col("event").is_in(events))
        if set(part["event"].unique()) != events:
            raise PipelineError(f"Missing frozen events in {name}.")
        parts[name] = part
    verify_partition_order(parts)
    fit, validation = parts["fit"], parts["validation"]
    prep = fit_preprocessing(fit)
    fit_values, fit_rows = transform(fit, prep), window_indices(fit, 8)
    batch_size = training["batch_size"]
    fit8 = DeviceWindows.from_arrays(*network_arrays(fit, fit_values, fit_rows, 8), device)
    emit(reporter, "load", "completed", "Fit and validation boundaries verified.", **counts(fit))
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_mode": "smoke_test" if smoke_test else "temporal_observed_exploration",
        "device": hardware,
        "plan_sha256": sha256_file(PLAN),
        "comparison_plan_sha256": plan["comparison_plan_sha256"],
        "dataset_sha256": plan["dataset_sha256"],
        "lock_sha256": sha256_file(ROOT / "uv.lock"),
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "phase2_temporal.py",
                "temporal_data.py",
                "temporal_net.py",
                "dataset.py",
                "phase2_compare.py",
                "train.py",
                "calibrate.py",
            )
        },
        "versions": {
            "python": sys.version.split()[0],
            **{package: version(package) for package in ("numpy", "polars", "scikit-learn")},
        },
        "partitions": {name: counts(part) for name, part in parts.items()},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".temporal-", dir=out.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        shutil.copyfile(PLAN, stage / "plan.json")
        write_json(stage / "preprocessing.json", prep)
        if smoke_test:
            emit(reporter, "fit", "active", "Running one FP32 CUDA epoch on the real fit set.")
            torch.manual_seed(training["seed"])
            model = WindowTransformer(len(feature_columns(1)), 8, plan["transformer"]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training["learning_rate"],
                weight_decay=training["weight_decay"],
                foreach=False,
                fused=False,
            )
            report["smoke_epoch"] = train_epoch(
                model,
                optimizer,
                fit8,
                batch_size,
                torch.Generator().manual_seed(training["seed"]),
                verify_step=True,
            )
            emit(reporter, "fit", "completed", "CUDA forward, backward and the fit epoch passed.")
        else:
            emit(reporter, "fit", "active", "Fitting all three candidates on Rio and Atlanta.")
            linear = fit_logistic(fit, fit_values, plan)
            write_json(stage / "logistic.json", linear)
            val_values, val_rows = transform(validation, prep), window_indices(validation, 8)
            held = {"logistic": logistic_predict(linear, val_values)}
            network_reports = {}
            for name, window in zip(MODELS[1:], plan["window_lengths"], strict=True):
                fit_input = (
                    fit8
                    if window == 8
                    else DeviceWindows.from_arrays(
                        *network_arrays(fit, fit_values, fit_rows, window), device
                    )
                )
                val_input = DeviceWindows.from_arrays(
                    *network_arrays(validation, val_values, val_rows, window), device
                )
                held[name], network_reports[name] = fit_network(
                    name,
                    fit_input,
                    val_input,
                    validation,
                    plan,
                    stage / f"{name}.pt",
                    reporter,
                )
            validation_scores = score(validation, held)
            winner = min(MODELS, key=lambda name: validation_scores[name]["log_loss"])
            selection = {
                "selected_by_porto_log_loss": winner,
                "checkpoint_epochs": {
                    name: r["selected_epoch"] for name, r in network_reports.items()
                },
                "validation_log_loss": {
                    name: r["log_loss"] for name, r in validation_scores.items()
                },
                "pre_exploration_sha256": {
                    path.name: sha256_file(path) for path in sorted(stage.iterdir())
                },
            }
            write_json(stage / "selection.json", selection)
            emit(
                reporter,
                "fit",
                "completed",
                f"Porto selection frozen: {winner}.",
                iterations=training["max_epochs"] * 2,
                model=winner,
            )
            emit(
                reporter,
                "evaluate",
                "active",
                "Scoring checkpoints on the previously observed 50 matches.",
                split="validation",
                metrics=validation_scores[winner],
            )
            observed_meta, observed = load_v1(
                ROOT, ROOT / comparison["exploratory_dataset"], plan["exploratory_dataset_sha256"]
            )
            if observed_meta["split_policy"]["method"] != "holdout_only":
                raise PipelineError("The observed 50 matches must remain holdout_only.")
            verify_partition_order({**parts, "observed_exploration": observed})
            observed_values, observed_rows = transform(observed, prep), window_indices(observed, 8)
            retrospective = {"logistic": logistic_predict(linear, observed_values)}
            for name, window in zip(MODELS[1:], plan["window_lengths"], strict=True):
                model = WindowTransformer(len(feature_columns(1)), 8, plan["transformer"]).to(
                    device
                )
                model.load_state_dict(
                    torch.load(stage / f"{name}.pt", map_location="cpu", weights_only=True)
                )
                observed_input = DeviceWindows.from_arrays(
                    *network_arrays(observed, observed_values, observed_rows, window), device
                )
                retrospective[name] = sigmoid(predict(model, observed_input, batch_size))
            observed_scores = score(observed, retrospective)
            report.update(
                selection=selection,
                transformer_training=network_reports,
                exploratory_dataset_sha256=plan["exploratory_dataset_sha256"],
                results={"validation": validation_scores, "observed_exploration": observed_scores},
                metrics={
                    "validation": {"model": validation_scores[winner]},
                    "test": {"model": observed_scores[winner]},
                },
                history_analysis=history_comparison(
                    observed, observed_rows, retrospective, comparison["bootstrap"]
                ),
            )
            report["partitions"]["observed_exploration"] = counts(observed)
            report["evaluation_scope"] = (
                "Porto chose epochs and the primary model. Comparison on the 50 previously "
                "observed maps is retrospective. No calibration or new blind test."
            )
            report["sample_alignment"] = {}
            for name, frame, rows, probabilities in (
                ("validation", validation, val_rows, held),
                ("observed_exploration", observed, observed_rows, retrospective),
            ):
                predictions = prediction_table(frame, rows, probabilities)
                path = stage / f"{name}_predictions.parquet"
                predictions.write_parquet(path)
                if not predictions.equals(pl.read_parquet(path)):
                    raise PipelineError(f"Prediction Parquet roundtrip changed rows for {name}.")
                report["sample_alignment"][name] = {
                    "all_three_models_on_same_rows": True,
                    "sample_count": frame.height,
                    "prediction_rows": predictions.height,
                    "round_weight_sum": float(predictions["round_weight"].sum()),
                }
            emit(
                reporter,
                "evaluate",
                "completed",
                "Aligned model probabilities and exploratory scores saved.",
            )
        emit(reporter, "save", "active", "Writing reproducibility artifacts to a fresh directory.")
        report["artifact_sha256"] = {
            path.name: sha256_file(path) for path in sorted(stage.iterdir())
        }
        write_json(stage / "report.json", report)
        if out.exists():
            raise PipelineError(f"Output directory was created during the run: {out}")
        stage.rename(out)
    emit(reporter, "save", "completed", "Run artifacts verified and saved.", output=str(out))
    emit(reporter, "run", "completed", "Temporal experiment completed.")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--smoke-test", action="store_true", help="Train one real CUDA epoch only.")
    args = parser.parse_args(argv)
    reporter = ProgressReporter(args.progress_file) if args.progress_file else None
    try:
        report = run(args.dataset, args.out, reporter, smoke_test=args.smoke_test)
    except (
        PipelineError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
        pl.exceptions.PolarsError,
    ) as exc:
        message = f"Temporal experiment failed: {exc}"
        if reporter:
            reporter.emit("run", "failed", message)
        else:
            print(message, file=sys.stderr)
        return 1
    if args.smoke_test:
        example = report["smoke_epoch"]
        print(
            f"CUDA smoke epoch: {example['seconds']:.2f}s, "
            f"{example['peak_allocated_mib']:.1f} MiB peak."
        )
    else:
        for partition, results in report["results"].items():
            print(
                f"{partition}: "
                + ", ".join(f"{name}={scores['log_loss']:.4f}" for name, scores in results.items())
            )
    print(f"Report: {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

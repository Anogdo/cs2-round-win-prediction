"""Import score-verified Mirage archives for additional historical validation only."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import (
    EXCLUDED_EVENT,
    HISTORICAL_VALIDATION_ONLY,
    TABLE_NAMES,
    build_map_tables,
    held_out_event,
    read_dataset,
    source_identity,
    write_json,
)
from cs2_pipeline.features import feature_columns
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import join_samples, recording_rounds, validate_timeline
from scripts.audit_phase2_data import audit_identities
from scripts.context_audit_batch import update
from scripts.import_holdout_50 import UNRAR, import_archive, published_score

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "configs/phase2_validation_more_20261118.json"
RAW = ROOT / "data/raw/phase2_validation_more"
EXTRACTED = RAW / "extracted_v1"
PROCESSED = ROOT / "data/processed/phase2_validation_more_v1"
DATASET = ROOT / "data/datasets/mirage_validation_more_v1"
PROGRESS = ROOT / "runs/phase2_validation_more_v1/progress.json"
PRIOR = ("mirage_expanded_2026_v1", "mirage_holdout_50_v1", "esl24_holdout_v1")


def load_plan() -> dict[str, Any]:
    plan = json.loads(PLAN.read_text("utf-8"))
    if plan["schema_version"] != 1 or plan["map_name"] != "de_mirage" or version("awpy") != "3.3.0":
        raise PipelineError("Import requires the pinned Mirage plan and Awpy 3.3.0.")
    archives = plan["archives"]
    for field in ("match_id", "archive", "archive_sha256"):
        if not archives or len({row[field] for row in archives}) != len(archives):
            raise PipelineError(f"Historical archive plan repeats {field}.")
    for row in archives:
        if (
            held_out_event(row["event"])
            or re.fullmatch(r"hltv:\d+", row["match_id"]) is None
            or not row["url"].startswith("https://www.hltv.org/matches/")
            or f"/{row['match_id'].split(':')[1]}/" not in row["url"]
            or held_out_event(row["url"])
            or not archive_basename(row["archive"])
            or re.fullmatch(r"[a-f0-9]{64}", row["archive_sha256"]) is None
            or date.fromisoformat(row["match_date"]).isoformat() != row["match_date"]
            or len(row["expected_score"]) != 2
            or any(type(score) is not int or score < 0 for score in row["expected_score"].values())
        ):
            raise PipelineError(f"Invalid historical match plan entry: {row['match_id']}")
    return plan


def archive_basename(name: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_() -]*\.rar", name, re.I) is not None


def prior_index() -> tuple[set[str], set[str], set[str]]:
    datasets = ROOT / "data/datasets"
    matches: set[str] = set()
    recordings: set[str] = set()
    events: set[str] = set()
    for name in PRIOR:
        _, tables = read_dataset(datasets / name)
        matches.update(tables["samples"]["match_id"].unique())
        recordings.update(tables["samples"]["demo_sha256"].unique())
        events.update(tables["samples"]["event"].unique())
    return matches, recordings, events


def worker(item: dict[str, Any], processed: Path, extracted: Path) -> None:
    local = {**item, "expected_rounds": sum(item["expected_score"].values())}
    import_archive(
        RAW / item["archive"], item, local, extracted_root=extracted, processed_root=processed
    )


def publish_dataset(
    plan: dict[str, Any],
    completed: list[dict[str, Any]],
    prior_hashes: set[str],
    out: Path,
    duplicates: list[dict[str, str]],
) -> dict[str, Any]:
    collected: dict[str, list[pl.DataFrame]] = {name: [] for name in TABLE_NAMES[:-1]}
    sources = []
    seen = set(prior_hashes)
    for item, receipt in zip(plan["archives"], completed, strict=True):
        manifest = ROOT / receipt["source_manifest"]
        entries = json.loads(manifest.read_text("utf-8"))["sources"]
        paths = [(manifest.parent / entry["path"]).resolve() for entry in entries]
        score = published_score(paths, item["expected_score"])
        if (
            receipt["match_id"] != item["match_id"]
            or receipt["event"] != item["event"]
            or receipt["match_date"] != item["match_date"]
            or receipt["archive_sha256"] != item["archive_sha256"]
            or receipt["published_score_verified"] != score
            or receipt["part_count"] != len(entries)
        ):
            raise PipelineError(f"Imported match receipt differs from plan: {item['match_id']}")
        for entry, path in zip(entries, paths, strict=True):
            metadata = json.loads((path / "metadata.json").read_text("utf-8"))
            checksum = metadata["demo_sha256"]
            if (
                checksum in seen
                or checksum not in receipt["demo_sha256"]
                or metadata["map_name"] != "de_mirage"
                or metadata["parser_version"] != "3.3.0"
            ):
                raise PipelineError(f"Incorrect or previously used recording: {path}")
            seen.add(checksum)
            identity = source_identity(entry, metadata)
            if (
                identity["match_id"] != item["match_id"]
                or identity["event"] != item["event"]
                or identity["match_date"] != item["match_date"]
                or identity["map_id"] != f"{item['match_id']}:m{item['map_index']}"
            ):
                raise PipelineError(f"Incorrect map identity: {path}")
            tables, counts = build_map_tables(
                metadata,
                pl.read_parquet(path / "rounds.parquet"),
                pl.read_parquet(path / "snapshots.parquet"),
                identity,
            )
            for name, frame in tables.items():
                collected[name].append(frame)
            sources.append(
                {
                    **identity,
                    **counts,
                    "path": path.relative_to(ROOT).as_posix(),
                    "demo_sha256": checksum,
                    "archive_sha256": item["archive_sha256"],
                    "published_score_verified": True,
                    "source_file_sha256": {
                        name: sha256_file(path / name)
                        for name in (
                            "metadata.json",
                            "rounds.parquet",
                            "players.parquet",
                            "snapshots.parquet",
                        )
                    },
                }
            )
    tables = {name: pl.concat(frames).sort("sample_id") for name, frames in collected.items()}
    samples = tables["samples"]
    tables["splits"] = samples.select("sample_id", pl.lit("validation").alias("split"))
    round_counts = {
        row["match_id"]: row["len"]
        for row in samples.select("match_id", "demo_sha256", "round_id")
        .unique()
        .group_by("match_id")
        .len()
        .iter_rows(named=True)
    }
    expected = {row["match_id"]: sum(row["expected_score"].values()) for row in plan["archives"]}
    if round_counts != expected:
        raise PipelineError(f"Model sample rounds differ from published scores: {round_counts}")
    validate_timeline(join_samples(tables), recording_rounds(ROOT, sources))
    identities = audit_identities(tables["nodes"], samples)
    metadata: dict[str, Any] = {
        "dataset_schema_version": 1,
        "feature_version": 1,
        "map_name": "de_mirage",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "feature_columns": feature_columns(1),
        "label": "y_ct_win: CT=1, T=0; labels stored separately",
        "source_manifest_sha256": sha256_file(PLAN),
        "reserved_event": EXCLUDED_EVENT,
        "split_policy": {
            "group_by": "event",
            "method": HISTORICAL_VALIDATION_ONLY,
            "evaluation_ready": False,
            "reason": plan["purpose"],
            "match_counts": {"validation": len(expected)},
        },
        "sources": sources,
        "excluded_duplicate_archives": duplicates,
        "identity_audit": identities,
        "match_count": len(expected),
        "round_count": sum(round_counts.values()),
        "sample_count": samples.height,
        "node_count": tables["nodes"].height,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".validation-stage-", dir=out.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        for name, frame in tables.items():
            path = stage / f"{name}.parquet"
            frame.write_parquet(path, compression="zstd")
        metadata["file_sha256"] = {
            f"{name}.parquet": sha256_file(stage / f"{name}.parquet") for name in TABLE_NAMES
        }
        write_json(stage / "dataset.json", metadata)
        _, restored = read_dataset(stage)
        if any(not restored[name].equals(frame) for name, frame in tables.items()):
            raise PipelineError("Historical validation Parquet roundtrip changed data.")
        stage.rename(out)
    return metadata


def run(plan: dict[str, Any], processed: Path, extracted: Path, out: Path, progress: Path) -> int:
    if out.exists():
        raise PipelineError(f"Dataset already exists: {out}; choose a fresh output.")
    if not UNRAR.is_file():
        raise PipelineError(f"UnRAR was not found at {UNRAR}.")
    paths = (processed, extracted, out, progress)
    if any(not path.resolve().is_relative_to(ROOT) for path in paths):
        raise PipelineError("All outputs must remain in the project workspace.")
    used_matches, used_hashes, used_events = prior_index()
    if any(
        row["match_id"] in used_matches or row["event"] in used_events for row in plan["archives"]
    ):
        raise PipelineError("The additional validation batch overlaps a previous match or event.")
    duplicates = []
    for item in plan["archives"]:
        archive = RAW / item["archive"]
        if not archive.is_file():
            raise PipelineError(f"Missing pinned archive: {archive}")
        copy_name = item.get("possible_duplicate")
        if copy_name:
            if not archive_basename(copy_name) or copy_name == archive.name:
                raise PipelineError("Duplicate archive must be another basename.")
            copy = RAW / copy_name
            if copy.is_file():
                if sha256_file(copy) != item["archive_sha256"]:
                    raise PipelineError(f"Archive copy is not byte-identical: {copy}")
                duplicates.append(
                    {
                        "archive": copy_name,
                        "duplicate_of": archive.name,
                        "sha256": item["archive_sha256"],
                    }
                )
    result: dict[str, Any] = {
        "status": "running",
        "total": len(plan["archives"]),
        "completed": [],
        "failed": [],
        "current": None,
        "plan_sha256": sha256_file(PLAN),
        "excluded_duplicate_archives": duplicates,
    }
    update(progress, result)
    for index, item in enumerate(plan["archives"], start=1):
        identifier = item["match_id"]
        result["current"] = {"index": index, "id": identifier, "matchup": item["matchup"]}
        update(progress, result)
        print(f"[{index}/{result['total']}] {item['matchup']} ({identifier})", flush=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "scripts.import_validation_more",
            "--worker",
            identifier,
            "--processed",
            str(processed),
            "--extracted",
            str(extracted),
        ]
        output = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONUTF8": "1"},
            text=True,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        print(output.stdout, end="", flush=True)
        if output.stderr:
            print(output.stderr, end="", file=sys.stderr, flush=True)
        if output.returncode:
            result["failed"].append(
                {
                    "match_id": identifier,
                    "exit_code": output.returncode,
                    "error": output.stderr.strip(),
                }
            )
        else:
            checked = processed / identifier.replace(":", "-") / "checked.json"
            result["completed"].append(json.loads(checked.read_text("utf-8")))
        update(progress, result)
    result["current"] = None
    result["status"] = "completed_with_errors" if result["failed"] else "completed"
    if not result["failed"]:
        try:
            metadata = publish_dataset(plan, result["completed"], used_hashes, out, duplicates)
            result["dataset"] = out.relative_to(ROOT).as_posix()
            result["dataset_sha256"] = sha256_file(out / "dataset.json")
            print(
                f"Verified {metadata['match_count']} matches, {metadata['round_count']} rounds, "
                f"{metadata['sample_count']} snapshots.",
                flush=True,
            )
        except (OSError, ValueError, KeyError, PipelineError, pl.exceptions.PolarsError) as exc:
            result["status"] = "completed_with_errors"
            result["failed"].append({"match_id": "dataset", "error": str(exc)})
            update(progress, result)
            raise
    update(progress, result)
    return int(bool(result["failed"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=PROCESSED)
    parser.add_argument("--extracted", type=Path, default=EXTRACTED)
    parser.add_argument("--out", type=Path, default=DATASET)
    parser.add_argument("--progress", type=Path, default=PROGRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        plan = load_plan()
        if args.worker:
            item = next(row for row in plan["archives"] if row["match_id"] == args.worker)
            worker(item, args.processed.resolve(), args.extracted.resolve())
            return 0
        return run(
            plan,
            args.processed.resolve(),
            args.extracted.resolve(),
            args.out.resolve(),
            args.progress.resolve(),
        )
    except (OSError, ValueError, KeyError, PipelineError, pl.exceptions.PolarsError) as exc:
        print(f"Historical validation import failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

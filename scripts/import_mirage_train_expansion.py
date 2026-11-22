"""Verify new Mirage recordings and publish the supplemental training tables."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from cs2_pipeline.dataset import build_dataset, held_out_event, read_dataset, slug, write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_data import ROOT
from cs2_pipeline.progress import ProgressReporter, emit
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import join_samples, recording_rounds, validate_timeline
from cs2_pipeline.validate import validate_directory
from scripts.audit_phase2_data import audit_identities
from scripts.import_holdout_50 import TEAM_ALIASES, UNRAR, import_archive, published_score
from scripts.import_validation_more import archive_basename

PLAN = ROOT / "configs/mirage_train_expansion_20261121.json"
PUBLISHED_TEAM_ALIASES = {
    "navi": "Natus Vincere",
    "nip": "Ninjas in Pyjamas",
    "k27 esports": "K27",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_source(plan: dict[str, Any], item: dict[str, Any]) -> Path | None:
    prefix = f"{item['event']}-{slug(item['matchup'])}-bo3-"
    archives = [
        path
        for path in (ROOT / plan["raw_directory"]).glob("*.rar")
        if path.name.lower().startswith(prefix)
    ]
    if len(archives) > 1 or (archives and not archive_basename(archives[0].name)):
        raise PipelineError(f"Archive name must unambiguously identify {item['match_id']}.")
    return archives[0] if archives else None


def checked_paths(receipt: dict[str, Any], processed: Path) -> list[dict[str, Any]]:
    manifest = ROOT / receipt["source_manifest"]
    entries = read_json(manifest)["sources"]
    normalized = []
    for entry in entries:
        source = (manifest.parent / entry["path"]).resolve()
        if not source.is_relative_to(processed) or not validate_directory(source)["passed"]:
            raise PipelineError(f"Invalid processed recording at {source}.")
        normalized.append({**entry, "path": source.as_posix()})
    return normalized


def existing_sources() -> tuple[set[str], set[str], set[str]]:
    inputs = (
        "mirage_expanded_2026_v1",
        "mirage_validation_more_v1",
        "mirage_holdout_50_v1",
        "esl24_holdout_v1",
    )
    # Only provenance is needed to exclude previously used matches, events and recordings.
    sources = [
        row
        for name in inputs
        for row in read_json(ROOT / "data/datasets" / name / "dataset.json")["sources"]
    ]
    return tuple({row[key] for row in sources} for key in ("match_id", "event", "demo_sha256"))


def run(
    plan: dict[str, Any],
    out: Path,
    processed: Path,
    extracted: Path,
    reporter: ProgressReporter,
) -> int:
    if out.exists():
        raise PipelineError(f"Choose a fresh dataset path: {out}.")
    matches, events, recordings = existing_sources()
    roster = plan["matches"]
    new_ids = {row["match_id"] for row in roster}
    if (
        len(new_ids) != len(roster)
        or new_ids & matches
        or any(
            row["event"] in events or held_out_event(row["event"]) or held_out_event(row["url"])
            for row in roster
        )
    ):
        raise PipelineError("The supplemental roster duplicates an earlier match or event.")

    completed, absent, failed, entries = [], [], [], []
    for item in roster:
        identifier = item["match_id"]
        archive = find_source(plan, item)
        if archive is None:
            absent.append(identifier)
            emit(reporter, "load", "active", f"Archive pending · {item['matchup']} ({identifier})")
            continue
        checked = processed / identifier.replace(":", "-") / "checked.json"
        emit(
            reporter,
            "load",
            "active",
            f"Checking Mirage · {item['matchup']} ({identifier})",
            completed=len(completed),
            total=len(roster),
            match_id=identifier,
        )
        try:
            if not checked.is_file():
                subprocess.run(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "scripts.import_mirage_train_expansion",
                        "--worker",
                        identifier,
                        "--archive",
                        str(archive),
                        "--out",
                        str(out),
                        "--processed",
                        str(processed),
                        "--extracted",
                        str(extracted),
                        "--progress-file",
                        str(reporter.path),
                    ],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONUTF8": "1"},
                    check=True,
                )
            receipt = read_json(checked)
            sources = checked_paths(receipt, processed)
            paths = [Path(source["path"]) for source in sources]
            digests = {read_json(path / "metadata.json")["demo_sha256"] for path in paths}
            if (
                receipt["match_id"] != identifier
                or receipt["event"] != item["event"]
                or receipt["match_date"] != item["match_date"]
                or receipt["archive_sha256"] != sha256_file(archive)
                or receipt["published_score_verified"]
                != published_score(paths, item["expected_score"])
                or digests != set(receipt["demo_sha256"])
                or digests & recordings
            ):
                raise PipelineError(f"Frozen replay identity or score differs for {identifier}.")
            recordings.update(digests)
            completed.append({**receipt, "archive": archive.name})
            entries.extend(sources)
            emit(reporter, "load", "active", f"Verified published score · {item['matchup']}")
        except (OSError, ValueError, PipelineError, subprocess.CalledProcessError) as exc:
            failed.append({"match_id": identifier, "reason": str(exc)})
            emit(reporter, "load", "active", f"Archive needs review · {identifier} · {exc}")

    receipt = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_file(PLAN),
        "matches": completed,
        "missing": absent,
        "failed": failed,
    }
    receipt_path = reporter.path.parent / f"{out.name}_intake.json"
    if failed or not completed:
        write_json(receipt_path, receipt)
        raise PipelineError("Review failed archives before the supplemental dataset is published.")

    processed.mkdir(parents=True, exist_ok=True)
    sources_manifest = processed / f"{out.name}_sources.json"
    write_json(
        sources_manifest, {"map_name": "de_mirage", "feature_version": 1, "sources": entries}
    )
    build_dataset(sources_manifest, out, group_by="event")
    meta, tables = read_dataset(out)
    samples = tables["samples"]
    if (
        set(samples["match_id"]) != {item["match_id"] for item in completed}
        or set(tables["splits"]["split"]) != {"train"}
        or meta["round_count"] != sum(item["rounds"] for item in completed)
    ):
        raise PipelineError("Supplemental dataset drops planned rounds or changes fit membership.")
    validate_timeline(join_samples(tables), recording_rounds(ROOT, meta["sources"]))
    receipt.update(
        dataset=out.relative_to(ROOT).as_posix(),
        dataset_sha256=sha256_file(out / "dataset.json"),
        map_count=len(completed),
        rounds=meta["round_count"],
        ticks=meta["sample_count"],
        identity_audit=audit_identities(tables["nodes"], samples),
    )
    write_json(receipt_path, receipt)
    emit(
        reporter,
        "load",
        "completed",
        f"Verified {len(completed)} new matches · {meta['round_count']} rounds · "
        f"{meta['sample_count']} ticks · {len(absent)} missing archive(s).",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--processed", required=True, type=Path)
    parser.add_argument("--extracted", required=True, type=Path)
    parser.add_argument("--progress-file", required=True, type=Path)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--archive", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out, processed, extracted = (
        value.resolve() for value in (args.out, args.processed, args.extracted)
    )
    if any(not path.is_relative_to(ROOT) for path in (out, processed, extracted)):
        parser.error("Replay artifacts must stay in the workspace.")
    plan = read_json(PLAN)
    if version("awpy") != "3.3.0" or not UNRAR.is_file():
        parser.error("This import requires the locked Awpy 3.3.0 and WinRAR's UnRAR.")
    # Normalize verified display names for scoreboard matching only.
    TEAM_ALIASES.update(PUBLISHED_TEAM_ALIASES)
    if args.worker:
        item = next(row for row in plan["matches"] if row["match_id"] == args.worker)
        ref = {**item, "expected_rounds": sum(item["expected_score"].values())}
        import_archive(args.archive, item, ref, extracted_root=extracted, processed_root=processed)
        return 0
    reporter = ProgressReporter(args.progress_file)
    try:
        return run(plan, out, processed, extracted, reporter)
    except (OSError, ValueError, PipelineError) as exc:
        emit(reporter, "run", "failed", str(exc))
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

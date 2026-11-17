"""Add causal context to already-parsed recordings, without redoing player snapshots."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cs2_pipeline.context import enrich_demo, read_context
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.phase2_data import ROOT, sources
from cs2_pipeline.schema import PipelineError


def update(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result["updated_at_utc"] = datetime.now(UTC).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # On Windows, the dashboard may briefly hold the old file open while polling.
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025)


def run(root: Path, out: Path, progress: Path) -> dict[str, Any]:
    entries = sources(root)
    index: dict[str, list[Path]] = defaultdict(list)
    for raw_root in (root / "data/raw", root / "data/datasets/mirage demo"):
        if raw_root.exists():
            for path in raw_root.rglob("*.dem"):
                index[path.name].append(path)
    status: dict[str, Any] = {
        "status": "running",
        "total": len(entries),
        "completed": 0,
        "failed": 0,
        "event_only": 0,
        "current": None,
        "errors": [],
        "processed": [],
        "note": "仅补充历史 Demo 的事件与规则；复用已有玩家快照；禁止拟合留出集。",
    }
    update(progress, status)
    for item in entries:
        source, metadata, processed = item["source"], item["metadata"], item["path"]
        checksum = metadata["demo_sha256"]
        destination = out / checksum
        status["current"] = f"{source['match_id']} / {metadata['demo_filename']}"
        update(progress, status)
        try:
            if destination.exists():
                read_context(destination, processed, checksum)
            else:
                candidates = index.get(metadata["demo_filename"], [])
                if len(candidates) > 1:
                    candidates = [p for p in candidates if sha256_file(p) == checksum]
                if len(candidates) != 1:
                    raise PipelineError(f"原始录制无法唯一匹配：{metadata['demo_filename']}")
                enrich_demo(candidates[0], processed, destination, allow_event_only=True)
                read_context(destination, processed, checksum)
            notes = json.loads((destination / "context_metadata.json").read_text("utf-8"))
            if notes.get("game_rules_status") == "event_only":
                status["event_only"] += 1
            status["completed"] += 1
            status["processed"].append({"match_id": source["match_id"], "sha256": checksum})
        except Exception as exc:
            status["failed"] += 1
            status["errors"].append(
                {"match_id": source["match_id"], "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"上下文补充失败：{status['current']}：{exc}", file=sys.stderr, flush=True)
        update(progress, status)
    status["current"] = None
    status["status"] = "completed_with_errors" if status["failed"] else "completed"
    update(progress, status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/context/phase2_audit")
    parser.add_argument("--progress", type=Path, default=ROOT / "runs/context_audit/progress.json")
    args = parser.parse_args()
    try:
        result = run(ROOT, args.out, args.progress)
        print(f"完成 {result['completed']}/{result['total']}；失败 {result['failed']}。")
        return int(result["failed"] > 0)
    except (OSError, KeyError, ValueError, PipelineError) as exc:
        print(f"上下文批处理无法启动：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

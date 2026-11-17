"""Enrich a verified Phase 1 corpus and create a feature-version-2 manifest."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from cs2_pipeline.context import enrich_demo, read_context
from cs2_pipeline.dataset import write_json
from cs2_pipeline.schema import PipelineError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/phase2_corpus.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/corpus"))
    parser.add_argument("--context", type=Path, default=Path("data/context/corpus"))
    parser.add_argument("--out", type=Path, default=Path("configs/phase2_corpus_v2.json"))
    args = parser.parse_args()
    original = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources = []
    for entry in original["sources"]:
        processed = (args.manifest.resolve().parent / entry["path"]).resolve()
        metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
        filename = metadata["demo_filename"]
        if Path(filename).name != filename:
            raise PipelineError("Phase 1 demo_filename 必须是文件名，不能包含路径。")
        out = args.context / processed.name
        if out.exists():
            read_context(out, processed, metadata["demo_sha256"])
            print(f"复用已验证上下文：{out}", flush=True)
        else:
            notes = enrich_demo(args.raw / filename, processed, out)
            print(f"上下文完成：{processed.name}，{notes['row_count']} 行", flush=True)
        sources.append(
            {
                **entry,
                "path": Path(os.path.relpath(processed, args.out.resolve().parent)).as_posix(),
                "context_path": Path(
                    os.path.relpath(out.resolve(), args.out.resolve().parent)
                ).as_posix(),
            }
        )
    result = {**original, "feature_version": 2, "sources": sources}
    if args.out.exists() and json.loads(args.out.read_text(encoding="utf-8")) != result:
        raise PipelineError("目标清单已存在且内容不同；请指定新的 --out。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, result)
    print(f"V2 清单：{args.out.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"补充上下文失败：{type(exc).__name__}: {exc}")
        raise SystemExit(1) from None

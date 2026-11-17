"""Reparse a pinned corpus with one Awpy version, preserving prior artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.dataset import held_out_event, write_json
from cs2_pipeline.parse import parse_demo, sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory

ROUND_KEYS = [
    "round_num",
    "start_tick",
    "freeze_end_tick",
    "end_tick",
    "winner_side",
    "is_valid_round",
]


def reparse(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if version("awpy") != plan["awpy_version"]:
        raise PipelineError(f"需要 Awpy {plan['awpy_version']}；请按锁文件安装依赖。")
    base = plan_path.resolve().parent
    report_path = (base / plan["report_path"]).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"complete": False, "parser_version": version("awpy"), "sources": []}
    seen: set[str] = set()
    for entry in plan["sources"]:
        if held_out_event(entry["event"]) or held_out_event(entry.get("match_url", "")):
            raise PipelineError("计划包含 Bucharest 保留赛事。")
        if entry["demo_sha256"] in seen:
            raise PipelineError("计划含重复 Demo SHA256。")
        seen.add(entry["demo_sha256"])
        raw = (base / entry["demo_path"]).resolve()
        original = (base / entry["original_processed"]).resolve()
        out = (base / entry["new_processed"]).resolve()
        if sha256_file(original / "metadata.json") != entry["original_metadata_sha256"]:
            raise PipelineError(f"原始解析元数据已改变：{original}")
        if sha256_file(original / "rounds.parquet") != entry["original_rounds_sha256"]:
            raise PipelineError(f"原始回合文件已改变：{original}")
        old = json.loads((original / "metadata.json").read_text(encoding="utf-8"))
        if (
            old.get("demo_sha256") != entry["demo_sha256"]
            or sha256_file(raw) != entry["demo_sha256"]
        ):
            raise PipelineError(f"原始 Demo SHA256 不符：{raw}")
        source = old.get("external_source")
        source_path = out.with_suffix(".source.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        if source:
            if (
                source_path.exists()
                and json.loads(source_path.read_text(encoding="utf-8")) != source
            ):
                raise PipelineError(f"新来源文件与原来源不符：{source_path}")
            if not source_path.exists():
                write_json(source_path, source)
        print(
            f"统一解析 {len(result['sources']) + 1}/{len(plan['sources'])}: {raw.name}", flush=True
        )
        if out.exists():
            current = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
            if (
                current.get("demo_sha256") != entry["demo_sha256"]
                or current.get("parser_version") != version("awpy")
                or current.get("external_source") != source
                or current.get("sampling_interval_seconds") != 5.0
            ):
                raise PipelineError(f"已有输出使用不同 Demo 或解析版本：{out}")
            quality = validate_directory(out)
        else:
            quality = parse_demo(raw, out, interval=5, source=source_path if source else None)
        if not quality["passed"]:
            raise PipelineError(f"质量检查失败：{out}: {quality['errors']}")
        old_rounds = pl.read_parquet(original / "rounds.parquet").select(ROUND_KEYS)
        new_rounds = pl.read_parquet(out / "rounds.parquet").select(ROUND_KEYS)
        if not old_rounds.equals(new_rounds):
            raise PipelineError(f"新版解析器的回合边界或胜方不同，需要人工核对：{out}")
        result["sources"].append(
            {
                "match_id": entry["match_id"],
                "demo_sha256": entry["demo_sha256"],
                "rounds": quality["round_count"],
                "valid_ticks": quality["valid_sample_tick_count"],
                "warnings": quality["warnings"],
                "passed": True,
            }
        )
        write_json(report_path, result)
    result["complete"] = True
    write_json(report_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/reparse_expanded_2026.json"))
    args = parser.parse_args()
    try:
        report = reparse(args.plan)
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"统一解析失败：{exc}", file=sys.stderr)
        return 1
    print(f"统一解析完成：{len(report['sources'])} 个录制文件，Awpy {report['parser_version']}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

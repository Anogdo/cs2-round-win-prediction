"""Import checksum-pinned local recordings, keeping multipart maps together."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl
from awpy import Demo

from cs2_pipeline.dataset import held_out_event, write_json
from cs2_pipeline.parse import parse_demo, sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory


def check_round_sequence(parts: list[list[int]], expected: int) -> None:
    """A recording boundary must not duplicate, renumber, or omit a played round."""
    sequence = [number for part in parts for number in part]
    if not parts or any(not part for part in parts) or sequence != list(range(1, expected + 1)):
        raise PipelineError("分段回合有缺口、重复或顺序错误；请补齐原始分段，不能重编号。")


def import_local(manifest: Path, processed: Path, dataset_manifest: Path) -> dict[str, Any]:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    report: dict[str, Any] = {"complete": False, "maps": []}
    accepted: list[dict[str, Any]] = []
    checksums: set[str] = set()
    groups: set[str] = set()
    processed.mkdir(parents=True, exist_ok=True)
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    for group in config["maps"]:
        if held_out_event(group["event"]) or held_out_event(group["match_url"]):
            raise PipelineError("本地来源包含 Bucharest 保留赛事，拒绝导入。")
        map_id = f"{group['match_id']}:m{group['map_index']}"
        if map_id in groups:
            raise PipelineError(f"地图重复声明：{map_id}")
        groups.add(map_id)
        parts = group["parts"]
        check_round_sequence([part["round_numbers"] for part in parts], group["expected_rounds"])
        paths = [(manifest.resolve().parent / p["path"]).resolve() for p in parts]
        for path, part in zip(paths, parts, strict=True):
            if not path.is_file():
                raise PipelineError(f"找不到分段 Demo：{path}")
            if path.stat().st_size != part["size_bytes"]:
                raise PipelineError(f"Demo 大小与固定清单不符：{path.name}")
            if part["sha256"] in checksums:
                raise PipelineError(f"重复 Demo SHA256：{path.name}")
            checksums.add(part["sha256"])
            if sha256_file(path) != part["sha256"]:
                raise PipelineError(f"Demo 内容已改变：{path.name}；请重新核验来源清单。")
        # Awpy validates multipart server/map/build and unfinished-round boundaries.
        native = Demo(paths if len(paths) > 1 else paths[0])
        native.load("rounds")
        if native.header.get("map_name") != config["map_name"]:
            raise PipelineError(f"地图与清单不一致：{map_id}")
        if native.rounds["round_num"].to_list() != list(range(1, group["expected_rounds"] + 1)):
            raise PipelineError(f"Awpy 完整地图回合与清单不符：{map_id}")
        map_report: dict[str, Any] = {
            "map_id": map_id,
            "native_multipart_rounds": len(native.rounds),
            "parts": [],
        }
        del native
        for index, (path, part) in enumerate(zip(paths, parts, strict=True), start=1):
            identifier = f"{group['match_id'].replace(':', '-')}-m{group['map_index']}-p{index}"
            output = processed / identifier
            source = {
                **{key: group[key] for key in ("match_id", "match_date", "event", "match_url")},
                "map": config["map_name"],
                "sha256": part["sha256"],
                "local_filename": path.name,
                "provenance": group.get(
                    "provenance",
                    "User supplied: IEM Cologne Major 2026; match/date checked on HLTV",
                ),
                "map_id": map_id,
                "part_index": index,
                "part_count": len(parts),
                "tick_scope": "Original local recording tick; never concatenate ticks across parts",
            }
            for key in ("archive_filename", "archive_sha256"):
                if key in group:
                    source[key] = group[key]
            source_file = processed / f"{identifier}.source.json"
            if source_file.exists():
                if json.loads(source_file.read_text(encoding="utf-8")) != source:
                    raise PipelineError(f"来源清单冲突：{source_file}")
            else:
                write_json(source_file, source)
            print(f"解析 {path.name}（分段 {index}/{len(parts)}）……", flush=True)
            if output.exists():
                metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                if (
                    metadata.get("demo_sha256") != part["sha256"]
                    or metadata.get("external_source") != source
                    or metadata.get("sampling_interval_seconds") != 5.0
                ):
                    raise PipelineError(f"已有结果的 Demo、来源或 5 秒采样策略不符：{output}")
                quality = validate_directory(output)
            else:
                quality = parse_demo(path, output, 5.0, source=source_file)
            if not quality["passed"]:
                raise PipelineError(f"{path.name} 验证失败：{quality['errors']}")
            rounds = pl.read_parquet(output / "rounds.parquet")
            if rounds["round_num"].to_list() != part["round_numbers"]:
                raise PipelineError(f"{path.name} 原始回合编号与清单不一致。")
            accepted.append(
                {
                    "path": Path(
                        os.path.relpath(output.resolve(), dataset_manifest.resolve().parent)
                    ).as_posix(),
                    **{key: source[key] for key in ("match_id", "match_date", "event")},
                    "map_id": map_id,
                    "part_index": str(index),
                }
            )
            map_report["parts"].append(
                {"id": identifier, "sha256": part["sha256"], "quality": quality}
            )
        report["maps"].append(map_report)
        write_json(processed / "import_report.json", report)
    result = {"schema_version": 1, "map_name": config["map_name"], "sources": accepted}
    if dataset_manifest.exists():
        if json.loads(dataset_manifest.read_text(encoding="utf-8")) != result:
            raise PipelineError("数据集清单已存在且内容不同；请使用新路径。")
    else:
        write_json(dataset_manifest, result)
    report.update(complete=True, recording_count=len(accepted), map_count=len(groups))
    write_json(processed / "import_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/cologne_local.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/cologne"))
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("configs/phase2_cologne.json")
    )
    args = parser.parse_args(argv)
    try:
        result = import_local(args.manifest, args.processed, args.dataset_manifest)
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"本地导入失败：{exc}", file=sys.stderr)
        return 1
    print(f"已验证 {result['recording_count']} 个真实录制文件 / {result['map_count']} 张完整地图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract only pinned Mirage files from user-supplied HLTV archives and parse them."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from cs2_pipeline.dataset import held_out_event, write_json
from cs2_pipeline.local import import_local
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.schema import PipelineError


def extract_sources(manifest: Path, unrar: Path, output: Path, local_manifest: Path) -> None:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    maps: list[dict[str, Any]] = []
    output.mkdir(parents=True, exist_ok=True)
    for item in config["maps"]:
        if held_out_event(item["event"]) or held_out_event(item["match_url"]):
            raise PipelineError("保留赛事不可导入。")
        archive = (manifest.resolve().parent / item["archive"]).resolve()
        if not archive.is_file() or sha256_file(archive) != item["archive_sha256"]:
            raise PipelineError(f"压缩包不存在或 SHA256 不符：{archive}")
        entry = item["entry"]
        if (
            Path(entry).name != entry
            or entry.startswith("-")
            or not entry.lower().endswith("-mirage.dem")
        ):
            raise PipelineError("压缩包内文件必须是无路径的 Mirage .dem 文件名。")
        identifier = item["match_id"].replace(":", "-")
        folder = output / identifier
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / entry
        checksum_file = folder / "extraction.json"
        if target.exists():
            existing = json.loads(checksum_file.read_text(encoding="utf-8"))
            if existing["archive_sha256"] != item["archive_sha256"] or existing[
                "demo_sha256"
            ] != sha256_file(target):
                raise PipelineError(f"已有文件与固定来源不符，拒绝覆盖：{target}")
        else:
            print(f"解压 {archive.name} → {entry}", flush=True)
            with tempfile.TemporaryDirectory(prefix=".extract-", dir=folder) as temporary:
                directory = Path(temporary)
                command = [
                    str(unrar),
                    "e",
                    "-p-",
                    "-o-",
                    "-idq",
                    str(archive),
                    entry,
                    str(directory) + os.sep,
                ]
                result = subprocess.run(command, capture_output=True, text=True, check=False)
                extracted = directory / entry
                if (
                    result.returncode != 0
                    or not extracted.is_file()
                    or extracted.stat().st_size == 0
                ):
                    raise PipelineError(
                        f"解压失败：{archive.name}；{result.stderr or result.stdout}"
                    )
                digest = sha256_file(extracted)
                extracted.rename(target)
                write_json(
                    checksum_file,
                    {
                        "archive_sha256": item["archive_sha256"],
                        "demo_sha256": digest,
                        "entry": entry,
                    },
                )
        digest = sha256_file(target)
        maps.append(
            {
                **{
                    k: item[k]
                    for k in (
                        "match_id",
                        "match_date",
                        "event",
                        "match_url",
                        "map_index",
                        "expected_rounds",
                    )
                },
                "archive_filename": archive.name,
                "archive_sha256": item["archive_sha256"],
                "provenance": "User downloaded from HLTV; match/date/map cross-checked with HLTV.",
                "parts": [
                    {
                        "path": Path(
                            os.path.relpath(target.resolve(), local_manifest.resolve().parent)
                        ).as_posix(),
                        "sha256": digest,
                        "size_bytes": target.stat().st_size,
                        "round_numbers": list(range(1, item["expected_rounds"] + 1)),
                    }
                ],
            }
        )
    local = {"schema_version": 1, "map_name": "de_mirage", "maps": maps}
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    if local_manifest.exists() and json.loads(local_manifest.read_text(encoding="utf-8")) != local:
        raise PipelineError("本地来源清单已存在且不一致，拒绝覆盖。")
    if not local_manifest.exists():
        write_json(local_manifest, local)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/user_hltv_sources.json"))
    parser.add_argument(
        "--unrar",
        type=Path,
        default=Path(shutil.which("unrar") or r"C:\Program Files\WinRAR\UnRAR.exe"),
    )
    parser.add_argument("--raw", type=Path, default=Path("data/raw/new_hltv_extracted"))
    parser.add_argument("--local-manifest", type=Path, default=Path("configs/user_hltv_local.json"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/new_hltv"))
    parser.add_argument(
        "--dataset-manifest", type=Path, default=Path("configs/phase2_new_hltv.json")
    )
    args = parser.parse_args()
    try:
        if not args.unrar.is_file():
            raise PipelineError("找不到 UnRAR；请使用 --unrar 指定已安装的解压程序。")
        extract_sources(args.manifest, args.unrar, args.raw, args.local_manifest)
        report = import_local(args.local_manifest, args.processed, args.dataset_manifest)
    except (PipelineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"用户 Demo 导入失败：{exc}", file=sys.stderr)
        return 1
    print(f"已解析并验证 {report['map_count']} 张真实 Mirage 地图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

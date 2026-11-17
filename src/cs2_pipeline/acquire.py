"""Download a pinned public corpus, verify checksums and parse each real Demo."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from cs2_pipeline.dataset import held_out_event, write_json
from cs2_pipeline.parse import parse_demo, sha256_file
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory


def validate_source(source: dict[str, Any]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", str(source.get("id", ""))):
        raise PipelineError("来源 id 只能含字母、数字、下划线和连字符，不能包含路径。")
    if not re.fullmatch(r"[a-f0-9]{64}", str(source.get("sha256", ""))):
        raise PipelineError("每个下载来源必须固定 SHA256。")
    if not isinstance(source.get("size_bytes"), int) or source["size_bytes"] <= 0:
        raise PipelineError("下载来源必须声明正的 size_bytes。")
    if urlparse(str(source.get("download_url", ""))).scheme != "https":
        raise PipelineError("仅支持明确的公开 HTTPS 下载地址。")
    if not source.get("license") or not source.get("attribution"):
        raise PipelineError("下载来源必须声明许可和署名，不能猜测可用权限。")


def download_verified(source: dict[str, Any], destination: Path) -> bool:
    """Return true for a new download, false for a verified cached file."""
    validate_source(source)
    if destination.exists():
        if (
            destination.stat().st_size != source["size_bytes"]
            or sha256_file(destination) != source["sha256"]
        ):
            raise PipelineError(f"已有文件 {destination} 与来源哈希不符，拒绝覆盖。")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source["download_url"], headers={"User-Agent": "cs2-pipeline/0.1"}
    )
    with tempfile.TemporaryDirectory(prefix=".download-", dir=destination.parent) as temporary:
        partial = Path(temporary) / "download.dem"
        digest_bytes = 0
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                digest_bytes += len(chunk)
                if digest_bytes > source["size_bytes"]:
                    raise PipelineError("下载内容超出声明大小；已拒绝保存。")
                handle.write(chunk)
        if (
            partial.stat().st_size != source["size_bytes"]
            or sha256_file(partial) != source["sha256"]
        ):
            raise PipelineError("下载文件大小或 SHA256 不符，未保存为 Demo。")
        partial.rename(destination)
    return True


def acquire_corpus(
    manifest: Path,
    raw: Path,
    processed: Path,
    dataset_manifest: Path,
    *,
    interval: float = 5.0,
) -> dict[str, Any]:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("sources"), list):
        raise PipelineError("下载清单必须含 sources 列表。")
    raw.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"acquisition_schema_version": 1, "sources": [], "complete": True}
    accepted = []
    for source in config["sources"]:
        if not isinstance(source, dict):
            raise PipelineError("下载清单 sources 中每项必须是对象。")
        entry = {"id": source.get("id"), "status": "pending"}
        if held_out_event(str(source.get("event", ""))) or held_out_event(
            str(source.get("match_url", ""))
        ):
            entry["status"] = "excluded_reserved_event"
            report["sources"].append(entry)
            continue
        try:
            validate_source(source)
            demo = raw / f"{source['id']}.dem"
            source_json = raw / f"{source['id']}.source.json"
            output = processed / source["id"]
            print(f"获取 {source['id']}（{source['size_bytes'] / 1e6:.0f} MB）……", flush=True)
            entry["downloaded"] = download_verified(source, demo)
            if source_json.exists():
                if json.loads(source_json.read_text(encoding="utf-8")) != source:
                    raise PipelineError("已有来源 JSON 与固定清单不一致，拒绝覆盖。")
            else:
                write_json(source_json, source)
            if output.exists():
                metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
                if metadata.get("demo_sha256") != source["sha256"]:
                    raise PipelineError("已有解析结果属于不同 Demo，拒绝复用。")
                quality = validate_directory(output)
            else:
                quality = parse_demo(demo, output, interval, source=source_json)
            if not quality["passed"]:
                raise PipelineError(f"Demo 未通过质量验证：{quality['errors']}")
            entry.update(status="accepted", quality=quality)
            accepted.append(
                {
                    "path": Path(
                        os.path.relpath(output.resolve(), dataset_manifest.resolve().parent)
                    ).as_posix(),
                    **{key: source[key] for key in ("match_id", "match_date", "event")},
                }
            )
        except urllib.error.HTTPError as exc:
            entry.update(status="download_failed", error=f"HTTP {exc.code}: {exc.reason}")
            report["complete"] = False
            if exc.code in (401, 403, 429):
                report["sources"].append(entry)
                print("下载来源要求授权或限制访问，已停止，不绕过保护。", file=sys.stderr)
                break
        except Exception as exc:
            entry.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["complete"] = False
            print(f"记录并排除 {source.get('id')}：{exc}", file=sys.stderr, flush=True)
        report["sources"].append(entry)
        write_json(raw / "acquisition_report.json", report)
    report["accepted_count"] = len(accepted)
    write_json(raw / "acquisition_report.json", report)
    if not accepted:
        raise PipelineError("没有成功获取并验证的比赛；详见 acquisition_report.json。")
    output_config = {"schema_version": 1, "map_name": "de_mirage", "sources": accepted}
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    if (
        dataset_manifest.exists()
        and json.loads(dataset_manifest.read_text(encoding="utf-8")) != output_config
    ):
        raise PipelineError("目标数据集清单已存在且内容不同；请指定新的 --dataset-manifest。")
    write_json(dataset_manifest, output_config)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("configs/corpus_downloads.json"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/corpus"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed/corpus"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("configs/phase2_corpus.json"))
    parser.add_argument("--interval", type=float, default=5.0)
    args = parser.parse_args(argv)
    try:
        report = acquire_corpus(
            args.manifest, args.raw, args.processed, args.dataset_manifest, interval=args.interval
        )
    except (PipelineError, OSError, ValueError, TypeError) as exc:
        print(f"获取失败：{exc}", file=sys.stderr)
        return 1
    print(
        f"验证通过的真实 Demo：{report['accepted_count']}；清单：{args.dataset_manifest.resolve()}"
    )
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

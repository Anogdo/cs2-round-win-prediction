"""Download only the pinned, public Awpy fixture and verify its SHA256."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(out: Path) -> None:
    source_file = Path(__file__).resolve().parents[1] / "docs" / "demo_source.json"
    source = json.loads(source_file.read_text(encoding="utf-8"))
    if out.exists():
        if digest(out) == source["sha256"]:
            print(f"已存在且 SHA256 正确：{out}")
            return
        raise ValueError(f"{out} 已存在且内容不同；请使用其他 --out，避免覆盖用户 Demo。")
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载公开 Awpy 测试样例（约 347 MB）：{source['description']}", flush=True)
    request = urllib.request.Request(source["download_url"], headers={"User-Agent": "cs2-pipeline"})
    with tempfile.TemporaryDirectory(prefix=".download-", dir=out.parent) as temp:
        partial = Path(temp) / "match.dem"
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if partial.stat().st_size != source["size_bytes"] or digest(partial) != source["sha256"]:
            raise ValueError("大小或 SHA256 与官方清单不同；已拒绝保存，请稍后重试公开下载。")
        partial.rename(out)
    print(f"下载成功且 SHA256 已验证：{out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/match.dem"))
    args = parser.parse_args(argv)
    try:
        download(args.out)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(
            f"无法获取 Demo：{exc}。请按 README 人工下载并解压；不要绕过登录或验证码。",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

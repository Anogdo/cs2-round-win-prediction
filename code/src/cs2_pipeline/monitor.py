"""Loopback-only training monitor using the Python standard library."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cs2_pipeline.progress import STAGES
from cs2_pipeline.schema import PipelineError

PRESETS = {
    "matched34_score": {
        "name": "Mirage · 同 34 场模型对照",
        "dataset": "data/datasets/mirage_holdout_50_v1",
        "features": 47,
        "algorithm": "逻辑回归 / XGBoost / Transformer",
        "regularization_label": "Transformer 窗口",
        "regularization_value": "1 / 8",
        "module": "cs2_pipeline.matched34_score",
        "description": "同一 34 场拟合，在已观察 50 场统一评分；不重训模型。",
        "validation_name": "已观察 50 场（回顾性）",
        "test_name": "已观察 50 场 · 统一 Log Loss",
    },
    "phase2_temporal_validation": {
        "name": "Mirage · 冻结模型跨赛事验证",
        "dataset": "data/datasets/mirage_validation_more_v1",
        "features": 47,
        "algorithm": "逻辑回归 / Transformer · CUDA 评分",
        "regularization_label": "窗口（快照数）",
        "regularization_value": "1 / 8",
        "module": "cs2_pipeline.phase2_temporal_validation",
        "description": "验证 BLAST Bounty 与 Esports World Cup；复用 Porto 选定模型。",
        "validation_name": "新增历史验证（Porto 选定模型）",
    },
    "phase2_temporal": {
        "name": "Mirage · 时序 Transformer 对照",
        "dataset": "data/datasets/mirage_expanded_2026_v1",
        "features": 47,
        "algorithm": "逻辑回归 / Transformer · CUDA",
        "regularization_label": "窗口（快照数）",
        "regularization_value": "1 / 8",
        "module": "cs2_pipeline.phase2_temporal",
        "description": "Rio/Atlanta 拟合、Porto 选轮次；Cologne 保留，50 场仅作回顾。",
        "validation_name": "Porto（按 Log Loss 选择）",
        "test_name": "已观察 50 场 · Porto 选定模型",
    },
    "phase2_gpu": {
        "name": "Mirage · RTX CUDA 模型对照",
        "dataset": "data/datasets/mirage_expanded_2026_v1",
        "features": 47,
        "algorithm": "XGBoost · CUDA",
        "regularization_label": "L2 正则",
        "regularization_value": "5.0",
        "module": "cs2_pipeline.phase2_gpu",
        "description": "RTX cuda:0；Rio/Atlanta 拟合、Cologne 校准、Porto 验证；50 场已观察。",
        "validation_name": "Porto（GPU 18 场候选）",
        "test_name": "已观察 50 场 · GPU 34 场候选",
    },
    "phase2_comparison": {
        "name": "Mirage · 因果特征与浅层模型对照",
        "dataset": "data/datasets/mirage_expanded_2026_v1",
        "features": "2 / 47 / 96",
        "algorithm": "逻辑回归 / 浅树",
        "module": "cs2_pipeline.phase2_compare",
        "description": "Rio/Atlanta 拟合、Cologne 校准、Porto 验证；50 场为已观察探索集。",
        "validation_name": "Porto（V1 候选，2 场）",
        "test_name": "已观察 50 场 · 固定 V1 参考",
    },
    "expanded_2026_v1": {
        "name": "2026 多赛事 · V1 扩充训练",
        "dataset": "data/datasets/mirage_expanded_2026_v1",
        "features": 47,
        "description": "训练：Rio、Atlanta、Cologne、Porto；验证：Beijing；测试：BLAST Rivals。",
        "validation_name": "IEM Beijing",
        "test_name": "BLAST Rivals",
    },
    "rio_v1": {
        "name": "Rio · V1 状态基线",
        "dataset": "data/datasets/mirage_rio_v1",
        "features": 47,
        "description": "复跑已观察的 Rio V1 基准；不使用 Cologne 或 Bucharest。",
        "validation_name": "IEM Rio",
        "test_name": "IEM Rio",
    },
    "rio_v2": {
        "name": "Rio · V2 上下文与武器",
        "dataset": "data/datasets/mirage_rio_v2_checked",
        "features": 96,
        "description": "复跑已观察的 Rio V2 基准；不使用 Cologne 或 Bucharest。",
        "validation_name": "IEM Rio",
        "test_name": "IEM Rio",
    },
}
STATIC = Path(__file__).with_name("monitor_ui")
JOB_ID = re.compile(r"^[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$")


class LoopbackHTTPServer(ThreadingHTTPServer):
    """Keep Windows from binding a second monitor to the same local port."""

    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def events_at(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
        except ValueError:
            continue  # A final JSONL line may still be in flight.
    return events


def log_tail(path: Path) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 32768))
        return handle.read().decode("utf-8", errors="replace")


def write_record(path: Path, record: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class TrainingMonitor:
    """One owned child at a time; completed run records survive a server restart."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.runs = self.root / "runs"
        self.children: dict[str, subprocess.Popen] = {}
        self.lock = threading.RLock()

    def _jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for path in sorted(self.runs.glob("*/run.json"), reverse=True):
            record = read_json(path)
            identifier = path.parent.name
            if not record or not JOB_ID.fullmatch(identifier):
                continue
            events = events_at(path.parent / "progress.jsonl")
            child = self.children.get(identifier)
            changed = False
            if child is not None and child.poll() is not None and record["status"] == "running":
                complete = (self.root / record["output"] / "report.json").is_file()
                record.update(
                    status="completed"
                    if child.returncode == 0 and complete
                    else "cancelled"
                    if record.get("cancel_requested")
                    else "failed",
                    finished_at=datetime.now(UTC).isoformat(),
                    exit_code=child.returncode,
                )
                changed = True
            elif child is None and record["status"] in ("running", "unknown"):
                terminal = next((e for e in reversed(events) if e.get("stage") == "run"), {})
                if terminal.get("state") in ("completed", "failed"):
                    record["status"] = terminal["state"]
                else:
                    record["status"] = "unknown"
                # A restart cannot safely infer whether an unowned process is still fitting.
                changed = record["status"] != read_json(path).get("status")
            if changed:
                write_record(path, record)
            jobs.append(
                {
                    **record,
                    "events": events,
                    "log": log_tail(path.parent / "training.log"),
                    "report": read_json(self.root / record["output"] / "report.json"),
                }
            )
        return jobs

    def start(self, preset_id: str) -> dict[str, Any]:
        with self.lock:
            if preset_id not in PRESETS:
                raise PipelineError("请选择已登记的训练配置。")
            if any(j["status"] in ("running", "unknown") for j in self._jobs()):
                raise PipelineError("已有运行中或状态待确认的任务，请先处理该任务。")
            preset = PRESETS[preset_id]
            dataset = self.root / preset["dataset"]
            if not (dataset / "dataset.json").is_file():
                raise PipelineError(f"找不到已构建数据集：{dataset}；请先按 README 构建。")
            identifier = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)
            directory = self.runs / identifier
            directory.mkdir(parents=True)
            output = f"models/monitor_runs/{identifier}"
            record = {
                "id": identifier,
                "name": preset["name"],
                "preset": preset_id,
                "dataset": preset["dataset"],
                "output": output,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
            }
            env = os.environ.copy()
            env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            source = str(Path(__file__).resolve().parent.parent)
            env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
            command = [
                sys.executable,
                "-u",
                "-m",
                preset.get("module", "cs2_pipeline.train"),
                "--dataset",
                str(dataset),
                "--out",
                str(self.root / output),
                "--progress-file",
                str(directory / "progress.jsonl"),
            ]
            try:
                with (directory / "training.log").open("w", encoding="utf-8") as log:
                    child = subprocess.Popen(
                        command,
                        cwd=self.root,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    )
                self.children[identifier] = child
                record["pid"] = child.pid
            except OSError as exc:
                record.update(status="failed", error=str(exc))
                write_record(directory / "run.json", record)
                raise PipelineError(f"无法启动训练进程：{exc}") from exc
            write_record(directory / "run.json", record)
            return record

    def cancel(self, identifier: str) -> None:
        with self.lock:
            jobs = {j["id"]: j for j in self._jobs()}
            if identifier not in jobs or jobs[identifier]["status"] != "running":
                raise PipelineError("该任务没有在此监控进程中运行，不能停止。")
            child = self.children.get(identifier)
            if child is None or child.poll() is not None:
                raise PipelineError("无法定位正在运行的任务。")
            path = self.runs / identifier / "run.json"
            record = read_json(path)
            record["cancel_requested"] = True
            write_record(path, record)
            child.terminate()

    def archive_unknown(self, identifier: str) -> None:
        """Archive an unowned record explicitly; never claim its process was stopped."""
        with self.lock:
            jobs = {j["id"]: j for j in self._jobs()}
            if identifier not in jobs or jobs[identifier]["status"] != "unknown":
                raise PipelineError("只能归档监控服务重启后无法确认状态的任务。")
            path = self.runs / identifier / "run.json"
            record = read_json(path)
            record.update(status="unconfirmed", note="用户归档了无法重连的记录；未终止原进程。")
            write_record(path, record)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            jobs = self._jobs()
            presets = []
            for key, preset in PRESETS.items():
                meta = read_json(self.root / preset["dataset"] / "dataset.json")
                presets.append(
                    {
                        "id": key,
                        **preset,
                        "available": bool(meta),
                        "sample_count": meta.get("sample_count"),
                        "round_count": meta.get("round_count"),
                        "match_counts": meta.get("split_policy", {}).get("match_counts"),
                    }
                )
            history = []
            paths = list((self.root / "models").glob("*/report.json"))
            paths += list((self.root / "models/monitor_runs").glob("*/report.json"))
            paths += list((self.root / "models/evaluations").glob("*/report.json"))
            for path in sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True):
                report = read_json(path)
                is_event_holdout = path.parent.parent.name == "evaluations"
                scores = (
                    report.get("model", {})
                    if is_event_holdout
                    else report.get("test")
                    or (report.get("metrics") or {}).get("test", {}).get("model", {})
                )
                if "calibrated" in scores:
                    scores = scores["calibrated"]
                if report.get("run_mode") == "temporal_historical_validation":
                    scores = report["metrics"]["validation"]["model"]
                history.append(
                    {
                        "name": path.parent.name,
                        "path": path.relative_to(self.root).as_posix(),
                        "created_at": report.get("created_at_utc")
                        or datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
                        "scores": scores,
                        "mode": "additional_event_holdout"
                        if is_event_holdout
                        else report.get("run_mode", "frozen_event_test_calibrated"),
                        "event": report.get("event") if is_event_holdout else None,
                    }
                )
            return {
                "now": datetime.now(UTC).isoformat(),
                "jobs": jobs,
                "presets": presets,
                "stages": [{"id": key, "name": name} for key, name in STAGES],
                "history": history[:30],
                "corpus": read_json(self.root / "reports/cologne_experiment.json"),
                "holdout_batch": read_json(self.root / "runs/holdout_50/progress.json"),
                "historical_validation_batch": read_json(
                    self.root / "runs/phase2_validation_more_v1/progress.json"
                ),
                "context_batch": read_json(self.root / "runs/context_audit/progress.json"),
            }


def make_server(root: Path, port: int) -> ThreadingHTTPServer:
    monitor = TrainingMonitor(root)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def valid_host(self) -> bool:
            return self.headers.get("Host") in {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def respond(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(body)

        def json_response(self, status: int, value: Any) -> None:
            self.respond(
                status,
                json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                "application/json; charset=utf-8",
            )

        def do_GET(self) -> None:
            if not self.valid_host():
                self.json_response(403, {"error": "请通过本机回环地址访问监控界面。"})
                return
            route = urlsplit(self.path).path
            if route == "/api/state":
                self.json_response(200, monitor.snapshot())
            elif route in ("/", "/index.html", "/app.js", "/style.css"):
                name = "index.html" if route == "/" else route[1:]
                body = (STATIC / name).read_bytes().replace(b"__CS2_TOKEN__", token.encode())
                mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
                self.respond(200, body, mime + "; charset=utf-8")
            else:
                self.json_response(404, {"error": "页面不存在。"})

        def do_POST(self) -> None:
            origin = self.headers.get("Origin")
            allowed = {
                f"http://127.0.0.1:{self.server.server_port}",
                f"http://localhost:{self.server.server_port}",
            }
            if self.headers.get("X-CS2-Token") != token or (origin and origin not in allowed):
                self.json_response(403, {"error": "请求来源或会话令牌无效，请刷新本地页面。"})
                return
            if not self.valid_host():
                self.json_response(403, {"error": "请通过本机回环地址访问监控界面。"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 4096:
                    raise ValueError("请求大小不合理")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("请求必须是 JSON 对象")
                route = urlsplit(self.path).path
                if route == "/api/start":
                    self.json_response(201, monitor.start(payload.get("preset", "")))
                elif route == "/api/cancel":
                    monitor.cancel(payload.get("id", ""))
                    self.json_response(200, {"ok": True})
                elif route == "/api/archive":
                    monitor.archive_unknown(payload.get("id", ""))
                    self.json_response(200, {"ok": True})
                else:
                    self.json_response(404, {"error": "接口不存在。"})
            except PipelineError as exc:
                self.json_response(409, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                self.json_response(400, {"error": str(exc)})

    server = LoopbackHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        server = make_server(args.root, args.port)
    except (OSError, ValueError) as exc:
        print(f"监控界面启动失败：{exc}；可通过 --port 更换端口。", file=sys.stderr)
        return 1
    print(f"CS2 训练监控：http://127.0.0.1:{server.server_port}（保持此进程运行）", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

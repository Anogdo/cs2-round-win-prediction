"""Monitor behavior without opening a network socket in restricted test environments."""

from __future__ import annotations

import importlib.util
import io
import json
import re
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

from cs2_pipeline.dataset import build_dataset, write_json
from cs2_pipeline.monitor import TrainingMonitor, events_at, log_tail, make_server, write_record
from cs2_pipeline.progress import STAGES, ProgressReporter
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.train import main as train_main
from cs2_pipeline.train import train_baseline


class FakeChild:
    pid = 12345
    returncode: int | None = None
    terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15


@pytest.fixture
def manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    path = tmp_path / "data/datasets/mirage_rio_v1/dataset.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    spawned = []

    def spawn(command: list[str], **kwargs: Any) -> FakeChild:
        child = FakeChild()
        spawned.append((child, command, kwargs))
        return child

    monkeypatch.setattr("cs2_pipeline.monitor.subprocess.Popen", spawn)
    return TrainingMonitor(tmp_path), spawned


def test_monitor_starts_only_registered_datasets_and_prevents_parallel_runs(manager: tuple):
    monitor, spawned = manager
    with pytest.raises(PipelineError, match="已登记"):
        monitor.start("../../shell")
    with pytest.raises(PipelineError, match="找不到已构建"):
        monitor.start("rio_v2")
    job = monitor.start("rio_v1")
    child, command, kwargs = spawned[0]
    assert command[1:4] == ["-u", "-m", "cs2_pipeline.train"]
    assert "--progress-file" in command and "--smoke-test" not in command
    assert kwargs.get("shell", False) is False
    assert kwargs["env"]["PYTHONUNBUFFERED"] == "1"
    assert monitor.snapshot()["jobs"][0]["status"] == "running"
    with pytest.raises(PipelineError, match="已有运行"):
        monitor.start("rio_v1")
    monitor.cancel(job["id"])
    assert child.terminated
    assert monitor.snapshot()["jobs"][0]["status"] == "cancelled"
    with pytest.raises(PipelineError, match="没有在此监控"):
        monitor.cancel(job["id"])


def test_monitor_checks_report_before_marking_success_and_keeps_prior_outputs(manager: tuple):
    monitor, spawned = manager
    first = monitor.start("rio_v1")
    spawned[0][0].returncode = 0
    assert monitor.snapshot()["jobs"][0]["status"] == "failed"
    second = monitor.start("rio_v1")
    output = monitor.root / second["output"]
    output.mkdir(parents=True)
    write_json(output / "report.json", {"metrics": {}})
    spawned[1][0].returncode = 0
    jobs = {j["id"]: j for j in monitor.snapshot()["jobs"]}
    assert jobs[second["id"]]["status"] == "completed"
    assert jobs[first["id"]]["status"] == "failed"
    assert first["output"] != second["output"]


def test_restart_unknown_status_can_recover_or_be_explicitly_archived(manager: tuple):
    monitor, _ = manager
    job = monitor.start("rio_v1")
    restarted = TrainingMonitor(monitor.root)
    assert restarted.snapshot()["jobs"][0]["status"] == "unknown"
    progress = ProgressReporter(monitor.runs / job["id"] / "progress.jsonl")
    progress.emit("run", "failed", "worker failed")
    assert restarted.snapshot()["jobs"][0]["status"] == "failed"
    # A second unowned run has no terminal event: archive its record without killing a PID.
    next_job = monitor.start("rio_v1")
    again = TrainingMonitor(monitor.root)
    assert any(j["status"] == "unknown" for j in again.snapshot()["jobs"])
    again.archive_unknown(next_job["id"])
    assert any(j["status"] == "unconfirmed" for j in again.snapshot()["jobs"])
    with pytest.raises(PipelineError, match="只能归档"):
        again.archive_unknown(job["id"])


def test_progress_handles_inflight_line_and_bounds_log_reads(tmp_path: Path):
    progress = tmp_path / "events.jsonl"
    ProgressReporter(progress).emit("load", "active", "开始读取", sample_count=3)
    with progress.open("a") as handle:
        handle.write('{"incomplete":')
    events = events_at(progress)
    assert len(events) == 1 and events[0]["details"]["sample_count"] == 3
    log = tmp_path / "log.txt"
    log.write_text("x" * 40000)
    assert len(log_tail(log)) == 32768


def test_history_includes_standalone_event_holdout(manager: tuple):
    monitor, _ = manager
    report_path = monitor.root / "models/evaluations/esl24/report.json"
    report_path.parent.mkdir(parents=True)
    write_json(
        report_path,
        {"event": "esl24", "model": {"log_loss": 0.44, "brier_score": 0.15}},
    )
    report = next(h for h in monitor.snapshot()["history"] if h["name"] == "esl24")
    assert report["mode"] == "additional_event_holdout"
    assert report["event"] == "esl24"
    assert report["scores"]["log_loss"] == 0.44
    assert report["created_at"]


def test_snapshot_exposes_batch_progress_without_creating_a_training_job(manager: tuple):
    monitor, _ = manager
    progress = monitor.root / "runs/holdout_50/progress.json"
    progress.parent.mkdir(parents=True)
    write_json(progress, {"status": "running", "total": 50, "completed": [{"match_id": "x"}]})
    state = monitor.snapshot()
    assert state["holdout_batch"]["total"] == 50
    assert len(state["holdout_batch"]["completed"]) == 1
    assert not state["jobs"]


def test_comparison_preset_and_context_progress_are_visible(manager: tuple):
    monitor, spawned = manager
    dataset = monitor.root / "data/datasets/mirage_expanded_2026_v1/dataset.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}", encoding="utf-8")
    progress = monitor.root / "runs/context_audit/progress.json"
    progress.parent.mkdir(parents=True)
    write_json(progress, {"status": "running", "total": 93, "completed": 7, "failed": 0})
    job = monitor.start("phase2_comparison")
    assert spawned[0][1][1:4] == ["-u", "-m", "cs2_pipeline.phase2_compare"]
    state = monitor.snapshot()
    assert state["context_batch"]["completed"] == 7
    assert state["jobs"][0]["id"] == job["id"]


def test_gpu_preset_uses_the_cuda_comparison_module(manager: tuple):
    monitor, spawned = manager
    dataset = monitor.root / "data/datasets/mirage_expanded_2026_v1/dataset.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("{}", encoding="utf-8")
    monitor.start("phase2_gpu")
    assert spawned[0][1][1:4] == ["-u", "-m", "cs2_pipeline.phase2_gpu"]
    assert "--progress-file" in spawned[0][1]


def test_two_monitors_cannot_listen_on_same_port(tmp_path: Path):
    first = make_server(tmp_path, 0)
    try:
        with pytest.raises(OSError):
            second = make_server(tmp_path, first.server_port)
            second.server_close()
    finally:
        first.server_close()


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="需要可选 ml 依赖")
def test_progress_does_not_change_model_parameters(phase2_manifest: Path, tmp_path: Path):
    dataset = tmp_path / "dataset"
    build_dataset(phase2_manifest, dataset)
    train_baseline(dataset, tmp_path / "plain")
    progress = tmp_path / "progress.jsonl"
    train_baseline(dataset, tmp_path / "observed", progress=ProgressReporter(progress))
    plain = json.loads((tmp_path / "plain/model.json").read_text())
    observed = json.loads((tmp_path / "observed/model.json").read_text())
    assert plain["parameters"] == observed["parameters"]
    events = events_at(progress)
    assert [(e["stage"], e["state"]) for e in events if e["state"] == "completed"] == [
        *[(key, "completed") for key, _ in STAGES],
        ("run", "completed"),
    ]
    assert (
        next(e for e in events if e["stage"] == "fit" and e["state"] == "completed")["details"][
            "iterations"
        ]
        == observed["training"]["optimizer_iterations"]
    )


def test_cli_failure_emits_actionable_terminal_event(tmp_path: Path):
    events = tmp_path / "failed.jsonl"
    assert (
        train_main(
            [
                "--dataset",
                str(tmp_path / "missing"),
                "--out",
                str(tmp_path / "out"),
                "--progress-file",
                str(events),
            ]
        )
        == 1
    )
    terminal = events_at(events)[-1]
    assert terminal["state"] == "failed" and terminal["message"].startswith("训练失败：")
    assert not (tmp_path / "out").exists()


def test_http_routes_origin_token_and_path_checks_without_network(tmp_path: Path):
    # Construct handlers with byte streams, exercising real routing/headers without sockets.
    server = make_server(tmp_path, 0)
    try:

        def request(method: str, path: str, body: Any = None, **headers: str) -> tuple:
            handler = server.RequestHandlerClass.__new__(server.RequestHandlerClass)
            handler.server = server
            handler.request_version = "HTTP/1.1"
            handler.requestline = method + " " + path + " HTTP/1.1"
            handler.command, handler.path = method, path
            handler.client_address = ("127.0.0.1", 12345)
            handler.rfile, handler.wfile = io.BytesIO(), io.BytesIO()
            handler.headers = Message()
            handler.headers["Host"] = headers.pop("Host", f"127.0.0.1:{server.server_port}")
            if body is not None:
                encoded = json.dumps(body).encode()
                handler.rfile.write(encoded)
                handler.rfile.seek(0)
                handler.headers["Content-Length"] = str(len(encoded))
            for key, value in headers.items():
                handler.headers[key] = value
            getattr(handler, "do_" + method)()
            head, response_body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
            return int(head.split(b" ")[1]), response_body

        status, html = request("GET", "/")
        assert status == 200 and "训练监控".encode() in html
        token = re.search(rb'name="cs2-token" content="([^"]+)"', html)[1].decode()
        assert request("GET", "/app.js")[0] == 200
        assert request("GET", "/api/state")[0] == 200
        assert request("GET", "/../../pyproject.toml")[0] == 404
        assert request("GET", "/api/state", Host="evil.example")[0] == 403
        assert request("POST", "/api/start", {"preset": "rio_v1"})[0] == 403
        valid = {"X-CS2-Token": token}
        assert (
            request(
                "POST",
                "/api/start",
                {"preset": "rio_v1"},
                **valid,
                Origin="https://elsewhere.example",
            )[0]
            == 403
        )
        assert request("POST", "/api/start", {"preset": "unknown"}, **valid)[0] == 409
        assert request("POST", "/api/start", [], **valid)[0] == 400
        assert request("POST", "/api/unknown", {}, **valid)[0] == 404
    finally:
        server.server_close()


def test_failed_spawn_is_recorded(manager: tuple, monkeypatch: pytest.MonkeyPatch):
    monitor, _ = manager

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("test launch failure")

    monkeypatch.setattr("cs2_pipeline.monitor.subprocess.Popen", fail)
    with pytest.raises(PipelineError, match="无法启动训练"):
        monitor.start("rio_v1")
    assert monitor.snapshot()["jobs"][0]["status"] == "failed"


def test_atomic_record_roundtrip(tmp_path: Path):
    path = tmp_path / "run.json"
    write_record(path, {"status": "running"})
    write_record(path, {"status": "completed"})
    assert json.loads(path.read_text())["status"] == "completed"
    assert not path.with_suffix(".tmp").exists()

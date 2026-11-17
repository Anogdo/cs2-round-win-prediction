"""Public download integrity/access-boundary tests; all HTTP is mocked."""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest

from cs2_pipeline.acquire import acquire_corpus, download_verified, validate_source
from cs2_pipeline.dataset import write_json
from cs2_pipeline.schema import PipelineError


def source(payload: bytes = b"mock download bytes, not a CS2 recording") -> dict:
    return {
        "id": "synthetic-download",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "download_url": "https://example.test/file.dem",
        "license": "CC-BY-4.0",
        "attribution": "synthetic test",
        "event": "synthetic-event",
    }


def test_download_is_hash_verified_and_reuses_only_identical_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"mock download bytes, not a CS2 recording"
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    out = tmp_path / "bytes.bin"
    assert download_verified(source(payload), out) is True
    assert download_verified(source(payload), out) is False
    out.write_bytes(b"corrupt")
    with pytest.raises(PipelineError, match="拒绝覆盖"):
        download_verified(source(payload), out)


def test_download_mismatch_never_publishes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong bytes")
    )
    out = tmp_path / "bytes.bin"
    with pytest.raises(PipelineError, match="SHA256"):
        download_verified(source(), out)
    assert not out.exists()


def test_access_protection_stops_requests_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def blocked(*args, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError("https://example.test", 403, "Forbidden", None, None)

    monkeypatch.setattr("urllib.request.urlopen", blocked)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"sources": [source(), {**source(), "id": "second"}]})
    with pytest.raises(PipelineError, match="没有成功"):
        acquire_corpus(manifest, tmp_path / "raw", tmp_path / "processed", tmp_path / "out.json")
    assert len(calls) == 1
    report = json.loads((tmp_path / "raw/acquisition_report.json").read_text(encoding="utf-8"))
    assert not report["complete"]


def test_reserved_event_is_never_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("reserved event must be excluded before network access")

    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"sources": [{**source(), "event": "PGL Masters Bucharest 2026"}]})
    with pytest.raises(PipelineError, match="没有成功"):
        acquire_corpus(manifest, tmp_path / "raw", tmp_path / "processed", tmp_path / "out.json")
    write_json(
        manifest,
        {
            "sources": [
                {
                    **source(),
                    "match_url": "https://www.hltv.org/matches/999/example-pgl-masters-bucharest-2026",
                }
            ]
        },
    )
    with pytest.raises(PipelineError, match="没有成功"):
        acquire_corpus(manifest, tmp_path / "raw2", tmp_path / "processed2", tmp_path / "out2.json")
    with pytest.raises(PipelineError, match="不能包含路径"):
        validate_source({**source(), "id": "../escape"})

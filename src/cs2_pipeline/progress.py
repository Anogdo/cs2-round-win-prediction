"""Small append-only progress stream shared by CLI training and the local monitor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STAGES = (
    ("load", "校验数据"),
    ("fit", "拟合模型"),
    ("evaluate", "验证与评估"),
    ("save", "保存并重读"),
)


class ProgressReporter:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, stage: str, state: str, message: str, **details: Any) -> None:
        event = {
            "time": datetime.now(UTC).isoformat(),
            "stage": stage,
            "state": state,
            "message": message,
            "details": details,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        print(message, flush=True)


def emit(
    reporter: ProgressReporter | None, stage: str, state: str, message: str, **details: Any
) -> None:
    if reporter is not None:
        reporter.emit(stage, state, message, **details)

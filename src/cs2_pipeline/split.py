"""Chronological group splits; overlapping calendar periods stay together."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import date
from typing import Any

from cs2_pipeline.schema import PipelineError

PARTITIONS = ("train", "validation", "calibration", "test")


def frozen_split(
    sources: list[dict[str, Any]], plan: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Apply a declared experiment without letting new matches change old assignments."""
    assignments = plan.get("assignments", {})
    matches = {source["match_id"] for source in sources}
    if set(assignments) != matches or set(assignments.values()) != set(PARTITIONS):
        raise PipelineError(
            "固定划分必须覆盖全部比赛，且 train/validation/calibration/test 均非空。"
        )
    known: dict[str, tuple[str, str]] = {}
    bounds: dict[str, list[str]] = {p: [] for p in PARTITIONS}
    test_events = plan.get("test_events", [])
    if not isinstance(test_events, list) or not test_events:
        raise PipelineError("固定划分必须声明独立测试赛事 test_events。")
    for source in sources:
        match, day, event = (source[c] for c in ("match_id", "match_date", "event"))
        if match in known and known[match] != (day, event):
            raise PipelineError(f"同一比赛 {match} 的日期/赛事不一致。")
        known[match] = (day, event)
        if date.fromisoformat(day).isoformat() != day:
            raise PipelineError("固定划分日期必须为 YYYY-MM-DD。")
        partition = assignments[match]
        bounds[partition].append(day)
        if (event in test_events) != (partition == "test"):
            raise PipelineError("独立测试赛事必须全部且仅位于 test，不能参与训练或校准。")
    for left, right in zip(PARTITIONS, PARTITIONS[1:], strict=False):
        if max(bounds[left]) >= min(bounds[right]):
            raise PipelineError("固定集合必须按日期严格先后排列，不能跨集合拆分同一天。")
    return dict(assignments), {
        "group_by": "match",
        "evaluation_ready": True,
        "method": "frozen_four_way",
        "partitions": list(PARTITIONS),
        "plan": plan,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "match_count": len(matches),
        "match_counts": {p: sum(a == p for a in assignments.values()) for p in PARTITIONS},
    }


def chronological_split(
    sources: list[dict[str, Any]],
    *,
    group_by: str = "match",
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Do not split a match, an event group, a date, or overlapping event windows.

    Fewer than three independent chronological blocks leaves every match in
    train. Training then requires an explicit smoke-test flag and has no metrics.
    """
    if group_by not in ("match", "event"):
        raise PipelineError("group_by 只能是 match 或 event。")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise PipelineError("训练/验证比例必须在 0 和 1 之间。")
    if train_fraction + validation_fraction >= 1:
        raise PipelineError("必须为测试集保留正的比例。")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matches: dict[str, tuple[str, date]] = {}
    for source in sources:
        try:
            day = date.fromisoformat(source["match_date"])
            if day.isoformat() != source["match_date"]:
                raise ValueError("需要 YYYY-MM-DD 日期格式")
            match_id, event = source["match_id"], source["event"]
        except (ValueError, KeyError, TypeError) as exc:
            raise PipelineError("每个来源必须有 match_id、event 和 YYYY-MM-DD 比赛日期。") from exc
        if not match_id or not event:
            raise PipelineError("比赛和赛事标识不能为空。")
        if match_id in matches and matches[match_id] != (event, day):
            raise PipelineError(f"同一比赛 {match_id} 的日期/赛事声明不一致。")
        matches[match_id] = (event, day)
        groups[match_id if group_by == "match" else event].append(source)
    if not matches:
        raise PipelineError("没有可划分的比赛。")
    windows = []
    for members in groups.values():
        days = [date.fromisoformat(m["match_date"]) for m in members]
        windows.append((min(days), max(days), {m["match_id"] for m in members}))
    blocks: list[tuple[date, date, set[str]]] = []
    for start, end, ids in sorted(windows, key=lambda w: (w[0], w[1], sorted(w[2]))):
        if blocks and start <= blocks[-1][1]:
            previous = blocks.pop()
            blocks.append((previous[0], max(previous[1], end), previous[2] | ids))
        else:
            blocks.append((start, end, ids))
    ready = len(blocks) >= 3
    assignments: dict[str, str] = {}
    if ready:
        n_train = max(1, min(len(blocks) - 2, int(len(blocks) * train_fraction)))
        n_validation = max(
            1, min(len(blocks) - n_train - 1, int(len(blocks) * validation_fraction))
        )
        for index, (_, _, ids) in enumerate(blocks):
            partition = (
                "train"
                if index < n_train
                else ("validation" if index < n_train + n_validation else "test")
            )
            assignments.update(dict.fromkeys(ids, partition))
    else:
        assignments = dict.fromkeys(matches, "train")
    summary = {
        "group_by": group_by,
        "chronological_blocks": len(blocks),
        "match_count": len(matches),
        "evaluation_ready": ready,
        "reason": None if ready else "不足 3 个独立日期区间；仅可进行明确标注的流程检查。",
        "requested_fractions": {"train": train_fraction, "validation": validation_fraction},
        "match_counts": {
            s: sum(v == s for v in assignments.values()) for s in ("train", "validation", "test")
        },
    }
    return assignments, summary

"""Import the pinned 50-match evaluation batch without using it for fitting."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import polars as pl
from awpy import Demo

from cs2_pipeline.dataset import held_out_event, read_dataset, slug, write_json
from cs2_pipeline.local import check_round_sequence, import_local
from cs2_pipeline.parse import parse_demo, sha256_file, write_outputs
from cs2_pipeline.recover import recover_rounds, verify_with_awpy
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "configs/holdout_50_plan.json"
REFERENCE = ROOT / "configs/holdout_50_reference.json"
TRAINING = ROOT / "data/datasets/mirage_expanded_2026_v1"
PRIOR_ESL = ROOT / "data/datasets/esl24_holdout_v1"
RAW = ROOT / "data/raw/holdout_50"
EXTRACTED = ROOT / "data/raw/holdout_50_extracted"
PROCESSED = ROOT / "data/processed/holdout_50"
UNRAR = Path(r"C:\Program Files\WinRAR\UnRAR.exe")
PROGRESS = ROOT / "runs/holdout_50/progress.json"

TEAM_ALIASES = {
    "team falcons": "Falcons",
    "team spirit": "Spirit",
    "team vitality": "Vitality",
    "aurora gaming": "Aurora",
    "b8 esports": "B8",
    "betboom team": "BETBOOM",
    "fut esports": "FUT",
    "g2 esports": "G2",
    "g2.esports": "G2",
    "9zglobant": "9z",
    "9z globant": "9z",
    "mongolz": "The MongolZ",
    "themongolz_": "The MongolZ",
    "pvision": "PARIVISION",
}


def load_plans() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    maps = {row["match_id"]: row for row in reference["maps"]}
    ids = [row["match_id"] for row in plan["matches"]]
    if len(ids) != 50 or len(set(ids)) != 50 or len(maps) != 50 or set(ids) != set(maps):
        raise PipelineError("50 场固定计划与外部日期/比分清单不一致。")
    if version("awpy") != "3.3.0":
        raise PipelineError("此批次需要锁定的 Awpy 3.3.0。")
    model = ROOT / plan["frozen_model"]
    if sha256_file(model) != plan["frozen_model_sha256"]:
        raise PipelineError("固定模型哈希已变化，拒绝处理独立测试。")
    for item in plan["matches"]:
        ref = maps[item["match_id"]]
        if (
            held_out_event(item["event"])
            or held_out_event(item["url"])
            or ref["source_url"] != item["url"]
            or ref["map_index"] != item["map_index"]
            or ref["match_date"] <= plan["training_cutoff"]
            or sum(ref["expected_score"].values()) != ref["expected_rounds"]
        ):
            raise PipelineError(f"测试计划和公开来源不符：{item['match_id']}")
    return plan, maps


def check_disjoint(plan: dict[str, Any]) -> None:
    model = json.loads((ROOT / plan["frozen_model"]).read_text(encoding="utf-8"))
    if model["dataset_manifest_sha256"] != sha256_file(TRAINING / "dataset.json"):
        raise PipelineError("固定模型与训练数据集哈希不匹配。")
    new = {item["match_id"] for item in plan["matches"]}
    for path in (TRAINING, PRIOR_ESL):
        _, tables = read_dataset(path)
        overlap = new & set(tables["samples"]["match_id"].to_list())
        if overlap:
            raise PipelineError(f"测试比赛与旧数据重叠：{sorted(overlap)}")


def find_archive(item: dict[str, Any]) -> Path:
    prefix = f"{item['event']}-{slug(item['matchup'])}-bo"
    matches = [p for p in RAW.glob("*.rar") if p.name.lower().startswith(prefix)]
    if len(matches) > 1:
        # Same teams met twice in this event: the final is Bo5; group match is Bo3.
        series = {"hltv:2399417": "bo5", "hltv:2399387": "bo3"}.get(item["match_id"])
        matches = [p for p in matches if f"-{series}-" in p.name.lower()]
    if len(matches) != 1:
        raise PipelineError(f"无法唯一匹配原始压缩包：{item['match_id']} ({len(matches)} 个)")
    return matches[0]


def list_entries(
    archive: Path, map_index: int, *, single_map_bo1_entry: str | None = None
) -> list[str]:
    result = subprocess.run(
        [str(UNRAR), "lb", "-p-", "--", str(archive)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise PipelineError(f"无法读取压缩包 {archive.name}：{result.stderr or result.stdout}")
    pattern = re.compile(rf"-m{map_index}-mirage(?:-p(\d+))?\.dem$", re.I)
    listed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if single_map_bo1_entry is not None and (
        map_index != 1
        or listed != [single_map_bo1_entry]
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*-mirage\.dem", single_map_bo1_entry)
    ):
        raise PipelineError(f"单地图 Bo1 压缩包成员与固定清单不符：{archive.name}")
    found = []
    for entry in listed:
        match = pattern.search(entry)
        if match or single_map_bo1_entry == entry:
            if Path(entry).name != entry or "/" in entry or "\\" in entry or entry.startswith("-"):
                raise PipelineError("压缩包成员必须是无路径的普通文件名。")
            found.append((int(match.group(1)) if match and match.group(1) else 0, entry))
    found.sort()
    indices = [number for number, _ in found]
    if not (indices == [0] or (indices and indices == list(range(1, len(indices) + 1)))):
        raise PipelineError(f"Mirage 地图分段缺失或混合：{archive.name}: {indices}")
    return [entry for _, entry in found]


def extract(
    archive: Path, item: dict[str, Any], *, extracted_root: Path = EXTRACTED
) -> tuple[str, list[dict[str, Any]]]:
    archive_sha = sha256_file(archive)
    if item.get("archive_sha256", archive_sha) != archive_sha:
        raise PipelineError(f"压缩包与固定 SHA256 不符：{archive.name}")
    entries = list_entries(
        archive, item["map_index"], single_map_bo1_entry=item.get("single_map_bo1_entry")
    )
    folder = extracted_root / item["match_id"].replace(":", "-")
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "extraction.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding="utf-8"))
        if (
            prior["archive_sha256"] != archive_sha
            or [p["entry"] for p in prior["parts"]] != entries
        ):
            raise PipelineError(f"已有解压记录与压缩包不符：{archive.name}")
        for part in prior["parts"]:
            path = folder / part["entry"]
            if not path.is_file() or sha256_file(path) != part["sha256"]:
                raise PipelineError(f"已有 Demo 与解压记录不符：{path}")
        return archive_sha, prior["parts"]
    if any((folder / entry).exists() for entry in entries):
        raise PipelineError(f"解压目录存在没有校验记录的文件，请人工核对：{folder}")
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=folder) as temp:
        stage = Path(temp)
        parts = []
        for entry in entries:
            result = subprocess.run(
                [
                    str(UNRAR),
                    "e",
                    "-p-",
                    "-o-",
                    "-idq",
                    "--",
                    str(archive),
                    entry,
                    str(stage) + os.sep,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            file = stage / entry
            if result.returncode or not file.is_file() or file.stat().st_size == 0:
                raise PipelineError(
                    f"解压 {archive.name}/{entry} 失败：{result.stderr or result.stdout}"
                )
            parts.append(
                {"entry": entry, "sha256": sha256_file(file), "size_bytes": file.stat().st_size}
            )
        for part in parts:
            (stage / part["entry"]).rename(folder / part["entry"])
        write_json(receipt, {"archive_sha256": archive_sha, "parts": parts})
    return archive_sha, parts


def published_score(paths: list[Path], expected: dict[str, int]) -> dict[str, int]:
    result = dict.fromkeys(expected, 0)
    folded = {key.casefold(): key for key in expected}
    if len(folded) != len(expected) or len(expected) != 2:
        raise PipelineError("公开比分队名不唯一。")
    for path in paths:
        rounds = pl.read_parquet(path / "rounds.parquet")
        states = pl.read_parquet(path / "snapshots.parquet")
        wins = (
            states.sort("tick")
            .group_by("round_id", "team")
            .first()
            .join(rounds.select("round_id", "winner_side"), on="round_id")
            .filter(pl.col("side") == pl.col("winner_side"))
            .group_by("team")
            .len()
        )
        for row in wins.iter_rows(named=True):
            raw_name = row["team"]
            if not isinstance(raw_name, str):
                raise PipelineError("队名缺失，无法独立核对公开比分。")
            normalized = TEAM_ALIASES.get(raw_name.casefold(), raw_name)
            name = folded.get(normalized.casefold())
            if name is None:
                raise PipelineError(f"未知队名别名：{raw_name}；预期 {list(expected)}")
            result[name] += row["len"]
    if result != expected:
        raise PipelineError(f"解析比分 {result} 与公开比分 {expected} 不一致。")
    return result


def recover_multipart(
    item: dict[str, Any],
    ref: dict[str, Any],
    archive: Path,
    archive_sha: str,
    extracted: list[dict[str, Any]],
    paths: list[Path],
    *,
    processed_root: Path = PROCESSED,
) -> dict[str, Any]:
    """Recover an initial incomplete recording only after matching Awpy on its companion."""
    if len(paths) != 2:
        raise PipelineError("严格恢复目前只支持两个录制分段。")
    native_parts = []
    headers = []
    for path in paths:
        demo = Demo(path)
        demo.load("rounds")
        native_parts.append(demo.rounds)
        headers.append(
            {key: demo.header.get(key) for key in ("map_name", "build_num", "server_name")}
        )
    if not native_parts[0].is_empty() or native_parts[1].is_empty():
        raise PipelineError("分段情况与可恢复的首段缺失回合模式不同。")
    if headers[0] != headers[1] or headers[0]["map_name"] != "de_mirage":
        raise PipelineError("两个录制分段的地图、服务器或游戏版本不一致。")
    recovered_first = recover_rounds(paths[0])
    recovered_second = recover_rounds(paths[1])
    verify_with_awpy(paths[1], recovered_second)
    sequences = [recovered_first["round_num"].to_list(), native_parts[1]["round_num"].to_list()]
    check_round_sequence(sequences, ref["expected_rounds"])
    print(
        f"规则状态恢复：前段 {len(sequences[0])} 回合；后段 {len(sequences[1])} 回合"
        "与 Awpy 的边界、胜方和原因逐项一致。",
        flush=True,
    )
    out = processed_root / item["match_id"].replace(":", "-")
    maps = out / "maps"
    maps.mkdir(parents=True, exist_ok=True)
    manifest = out / "sources.json"
    entries: list[dict[str, Any]] = []
    recovery = {
        "method": "demoparser2 game-rules transitions with Awpy freeze-end events",
        "reason": "Awpy 3.3.0 returned zero rounds for the first recording",
        "demoparser2_version": version("demoparser2"),
        "companion_demo_sha256": extracted[1]["sha256"],
        "companion_awpy_rounds_matched": len(sequences[1]),
        "status_mapping": {"2": "terrorist", "3": "counter-terrorist"},
    }
    for index, (path, info) in enumerate(zip(paths, extracted, strict=True), start=1):
        identifier = f"{item['match_id'].replace(':', '-')}-m{item['map_index']}-p{index}"
        source = {
            "match_id": item["match_id"],
            "match_date": ref["match_date"],
            "event": item["event"],
            "match_url": item["url"],
            "map": "de_mirage",
            "sha256": info["sha256"],
            "local_filename": path.name,
            "provenance": "User downloaded from HLTV; date/map/score checked on match page.",
            "map_id": f"{item['match_id']}:m{item['map_index']}",
            "part_index": index,
            "part_count": len(paths),
            "tick_scope": "Original local recording tick; never concatenate ticks across parts",
            "archive_filename": archive.name,
            "archive_sha256": archive_sha,
        }
        source_file = maps / f"{identifier}.source.json"
        if source_file.exists() and json.loads(source_file.read_text(encoding="utf-8")) != source:
            raise PipelineError(f"已有分段来源不一致：{source_file}")
        if not source_file.exists():
            write_json(source_file, source)
        output = maps / identifier
        if output.exists():
            meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
            if (
                meta.get("demo_sha256") != info["sha256"]
                or meta.get("external_source") != source
                or (index == 1 and meta.get("rounds_recovery") != recovery)
            ):
                raise PipelineError(f"已有分段结果不一致：{output}")
            quality = validate_directory(output)
        else:
            quality = parse_demo(
                path,
                output,
                interval=5,
                source=source_file,
                rounds_override=recovered_first if index == 1 else None,
                rounds_recovery=recovery if index == 1 else None,
            )
        if not quality["passed"]:
            raise PipelineError(f"恢复的分段质量检查失败：{quality['errors']}")
        if (
            pl.read_parquet(output / "rounds.parquet")["round_num"].to_list()
            != sequences[index - 1]
        ):
            raise PipelineError("恢复分段的原始回合编号发生变化。")
        entries.append(
            {
                "path": Path(os.path.relpath(output, manifest.parent)).as_posix(),
                "match_id": item["match_id"],
                "match_date": ref["match_date"],
                "event": item["event"],
                "map_id": source["map_id"],
                "part_index": str(index),
            }
        )
    source_manifest = {"schema_version": 1, "map_name": "de_mirage", "sources": entries}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != source_manifest:
        raise PipelineError("已有分段来源清单与恢复结果不同。")
    if not manifest.exists():
        write_json(manifest, source_manifest)
    outputs = [(manifest.parent / entry["path"]).resolve() for entry in entries]
    score = published_score(outputs, ref["expected_score"])
    result = {
        "match_id": item["match_id"],
        "event": item["event"],
        "match_date": ref["match_date"],
        "archive_sha256": archive_sha,
        "demo_sha256": [part["sha256"] for part in extracted],
        "rounds": ref["expected_rounds"],
        "published_score_verified": score,
        "part_count": len(paths),
        "rounds_recovery": recovery,
        "source_manifest": manifest.relative_to(ROOT).as_posix(),
    }
    write_json(out / "checked.json", result)
    return result


def repair_observed_team_switch(
    item: dict[str, Any], ref: dict[str, Any], archive_sha: str, extracted: list[dict[str, Any]]
) -> dict[str, Any]:
    """Quarantine four corrupt-side ticks, using HLTV to correct one roster identity."""
    if item["match_id"] != "hltv:2398743" or len(extracted) != 1:
        raise PipelineError("此质量修正只适用于已核实的单场比赛。")
    out = PROCESSED / "hltv-2398743"
    identifier = "hltv-2398743-m2-p1"
    original = out / "maps" / identifier
    metadata = json.loads((original / "metadata.json").read_text(encoding="utf-8"))
    quality = validate_directory(original)
    if quality["errors"] != [
        {
            "code": "player_count",
            "count": 4,
            "message": "单个采样 tick 超过职业 5v5 人数；检查重复玩家/观战者过滤。",
        }
    ]:
        raise PipelineError("原始质量异常与已核实的四个异常时点不同。")
    rounds = pl.read_parquet(original / "rounds.parquet")
    players = pl.read_parquet(original / "players.parquet")
    snapshots = pl.read_parquet(original / "snapshots.parquet")
    player_id = 76561198164970560
    player = players.filter(pl.col("steamid") == player_id)
    if player.height != 1 or player["name"][0] != "yuurih" or player["team"][0] != "Aurora Gaming":
        raise PipelineError("异常选手身份与已核实记录不符。")
    bad = (
        snapshots.group_by("tick", "side").len().filter(pl.col("len") > 5)["tick"].sort().to_list()
    )
    switched = snapshots.filter((pl.col("steamid") == player_id) & pl.col("tick").is_in(bad))
    if (
        len(bad) != 4
        or switched.height != 4
        or set(switched["round_num"].to_list()) != {22}
        or set(switched["side"].to_list()) != {"T"}
        or not switched["death_observed"].all()
        or switched["is_alive"].any()
    ):
        raise PipelineError("异常换边的 tick、身份或死亡记录不符，拒绝自动修正。")
    # HLTV's match lineup identifies yuurih as FURIA. His final observed clan
    # name comes from an anomalous mid-round team switch; never infer side here.
    players = players.with_columns(
        pl.when(pl.col("steamid") == player_id)
        .then(pl.lit("FURIA"))
        .otherwise(pl.col("team"))
        .alias("team"),
        pl.when(pl.col("steamid") == player_id)
        .then(pl.lit(["FURIA"], dtype=pl.List(pl.String)))
        .otherwise(pl.col("team_names"))
        .alias("team_names"),
    )
    snapshots = snapshots.filter(~pl.col("tick").is_in(bad)).with_columns(
        pl.when(pl.col("steamid") == player_id)
        .then(pl.lit("FURIA"))
        .otherwise(pl.col("team"))
        .alias("team")
    )
    metadata["quality_adjustment"] = {
        "excluded_ticks": bad,
        "excluded_snapshot_rows": 40,
        "reason": (
            "Four 6v4 ticks following an observed mid-round team switch; no side was inferred."
        ),
        "corrected_player_steamid": str(player_id),
        "recorded_clan_name": "Aurora Gaming",
        "verified_team": "FURIA",
        "lineup_source": item["url"],
        "original_directory": original.relative_to(ROOT).as_posix(),
    }
    adjusted = out / "maps_quality_checked" / identifier
    if adjusted.exists():
        saved = json.loads((adjusted / "metadata.json").read_text(encoding="utf-8"))
        if saved.get("quality_adjustment") != metadata["quality_adjustment"]:
            raise PipelineError("已有质量修正与核对结果不同。")
        checked_quality = validate_directory(adjusted)
    else:
        checked_quality = write_outputs(adjusted, metadata, rounds, players, snapshots)
    if not checked_quality["passed"]:
        raise PipelineError(f"隔离异常时点后质量检查仍失败：{checked_quality['errors']}")
    source = {
        "path": Path(os.path.relpath(adjusted, out)).as_posix(),
        "match_id": item["match_id"],
        "match_date": ref["match_date"],
        "event": item["event"],
        "map_id": f"{item['match_id']}:m{item['map_index']}",
        "part_index": "1",
    }
    manifest = out / "sources.json"
    source_manifest = {"schema_version": 1, "map_name": "de_mirage", "sources": [source]}
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != source_manifest:
        raise PipelineError("质量修正的来源清单与已有文件冲突。")
    if not manifest.exists():
        write_json(manifest, source_manifest)
    score = published_score([adjusted], ref["expected_score"])
    result = {
        "match_id": item["match_id"],
        "event": item["event"],
        "match_date": ref["match_date"],
        "archive_sha256": archive_sha,
        "demo_sha256": [part["sha256"] for part in extracted],
        "rounds": ref["expected_rounds"],
        "published_score_verified": score,
        "part_count": 1,
        "quality_adjustment": metadata["quality_adjustment"],
        "source_manifest": manifest.relative_to(ROOT).as_posix(),
    }
    write_json(out / "checked.json", result)
    return result


def import_one(item: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    archive = find_archive(item)
    return import_archive(archive, item, ref)


def import_archive(
    archive: Path,
    item: dict[str, Any],
    ref: dict[str, Any],
    *,
    extracted_root: Path = EXTRACTED,
    processed_root: Path = PROCESSED,
) -> dict[str, Any]:
    archive_sha, extracted = extract(archive, item, extracted_root=extracted_root)
    raw_dir = extracted_root / item["match_id"].replace(":", "-")
    paths = [raw_dir / part["entry"] for part in extracted]
    try:
        native = Demo(paths if len(paths) > 1 else paths[0])
    except ValueError as exc:
        if len(paths) == 2 and "part 0 has no completed match rounds" in str(exc):
            return recover_multipart(
                item, ref, archive, archive_sha, extracted, paths, processed_root=processed_root
            )
        raise
    native.load("rounds")
    native_rounds = native.rounds["round_num"].to_list()
    if native.header.get("map_name") != "de_mirage" or native_rounds != list(
        range(1, ref["expected_rounds"] + 1)
    ):
        raise PipelineError(f"Awpy 整图回合或地图不符：{item['match_id']}: {native_rounds}")
    del native
    round_lists = []
    if len(paths) == 1:
        round_lists = [native_rounds]
    else:
        for path in paths:
            part = Demo(path)
            part.load("rounds")
            round_lists.append(part.rounds["round_num"].to_list())
            del part
        check_round_sequence(round_lists, ref["expected_rounds"])
    out = processed_root / item["match_id"].replace(":", "-")
    out.mkdir(parents=True, exist_ok=True)
    local_manifest = out / "local.json"
    local = {
        "schema_version": 1,
        "map_name": "de_mirage",
        "maps": [
            {
                "match_id": item["match_id"],
                "match_date": ref["match_date"],
                "event": item["event"],
                "match_url": item["url"],
                "map_index": item["map_index"],
                "expected_rounds": ref["expected_rounds"],
                "archive_filename": archive.name,
                "archive_sha256": archive_sha,
                "provenance": "User downloaded from HLTV; date/map/score checked on match page.",
                "parts": [
                    {
                        "path": Path(os.path.relpath(path, out)).as_posix(),
                        "sha256": info["sha256"],
                        "size_bytes": info["size_bytes"],
                        "round_numbers": numbers,
                    }
                    for path, info, numbers in zip(paths, extracted, round_lists, strict=True)
                ],
            }
        ],
    }
    if local_manifest.exists() and json.loads(local_manifest.read_text(encoding="utf-8")) != local:
        raise PipelineError(f"已有本地清单不一致：{local_manifest}")
    if not local_manifest.exists():
        write_json(local_manifest, local)
    manifest = out / "sources.json"
    try:
        report = import_local(local_manifest, out / "maps", manifest)
    except PipelineError:
        if item["match_id"] == "hltv:2398743":
            return repair_observed_team_switch(item, ref, archive_sha, extracted)
        raise
    if not report["complete"]:
        raise PipelineError(f"地图解析未完成：{item['match_id']}")
    sources = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    outputs = [(manifest.parent / source["path"]).resolve() for source in sources]
    if not all(validate_directory(path)["passed"] for path in outputs):
        raise PipelineError(f"数据重新验证失败：{item['match_id']}")
    score = published_score(outputs, ref["expected_score"])
    result = {
        "match_id": item["match_id"],
        "event": item["event"],
        "match_date": ref["match_date"],
        "archive_sha256": archive_sha,
        "demo_sha256": [part["sha256"] for part in extracted],
        "rounds": ref["expected_rounds"],
        "published_score_verified": score,
        "part_count": len(paths),
        "source_manifest": str(manifest.relative_to(ROOT).as_posix()),
    }
    write_json(out / "checked.json", result)
    return result


def write_progress(result: dict[str, Any]) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    temporary = PROGRESS.with_name(f"progress-{os.getpid()}.tmp")
    result["updated_at_utc"] = datetime.now(UTC).isoformat()
    write_json(temporary, result)
    # On Windows, a simultaneous UI read can briefly lock the old destination.
    for attempt in range(100):
        try:
            temporary.replace(PROGRESS)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(0.05)


def run_batch(
    plan: dict[str, Any], reference: dict[str, dict[str, Any]], *, resume: bool = False
) -> int:
    check_disjoint(plan)
    if not UNRAR.is_file():
        raise PipelineError("找不到 UnRAR；请检查 WinRAR 安装。")
    # Refuse accidental reuse of one source archive for two distinct matches.
    archive_names = [find_archive(item).name for item in plan["matches"]]
    if len(set(archive_names)) != len(archive_names):
        raise PipelineError("存在两场比赛指向同一个压缩包。")
    result: dict[str, Any] = {
        "status": "running",
        "total": len(plan["matches"]),
        "completed": [],
        "failed": [],
        "current": None,
        "plan_sha256": sha256_file(PLAN),
        "reference_sha256": sha256_file(REFERENCE),
    }
    done: set[str] = set()
    if resume:
        previous = json.loads(PROGRESS.read_text(encoding="utf-8"))
        if (
            previous.get("plan_sha256") != result["plan_sha256"]
            or previous.get("reference_sha256") != result["reference_sha256"]
        ):
            raise PipelineError("断点续传时计划或比分来源已变化，拒绝复用结果。")
        for item in plan["matches"]:
            checked = PROCESSED / item["match_id"].replace(":", "-") / "checked.json"
            if not checked.is_file():
                continue
            saved = json.loads(checked.read_text(encoding="utf-8"))
            ref = reference[item["match_id"]]
            if (
                saved.get("match_id") != item["match_id"]
                or saved.get("event") != item["event"]
                or saved.get("match_date") != ref["match_date"]
                or saved.get("rounds") != ref["expected_rounds"]
                or saved.get("published_score_verified") != ref["expected_score"]
            ):
                raise PipelineError(f"已完成记录与计划不符：{checked}")
            done.add(item["match_id"])
            result["completed"].append(saved)
        print(f"断点续传：复用 {len(done)} 场已核对的比赛。", flush=True)
    for index, item in enumerate(plan["matches"], start=1):
        if item["match_id"] in done:
            continue
        result["current"] = {"index": index, "id": item["match_id"], "matchup": item["matchup"]}
        write_progress(result)
        print(f"[{index}/{result['total']}] {item['matchup']} ({item['match_id']})", flush=True)
        # Each recording uses a fresh process so native parser memory is released between maps.
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", item["match_id"]]
        proc = subprocess.run(command, cwd=ROOT, check=False)
        checked = PROCESSED / item["match_id"].replace(":", "-") / "checked.json"
        if proc.returncode == 0 and checked.is_file():
            result["completed"].append(json.loads(checked.read_text(encoding="utf-8")))
        else:
            result["failed"].append({"match_id": item["match_id"], "exit_code": proc.returncode})
        write_progress(result)
    result["current"] = None
    result["status"] = "completed" if not result["failed"] else "completed_with_errors"
    write_progress(result)
    print(f"处理完成：{len(result['completed'])} 成功，{len(result['failed'])} 失败。", flush=True)
    return 0 if not result["failed"] else 2


def retry_failed(plan: dict[str, Any]) -> int:
    check_disjoint(plan)
    result = json.loads(PROGRESS.read_text(encoding="utf-8"))
    if (
        result.get("status") != "completed_with_errors"
        or result.get("plan_sha256") != sha256_file(PLAN)
        or result.get("reference_sha256") != sha256_file(REFERENCE)
    ):
        raise PipelineError("只可重试已结束且计划未变化的批次。")
    items = {item["match_id"]: item for item in plan["matches"]}
    errors = result["failed"]
    result["status"] = "running"
    result["failed"] = []
    for index, previous in enumerate(errors, start=1):
        item = items[previous["match_id"]]
        result["current"] = {"index": index, "id": item["match_id"], "matchup": item["matchup"]}
        write_progress(result)
        print(f"重试 {index}/{len(errors)}：{item['matchup']} ({item['match_id']})", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()), "worker", item["match_id"]],
            cwd=ROOT,
            check=False,
        )
        checked = PROCESSED / item["match_id"].replace(":", "-") / "checked.json"
        if proc.returncode == 0 and checked.is_file():
            result["completed"].append(json.loads(checked.read_text(encoding="utf-8")))
        else:
            result["failed"].append({"match_id": item["match_id"], "exit_code": proc.returncode})
        write_progress(result)
    result["current"] = None
    result["status"] = "completed" if not result["failed"] else "completed_with_errors"
    write_progress(result)
    print(f"重试结束：{len(result['completed'])} 成功，{len(result['failed'])} 失败。", flush=True)
    return 0 if not result["failed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "resume", "worker", "retry"])
    parser.add_argument("match_id", nargs="?")
    args = parser.parse_args()
    try:
        plan, reference = load_plans()
        if args.action in ("run", "resume"):
            return run_batch(plan, reference, resume=args.action == "resume")
        if args.action == "retry":
            return retry_failed(plan)
        matches = [m for m in plan["matches"] if m["match_id"] == args.match_id]
        if len(matches) != 1:
            raise PipelineError("worker 需要清单中的唯一比赛 ID。")
        result = import_one(matches[0], reference[args.match_id])
        print(f"已验证 {result['match_id']}，{result['rounds']} 回合。", flush=True)
        return 0
    except Exception as exc:
        print(f"批次导入失败（{type(exc).__name__}）：{exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict recovery of completed rounds when Awpy cannot read an initial part.

Player snapshots and kills remain Awpy outputs. Raw game-rules state is used only
for round boundaries and labels, and must match Awpy on the companion recording.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cs2_pipeline.schema import PipelineError

RULE_FIELDS = [
    "total_rounds_played",
    "game_phase",
    "is_warmup_period",
    "is_freeze_period",
    "round_win_status",
    "round_win_reason",
]
ROUND_COLUMNS = ["round_num", "start_tick", "freeze_end_tick", "end_tick", "winner_side", "reason"]


def reconstruct_rounds(states: pl.DataFrame, freeze_ticks: list[int]) -> pl.DataFrame:
    """Use actual game-rules transitions, refusing ambiguous or missing boundaries."""
    if not freeze_ticks or len(freeze_ticks) != len(set(freeze_ticks)):
        raise PipelineError("分段中不存在唯一的回合冻结结束事件。")
    if states.is_empty() or any(column not in states.columns for column in ["tick", *RULE_FIELDS]):
        raise PipelineError("回合恢复缺少游戏状态字段。")
    states = states.select("tick", *RULE_FIELDS).unique().sort("tick")
    if states["tick"].n_unique() != states.height:
        raise PipelineError("同一 tick 的游戏规则状态存在冲突。")
    if states.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        raise PipelineError("游戏规则状态包含 null；不能推断回合。")
    initial = states.row(0, named=True)
    if initial["is_warmup_period"] or not initial["is_freeze_period"]:
        raise PipelineError("录制首个状态不是比赛冻结期，无法可靠恢复首回合。")
    freezes = sorted(freeze_ticks)
    output: list[dict] = []
    for index, tick in enumerate(freezes):
        matching = states.filter(pl.col("tick") == tick)
        if matching.height != 1:
            raise PipelineError(f"冻结结束 tick {tick} 缺少唯一的规则状态。")
        state = matching.row(0, named=True)
        number = int(state["total_rounds_played"]) + 1
        if (
            state["is_warmup_period"]
            or state["is_freeze_period"]
            or state["round_win_status"] != 0
            or state["game_phase"] not in (2, 3)
            or (output and number != output[-1]["round_num"] + 1)
        ):
            raise PipelineError(f"第 {number} 回合冻结状态或编号不连续。")
        boundary = freezes[index + 1] if index + 1 < len(freezes) else states["tick"].max() + 1
        winners = states.filter(
            (pl.col("tick") > tick)
            & (pl.col("tick") < boundary)
            & (pl.col("total_rounds_played") == number)
            & pl.col("round_win_status").is_in([2, 3])
        )
        if winners.is_empty():
            raise PipelineError(f"第 {number} 回合缺少明确的结束状态和胜方。")
        winner = winners.row(0, named=True)
        if output:
            starts = states.filter(
                (pl.col("tick") > output[-1]["end_tick"])
                & (pl.col("tick") < tick)
                & (pl.col("total_rounds_played") == number - 1)
                & pl.col("is_freeze_period")
                & (pl.col("round_win_status") == 0)
            )
            if starts.is_empty():
                raise PipelineError(f"第 {number} 回合缺少冻结开始状态。")
            start = starts["tick"].min()
        else:
            start = states["tick"].min()
        output.append(
            {
                "round_num": number,
                "start_tick": start,
                "freeze_end_tick": tick,
                "end_tick": winner["tick"],
                "winner_side": "terrorist"
                if winner["round_win_status"] == 2
                else "counter-terrorist",
                "reason": winner["round_win_reason"],
                "is_knife_round": False,
            }
        )
    return pl.DataFrame(output)


def recover_rounds(path: Path) -> pl.DataFrame:
    """Read game-rules netprops and freeze events from a real local recording."""
    from awpy import Demo
    from demoparser2 import DemoParser

    demo = Demo(path)
    freezes = demo.events["round_freeze_end"]["tick"].to_list()
    parser = DemoParser(str(path))
    frame = parser.parse_ticks(RULE_FIELDS)
    # Game-rules values are common to every player; drop the IDs before conversion.
    frame = frame[["tick", *RULE_FIELDS]].drop_duplicates()
    return reconstruct_rounds(pl.from_pandas(frame), freezes)


def verify_with_awpy(path: Path, recovered: pl.DataFrame) -> None:
    """The same method must reproduce every available Awpy companion round."""
    from awpy import Demo

    demo = Demo(path)
    demo.load("rounds")
    awpy_rounds = demo.rounds.select(ROUND_COLUMNS).sort("round_num")
    if awpy_rounds.is_empty() or not awpy_rounds.equals(
        recovered.select(ROUND_COLUMNS).sort("round_num"), null_equal=True
    ):
        raise PipelineError("恢复方法与同一比赛的 Awpy 分段回合边界或胜方不一致。")

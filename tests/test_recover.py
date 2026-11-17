"""Round recovery must reject ambiguity rather than fabricate boundaries."""

import polars as pl
import pytest

from cs2_pipeline.recover import reconstruct_rounds
from cs2_pipeline.schema import PipelineError


def test_game_rules_recovery_preserves_observed_boundaries_and_winners() -> None:
    states = pl.DataFrame(
        {
            "tick": [1, 10, 30, 35, 50, 80],
            "total_rounds_played": [0, 0, 1, 1, 1, 2],
            "game_phase": [2] * 6,
            "is_warmup_period": [False] * 6,
            "is_freeze_period": [True, False, False, True, False, False],
            "round_win_status": [0, 0, 3, 0, 0, 2],
            "round_win_reason": [0, 0, 8, 0, 0, 1],
        }
    )
    rounds = reconstruct_rounds(states, [10, 50]).to_dicts()
    assert [
        (r["round_num"], r["start_tick"], r["freeze_end_tick"], r["end_tick"]) for r in rounds
    ] == [
        (1, 1, 10, 30),
        (2, 35, 50, 80),
    ]
    assert [r["winner_side"] for r in rounds] == ["counter-terrorist", "terrorist"]

    with pytest.raises(PipelineError, match="唯一"):
        reconstruct_rounds(states, [10, 10])
    with pytest.raises(PipelineError, match="缺少明确"):
        reconstruct_rounds(states.filter(pl.col("tick") != 80), [10, 50])

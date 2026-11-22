"""Causal planted state and full-coverage checks for the one-feature ablation."""

from __future__ import annotations

import importlib.util

import polars as pl
import pytest

from cs2_pipeline.context import verify_event_only_coverage
from cs2_pipeline.schema import PipelineError

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="requires ml extra"
)


def test_bomb_feature_only_observes_same_round_transitions_at_or_before_sample_tick():
    from cs2_pipeline.phase2_bomb import bomb_feature

    samples = pl.DataFrame(
        {
            "sample_id": ["earlier", "exact", "armed", "defused", "other-round"],
            "round_id": [1, 1, 1, 1, 2],
            "tick": [99, 100, 110, 120, 110],
        }
    )
    events = pl.DataFrame(
        {
            "round_id": [1, 1, 2, 2],
            "tick": [120, 100, 111, 105],
            "event": ["defuse", "finish_plant", "finish_plant", "begin_plant"],
        }
    )
    states = dict(bomb_feature(samples, events).iter_rows())
    assert states == {
        "earlier": 0.0,
        "exact": 1.0,
        "armed": 1.0,
        "defused": 0.0,
        "other-round": 0.0,
    }


def test_incomplete_bomb_coverage_cannot_be_used_to_guarantee_known_false_states():
    rounds = pl.DataFrame(
        {
            "round_id": [1, 2, 3],
            "is_valid_round": [True, True, False],
            "reason_name": ["bomb_defused", "target_bombed", "target_bombed"],
        }
    )
    events = pl.DataFrame({"round_id": [1, 2], "event": ["finish_plant", "defuse"]})
    with pytest.raises(PipelineError):
        verify_event_only_coverage(rounds, events)
    verify_event_only_coverage(rounds, events.with_columns(pl.lit("finish_plant").alias("event")))


def test_paired_scores_keep_round_weights_and_compared_predictions_on_same_targets():
    from cs2_pipeline.phase2_bomb import evaluate

    frame = pl.DataFrame(
        {
            "match_id": ["one", "one", "two"],
            "match_date": ["2026-07-31", "2026-07-31", "2026-08-01"],
            "event": "event",
            "demo_sha256": ["alpha", "alpha", "beta"],
            "round_id": [1, 1, 1],
            "y_ct_win": [1, 1, 0],
            "bomb_planted": [0.0, 1.0, 0.0],
            "v1": [0.5, 0.5, 0.5],
            "v1_bomb": [0.5, 0.75, 0.5],
        }
    )
    result = evaluate(frame, {"draws": 50, "seed": 42})
    assert result["metrics"]["v1"]["round_weight_sum"] == 2
    assert result["v1_bomb_minus_v1"]["brier_score"] == pytest.approx(-0.046875)
    assert "log_loss" in result["paired_ci95"]

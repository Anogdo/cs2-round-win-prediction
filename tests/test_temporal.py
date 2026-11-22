"""Causality, sample alignment, identical architectures, and training-only transforms."""

from __future__ import annotations

import importlib.util

import numpy as np
import polars as pl
import pytest

from cs2_pipeline.features import feature_columns
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.temporal_data import ELAPSED, join_samples, validate_timeline, window_indices


def test_shuffled_windows_do_not_cross_round_or_recording_boundaries():
    frame = pl.DataFrame(
        {
            "match_id": ["one"] * 6 + ["two"],
            "map_id": ["main", "main", "main", "main", "main", "other", "main"],
            "demo_sha256": ["first", "first", "first", "second", "first", "first", "first"],
            "round_id": [0, 1, 0, 0, 0, 0, 0],
            "tick": [300, 100, 100, 100, 200, 250, 250],
        }
    )
    rows = window_indices(frame, 3)
    assert rows.tolist() == [
        [2, 4, 0],
        [-1, -1, 1],
        [-1, -1, 2],
        [-1, -1, 3],
        [-1, 2, 4],
        [-1, -1, 5],
        [-1, -1, 6],
    ]
    assert window_indices(frame, 2)[0].tolist() == [4, 0]
    assert window_indices(frame, 1).ravel().tolist() == list(range(frame.height))
    with pytest.raises(PipelineError, match="Duplicate tick"):
        window_indices(pl.concat([frame, frame.head(1)]), 3)


@pytest.fixture
def round_and_samples() -> tuple[pl.DataFrame, pl.DataFrame]:
    identity = {
        "match_id": "match",
        "map_id": "map",
        "demo_sha256": "first",
        "round_id": 0,
        "part_index": "1",
        "event": "event",
        "match_date": "2026-04-15",
    }
    sample_ids = ["first:0:100", "first:0:164"]
    tables = {
        "samples": pl.DataFrame(
            {
                "sample_id": sample_ids,
                "map_name": "de_mirage",
                "round_num": 1,
                "tick": [100, 164],
                ELAPSED: [0.0, 1.0],
                **identity,
            }
        ),
        "features": pl.DataFrame({"sample_id": sample_ids[::-1], ELAPSED: [1.0, 0.0]}),
        "labels": pl.DataFrame({"sample_id": sample_ids[::-1], "y_ct_win": [1, 1]}),
    }
    rounds = pl.DataFrame(
        [
            {
                **identity,
                "recorded_round_num": 1,
                "freeze_end_tick": 100,
                "end_tick": 165,
                "tick_rate": 64.0,
                "is_valid_round": True,
                "winner_side": "CT",
            }
        ]
    )
    return join_samples(tables), rounds


def test_join_keeps_target_sample_order_and_separates_round_audit_fields(round_and_samples):
    samples, rounds = round_and_samples
    validate_timeline(samples, rounds)
    assert samples["sample_id"].to_list() == ["first:0:100", "first:0:164"]
    assert samples[ELAPSED].to_list() == [0.0, 1.0]
    assert "winner_side" not in samples and "end_tick" not in samples


@pytest.mark.parametrize(
    "change",
    [
        [
            pl.lit(99).alias("tick"),
            pl.lit("first:0:99").alias("sample_id"),
            pl.lit(-1 / 64).alias("elapsed_seconds"),
            pl.lit(-1 / 64).alias(ELAPSED),
        ],
        [
            pl.lit(165).alias("tick"),
            pl.lit("first:0:165").alias("sample_id"),
            pl.lit(65 / 64).alias("elapsed_seconds"),
            pl.lit(65 / 64).alias(ELAPSED),
        ],
        [pl.lit("2").alias("part_index")],
        [pl.lit(2).alias("round_num")],
        [pl.lit(0).alias("y_ct_win")],
        [pl.lit(None, dtype=pl.Float64).alias(ELAPSED)],
    ],
    ids=["before-freeze-end", "at-end-tick", "wrong-segment", "wrong-round", "wrong-label", "null"],
)
def test_verification_rejects_invalid_target_ticks_and_identity(round_and_samples, change):
    samples, rounds = round_and_samples
    with pytest.raises(PipelineError, match="Misaligned sample"):
        validate_timeline(samples.head(1).with_columns(*change), rounds)


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="requires the ml extra")
def test_preprocessing_only_fits_training_and_keeps_exactly_v1_feature_columns():
    from cs2_pipeline.phase2_temporal import fit_preprocessing, transform

    columns = feature_columns(1)
    features = {name: [1.0, None, 3.0, 10.0] for name in columns}
    frame = pl.DataFrame({**features, "demo_sha256": ["abc"] * 4, "round_id": [0, 0, 0, 1]})
    state = fit_preprocessing(frame)
    assert state["columns"] == columns and state["medians"][0] == 3
    assert state["round_weighted_mean"][0] == pytest.approx((1 + 3 + 3) / 6 + 10 / 2)
    later = frame.head(2).with_columns(pl.lit(1000.0).alias(columns[-1]))
    transformed = transform(later, state)
    assert transformed.shape == (2, 47)
    assert transformed[1, 0] == pytest.approx(
        (3 - state["round_weighted_mean"][0]) / state["round_weighted_scale"][0]
    )
    assert state["medians"][-1] == 3


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires temporal extra")
def test_padded_tokens_cannot_affect_current_prediction_and_parameters_match():
    import torch

    from cs2_pipeline.temporal_net import DeviceWindows, WindowTransformer

    spec = {"layers": 2, "hidden_dim": 16, "heads": 4, "feed_forward_dim": 32, "dropout": 0.1}
    torch.manual_seed(42)
    single = WindowTransformer(47, 8, spec).eval()
    torch.manual_seed(42)
    history = WindowTransformer(47, 8, spec).eval()
    for key, expected in single.state_dict().items():
        assert torch.equal(expected, history.state_dict()[key])
    values = np.arange(3 * 47, dtype=np.float32).reshape(3, 47) / 47
    rows = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 2]])
    data = DeviceWindows.from_arrays(
        values, rows, np.array([0, 1, 1]), np.ones(3), torch.device("cpu")
    )
    padded, mask = data.batch(slice(None))
    assert mask.tolist() == [[True, True, False], [True, False, False], [False, False, False]]
    poisoned = padded.clone()
    poisoned[mask] = 100.0
    with torch.inference_mode():
        torch.testing.assert_close(history(padded, mask), history(poisoned, mask))
        torch.testing.assert_close(
            single(padded[:1, -1:], mask[:1, -1:]), history(padded[:1], mask[:1])
        )


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="requires the ml extra")
def test_historical_replay_rejects_altered_checkpoint_or_selection_receipt(tmp_path):
    import json

    from cs2_pipeline.parse import sha256_file
    from cs2_pipeline.phase2_temporal_validation import FROZEN_FILES, verify_frozen_run

    hashes = {}
    for name in FROZEN_FILES:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        hashes[name] = sha256_file(path)
    receipt = tmp_path / "selection.json"
    receipt.write_text(json.dumps({"pre_exploration_sha256": hashes}), encoding="utf-8")
    selection_hash = sha256_file(receipt)
    assert verify_frozen_run(tmp_path, selection_hash)["pre_exploration_sha256"] == hashes
    (tmp_path / "history8_transformer.pt").write_bytes(b"different checkpoint")
    with pytest.raises(PipelineError, match="history8_transformer.pt"):
        verify_frozen_run(tmp_path, selection_hash)
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(PipelineError, match="selection receipt"):
        verify_frozen_run(tmp_path, selection_hash)

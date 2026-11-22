"""Reject references that cannot be paired with the new models exactly by target ID."""

import importlib.util

import polars as pl
import pytest

from cs2_pipeline.schema import PipelineError

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None, reason="requires the ml extra"
)


def test_reference_alignment_verifies_checksum_and_exact_ids(tmp_path):
    from cs2_pipeline.matched34_score import reference_probabilities
    from cs2_pipeline.parse import sha256_file

    path = tmp_path / "predictions.parquet"
    model = "frozen_v1"
    pl.DataFrame({"sample_id": ["b", "a"], model: [0.7, 0.3]}).write_parquet(path)
    digest = sha256_file(path)
    ids = pl.Series("sample_id", ["a", "b"])
    assert reference_probabilities(path, digest, model, ids).to_list() == [0.3, 0.7]
    with pytest.raises(PipelineError, match="different target IDs"):
        reference_probabilities(path, digest, model, pl.Series("sample_id", ["a", "different"]))
    with pytest.raises(PipelineError, match="predictions changed"):
        reference_probabilities(path, "0" * 64, model, ids)

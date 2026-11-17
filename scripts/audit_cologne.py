"""Recheck the frozen experiment's predictions, metrics and source quality."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from cs2_pipeline.calibrate import experiment_frame, metrics
from cs2_pipeline.dataset import read_dataset, write_json
from cs2_pipeline.parse import sha256_file
from cs2_pipeline.predict import predict_probabilities
from cs2_pipeline.schema import PipelineError
from cs2_pipeline.validate import validate_directory


def main() -> None:
    dataset = Path("data/datasets/mirage_cologne_experiment_v1")
    run = Path("models/logistic_cologne_calibrated_v1")
    metadata, tables = read_dataset(dataset)
    model = json.loads((run / "model.json").read_text(encoding="utf-8"))
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if report["dataset_manifest_sha256"] != sha256_file(dataset / "dataset.json") or report[
        "calibrated_model_sha256"
    ] != sha256_file(run / "model.json"):
        raise PipelineError("实验报告的模型/数据哈希不符。")
    test = experiment_frame(tables).filter(pl.col("split") == "test")
    predictions = pl.read_parquet(run / "predictions.parquet").sort("sample_id")
    if predictions["sample_id"].to_list() != test["sample_id"].to_list():
        raise PipelineError("预测与独立测试集 sample_id 不一致。")
    predicted = predict_probabilities(model, test.select("sample_id", *model["feature_columns"]))
    if not predictions.select("sample_id", "p_ct_win").equals(predicted):
        raise PipelineError("保存的预测与冻结模型重新推理不同。")
    base = json.loads(Path("models/logistic_rio_v1_stable/model.json").read_text())
    native_base = predict_probabilities(base, test.select("sample_id", *base["feature_columns"]))
    if native_base["p_ct_win"].to_list() != predictions["p_ct_win_base"].to_list():
        raise PipelineError("原始基线预测不符。")
    for name, column in (("base", "p_ct_win_base"), ("calibrated", "p_ct_win")):
        if metrics(test, predictions[column].to_list()) != report["test"][name]:
            raise PipelineError(f"{name} 报告指标与保存预测重算不同。")
    local = json.loads(Path("configs/cologne_local.json").read_text())
    quality = []
    missing: dict[str, int] = {}
    derived = 0
    for group in local["maps"]:
        for index, part in enumerate(group["parts"], 1):
            identifier = f"{group['match_id'].replace(':', '-')}-m{group['map_index']}-p{index}"
            path = Path("data/processed/cologne") / identifier
            q = validate_directory(path)
            if not q["passed"]:
                raise PipelineError(f"来源重读验证失败：{identifier}")
            for column, count in q["missing_values"]["snapshots"].items():
                missing[column] = missing.get(column, 0) + count
            derived += q["event_derived_snapshot_count"]
            quality.append(
                {
                    "recording": part["path"],
                    "sha256": part["sha256"],
                    "map_id": f"{group['match_id']}:m{group['map_index']}",
                    **{
                        k: q[k]
                        for k in (
                            "round_count",
                            "snapshot_count",
                            "valid_sample_tick_count",
                            "anomaly_count",
                            "warnings",
                            "passed",
                            "files_readable",
                        )
                    },
                }
            )
    result = {
        "audit_passed": True,
        "map_name": "de_mirage",
        "local_recordings": len(quality),
        "local_maps": len(local["maps"]),
        "local_rounds": sum(q["round_count"] for q in quality),
        "local_player_rows": sum(q["snapshot_count"] for q in quality),
        "local_valid_ticks": sum(q["valid_sample_tick_count"] for q in quality),
        "local_unique_players": tables["nodes"]
        .join(test.select("sample_id"), on="sample_id")["steamid"]
        .n_unique(),
        "local_anomalies": sum(q["anomaly_count"] for q in quality),
        "local_event_derived_dead_rows": derived,
        "missing_values_snapshots": missing,
        "source_quality": quality,
        **{
            k: report[k]
            for k in (
                "counts",
                "test",
                "calibration",
                "paired_match_bootstrap",
                "base_model_sha256",
                "calibrated_model_sha256",
                "dataset_manifest_sha256",
                "split_plan_sha256",
                "limitations",
            )
        },
    }
    write_json(Path("reports/cologne_experiment.json"), result)
    print(
        f"审计通过：{result['local_maps']} 地图 / {result['local_rounds']} 回合 / "
        f"{result['local_valid_ticks']} 时点；reports/cologne_experiment.json"
    )


if __name__ == "__main__":
    main()

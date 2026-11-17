"""Portable logistic inference: JSON parameters, explicit feature allowlist."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl

from cs2_pipeline.features import feature_columns, validate_feature_table
from cs2_pipeline.schema import PipelineError


def predict_logits(model: dict[str, Any], features: pl.DataFrame) -> pl.DataFrame:
    """Apply frozen training-only preprocessing and return uncalibrated linear logits."""
    columns = feature_columns(model.get("feature_version", 1))
    validate_feature_table(features, feature_version=model.get("feature_version", 1))
    if model.get("model_schema_version") != 1 or model.get("feature_columns") != columns:
        raise PipelineError("模型格式或特征白名单不兼容。")
    parameters = model.get("parameters", {})
    try:
        medians, indicators, means, scales, weights = (
            parameters[k]
            for k in (
                "imputation_values",
                "indicator_indices",
                "scale_mean",
                "scale_std",
                "coefficients",
            )
        )
        intercept = parameters["intercept"]
        width = len(columns)
        if len(medians) != width or len(means) != width + len(indicators):
            raise ValueError("预处理参数长度不符")
        if len(weights) != len(means) or len(scales) != len(means):
            raise ValueError("线性模型参数长度不符")
        if len(set(indicators)) != len(indicators) or any(
            not isinstance(i, int) or not 0 <= i < width for i in indicators
        ):
            raise ValueError("缺失值指示器索引错误")
        if not all(math.isfinite(v) for v in [*medians, *means, *scales, *weights, intercept]):
            raise ValueError("模型参数含无效数值")
        if any(v <= 0 for v in scales):
            raise ValueError("缩放尺度必须为正")
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError(f"模型参数损坏：{exc}。请重新训练。") from exc
    predictions = []
    for row in features.select("sample_id", *columns).iter_rows():
        raw = row[1:]
        vector = [medians[i] if value is None else value for i, value in enumerate(raw)]
        vector += [float(raw[i] is None) for i in indicators]
        logit = intercept + sum(
            coefficient * (value - mean) / scale
            for coefficient, value, mean, scale in zip(weights, vector, means, scales, strict=True)
        )
        if not math.isfinite(logit):
            raise PipelineError("预测计算溢出；请检查输入数值及模型参数。")
        predictions.append((row[0], logit))
    return pl.DataFrame(
        predictions, schema={"sample_id": pl.String, "logit": pl.Float64}, orient="row"
    )


def predict_probabilities(model: dict[str, Any], features: pl.DataFrame) -> pl.DataFrame:
    """Apply the optional frozen sigmoid calibration; old model JSON stays compatible."""
    logits = predict_logits(model, features)
    calibration = model.get("calibration")
    slope, intercept = 1.0, 0.0
    if calibration is not None:
        try:
            slope, intercept = calibration["slope"], calibration["intercept"]
            if calibration["method"] != "sigmoid_logit" or slope <= 0:
                raise ValueError("校准必须使用正斜率 sigmoid_logit")
            if not all(math.isfinite(v) for v in (slope, intercept)):
                raise ValueError("校准参数非有限数值")
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"校准参数损坏：{exc}") from exc
    probabilities = []
    for logit in logits["logit"]:
        value = slope * logit + intercept
        if not math.isfinite(value):
            raise PipelineError("校准计算溢出；请检查模型参数。")
        probabilities.append(
            1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
        )
    return logits.select("sample_id").with_columns(
        pl.Series("p_ct_win", probabilities, dtype=pl.Float64)
    )


def predict_file(
    model_path: Path,
    features_path: Path,
    out: Path,
    *,
    map_name: str | None = None,
) -> pl.DataFrame:
    if out.exists():
        raise PipelineError(f"输出文件已存在：{out}；请指定新的 --out。")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if not isinstance(model, dict):
        raise PipelineError("model.json 必须是对象。")
    if (features_path.parent / "dataset.json").is_file():
        # Local import avoids a dependency cycle with the lightweight model exporter.
        from cs2_pipeline.dataset import read_dataset

        dataset_metadata, tables = read_dataset(features_path.parent)
        if features_path.name != "features.parquet":
            raise PipelineError("数据集推理必须使用其已验证的 features.parquet。")
        map_name = dataset_metadata["map_name"]
        features = tables["features"]
    else:
        features = pl.read_parquet(features_path)
    if not map_name or map_name != model.get("map_name"):
        raise PipelineError("地图未知或与模型不同；独立特征文件必须通过 --map 声明正确地图。")
    predictions = predict_probabilities(model, features)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".predict-stage-", dir=out.parent) as temp:
        file = Path(temp) / "predictions.parquet"
        predictions.write_parquet(file, compression="zstd")
        if not pl.read_parquet(file).equals(predictions):
            raise PipelineError("预测文件往返读取失败。")
        file.rename(out)
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map", dest="map_name", help="独立特征文件的地图；数据集目录内会自动校验")
    args = parser.parse_args(argv)
    try:
        predictions = predict_file(args.model, args.features, args.out, map_name=args.map_name)
    except (PipelineError, OSError, ValueError, TypeError, pl.exceptions.PolarsError) as exc:
        print(f"预测失败：{exc}", file=sys.stderr)
        return 1
    print(f"已生成 {predictions.height} 条 CT 回合胜率：{args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

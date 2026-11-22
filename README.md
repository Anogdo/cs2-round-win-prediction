# CS2 Mirage Round Win Prediction

A STAT UN3108 study of Counter-Terrorist (CT) round win probabilities in professional Counter-Strike 2 Mirage replays. It compares logistic regression, XGBoost, and small Transformers trained on the same maps.

[中文项目文档](README.zh-CN.md) · [Experimental design](docs/mirage_temporal_experiment.md) · [Data and setup](data/README.md) · [Results](results/README.md)

## Study target

How do traditional models and small Transformers compare when they are fitted on the same Mirage data? The original CS:GO study used changes in round win probability to value in-game actions. This project investigates its probability estimation step and the improvement in prediction as a round progresses, motivated by Table II and Figure 6. The Transformer experiments also test recent history, training duration, training set size, and dropout. [Xenopoulos, Doraiswamy, and Silva (2020)](https://arxiv.org/abs/2011.01324)

## Approach

- Awpy, Polars, and PyArrow turn professional demos into five-second snapshots. The matched historical fit uses 34 Mirage maps, 730 rounds, and 13,208 prediction ticks. The two-layer Transformer takes one snapshot or up to eight from the same round, ending at the prediction tick.
- Logistic regression and both Transformers share the 47 V1 features, fit-only imputation and scaling, and targets. The archived XGBoost reference uses the same 34 fit maps. Every round has equal total weight. Log loss is the primary metric, with Brier score and ROC AUC as secondary metrics.
- The matched comparison reuses 50 previously observed maps for retrospective scoring. Later checkpoint and dropout experiments use 10 historical development maps from BLAST Bounty and the Esports World Cup. [Data splits and controls](docs/mirage_temporal_experiment.md)

## Matched historical results

| Model | Log loss ↓ | Brier score ↓ | ROC AUC ↑ |
| --- | ---: | ---: | ---: |
| Matched logistic regression | **0.4322** | **0.1448** | **0.8713** |
| Single-snapshot Transformer, seed 42 | 0.4348 | 0.1455 | 0.8695 |
| Up-to-eight-snapshot Transformer, seed 42 | 0.4386 | 0.1472 | 0.8669 |
| Archived CUDA XGBoost | 0.4435 | 0.1465 | 0.8682 |

On those 50 maps, the single-snapshot Transformer minus logistic regression log loss was **+0.00258**, with a 95% paired match-bootstrap interval of **[−0.00391, +0.00927]**. The effect of adding history varied by seed. The study also reproduced the direction of the original paper's time trend. On the same 906 rounds sampled at both times, the earlier V1 logistic model's log loss fell from 0.6392 at the start of the round to 0.4021 at 60 seconds. [Matched comparison](docs/mirage_temporal_experiment.md#follow-up-identical-34-map-training-set-november-21-2026) · [Timing and error analysis](results/README.md#timing-and-error-analysis)

## Current conclusion

With 32 training maps, longer Transformer training lowered training loss and raised development loss. After refitting on 34 maps, all three development-selected checkpoints had higher retrospective log loss than matched logistic regression. Expanding to 47 training maps lowered both candidates' development log loss for seed 42, with uncertainty around the gap. Dropout 0.20 reduced late overfitting and missed the prespecified requirement to improve mean log loss in every development cohort. [Follow-up experiments](docs/mirage_temporal_experiment.md#training-duration)

The prospective PGL Major Singapore 2026 comparison uses 49 fit maps for both logistic regression and the single-snapshot Transformer. The candidate weights and scoring rules were frozen on November 21, 2026. Singapore scoring is pending. [Evaluation scope and frozen settings](docs/mirage_temporal_experiment.md#frozen-singapore-evaluation)

## Run the temporal experiment

The comparison uses the optional `temporal` extra on Windows or Linux with a compatible NVIDIA GPU. The first real-data smoke epoch used FP32, ran 99 batches in 1.12 seconds on an RTX 4070 Ti, and peaked at 72.1 MiB allocated by PyTorch. The matched and follow-up runs use archived feature tables, processed round records, and verified artifacts from earlier stages. [Archived inputs](data/README.md#archived-temporal-inputs) · [Run commands](docs/mirage_temporal_experiment.md#follow-up-reproduction)

## Reproduce the single-demo pipeline

Use Python 3.11+ and [uv](https://docs.astral.sh/uv/). Run the following from the repository root with fresh output directories. The example runs on CPU and downloads one public Awpy fixture (347 MB).

```sh
uv sync --frozen --extra dev --extra ml --extra context
uv run --frozen python scripts/download_sample.py
uv run --frozen cs2-parse --demo data/raw/match.dem --source docs/demo_source.json --out data/processed/match --interval 5
uv run --frozen cs2-validate --input data/processed/match
uv run --frozen cs2-build-dataset --manifest configs/phase2.json --out data/datasets/sample_v1
uv run --frozen cs2-train --dataset data/datasets/sample_v1 --out models/sample_v1 --smoke-test
uv run --frozen cs2-predict --model models/sample_v1/model.json --features data/datasets/sample_v1/features.parquet --out models/sample_v1/predicted_all.parquet
uv run --frozen python -m pytest -q
```

Expect 18 rounds and 295 sampled ticks, a JSON model, and a Parquet file of win probabilities. The [public Rio benchmark](data/README.md#public-rio-benchmark) runs a scored train, validation, and test split on nine pinned replays. The CUDA experiments use the `gpu` extra and require a supported NVIDIA GPU and driver.

## Local demo

After installing the dependencies above, run `uv run --frozen --no-sync python -m cs2_pipeline.monitor` and open `http://127.0.0.1:8765`. The dashboard shows training stages, logs, and metrics. The temporal and matched evaluation presets use the CUDA environment and archived inputs described above. [Monitor guide (中文)](docs/training_ui.md)

## Repository contents

| Path | Contents |
| --- | --- |
| [`code/`](code/README.md) | Python implementation and links to the experiment manifests, scripts, and tests |
| [`data/`](data/README.md) | Source attribution, download and import instructions, and dataset layout |
| [`results/`](results/README.md) | Model comparisons, machine-readable scores, and learning curves |
| [`docs/`](docs/) | Detailed experiment notes, source provenance, and monitor documentation |
| [`poster/`](poster/) and [`report/`](report/) | Locations for the course presentation and summary PDFs |

## References

- Xenopoulos, Doraiswamy, and Silva (2020) used changes in win probability to value actions in CS:GO. This project studies the round-probability estimation step on a separate CS2 corpus. [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324)
- Vaswani et al. (2017) introduced the Transformer architecture. The comparison uses a two-layer encoder and a binary readout for the round outcome. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Demo parsing uses Awpy, Polars, and PyArrow. The baselines use scikit-learn and XGBoost, and the temporal models use PyTorch. [Dependencies](pyproject.toml), [Awpy documentation](https://awpy.readthedocs.io/en/latest/)

## Acknowledgements

This work was prepared for STAT UN3108, Applied Deep Learning and AI, at Columbia University. OpenAI Codex assisted with implementation, debugging, tests, and repository documentation. The project uses a public Awpy fixture and replay sources documented by OpenCS2 and HLTV. [Data attribution](docs/DATA_SOURCES.md)

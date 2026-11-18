# CS2 Mirage Round Win Prediction

A STAT UN3108 project on estimating the Counter-Terrorist (CT) side's chance of winning a round from professional Counter-Strike 2 replays. The current study tests whether recent game states improve predictions at a given tick.

[中文项目文档](README.zh-CN.md) · [Temporal experiment](docs/mirage_temporal_experiment.md) · [Data and setup](data/README.md) · [Results](results/README.md)

## Study target

Do recent snapshots help a small Transformer estimate CT round win probabilities? We compare the same architecture with one or up to eight snapshots, alongside a logistic regression baseline. Table II and Figure 6 of the earlier CS:GO study motivate the probability estimates and analysis by round time. [Xenopoulos, Doraiswamy, and Silva (2020)](https://arxiv.org/abs/2011.01324)

## Approach

- Awpy, Polars, and PyArrow parse professional demos into five-second snapshots. The expanded corpus contains 38 Mirage maps, 811 rounds, and 14,630 snapshots.
- The temporal models use the same 47 V1 features and target ticks. Windows contain observations up to the target tick from the same recording and round. Preprocessing is fitted on training data, and each round receives equal total weight.
- The three candidates were fitted on 18 Rio and Atlanta matches. Porto determined checkpoint selection, with two matches and 44 rounds. The selected checkpoints were then scored on 50 previously observed matches. Cologne remains reserved for calibration. [Temporal protocol](docs/mirage_temporal_experiment.md)

## First temporal results

| Model | Porto log loss ↓ | Previously observed 50 log loss ↓ |
| --- | ---: | ---: |
| Logistic regression | **0.3761** | **0.4420** |
| Current-snapshot Transformer | 0.3910 | 0.4469 |
| Up-to-eight-snapshot Transformer | 0.3965 | 0.4474 |

On the 50 observed matches, adding history changed log loss by **+0.00046** relative to the current-snapshot Transformer. The 95% interval from paired resampling by match was **[−0.00388, +0.00519]**. [Brier, ROC AUC, and error analysis](docs/mirage_temporal_experiment.md#results)

The earlier 34-match V1 logistic model reached 0.4321 log loss on the same observed set, against 0.5125 for its alive-player and elapsed-time reference. Its larger fit set included Porto. [Historical baselines](results/README.md#historical-50-match-evaluation)

## Conclusion

This first comparison used one training seed and one validation event. The estimated effect of adding history remains uncertain. Broader validation across events and seeds is needed before scoring on an unseen event.

## Run the temporal experiment

The comparison uses the optional `temporal` extra on Windows or Linux with a compatible NVIDIA GPU. The first real-data smoke epoch used FP32, ran 99 batches in 1.12 seconds on an RTX 4070 Ti, and peaked at 72.1 MiB allocated by PyTorch. The full run requires the original feature tables and their processed round records. [Archived inputs](data/README.md#archived-temporal-inputs), [run commands](docs/mirage_temporal_experiment.md#cuda-verification-and-reproduction)

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

Run `uv run --frozen --extra ml python -m cs2_pipeline.monitor` and open the local training monitor at `http://127.0.0.1:8765`. The dashboard shows training stages, logs, and metrics. The temporal preset uses the CUDA environment and archived inputs described above. [Monitor guide (中文)](docs/training_ui.md)

## Repository contents

| Path | Contents |
| --- | --- |
| [`code/`](code/README.md) | Python implementation and links to the experiment manifests, scripts, and tests |
| [`data/`](data/README.md) | Source attribution, download and import instructions, and dataset layout |
| [`results/`](results/README.md) | Model comparison and links to the machine-readable results under `reports/` |
| [`docs/`](docs/) | Detailed experiment notes, source provenance, and monitor documentation |
| [`poster/`](poster/) and [`report/`](report/) | Locations for the course presentation and summary PDFs |

## References

- Xenopoulos, Doraiswamy, and Silva (2020) used changes in win probability to value actions in CS:GO. This project studies the round-probability estimation step on a separate CS2 corpus. [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324)
- Vaswani et al. (2017) introduced the Transformer architecture. The comparison uses a two-layer encoder and a binary readout for the round outcome. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Demo parsing uses Awpy, Polars, and PyArrow. The baselines use scikit-learn and XGBoost, and the temporal models use PyTorch. [Dependencies](pyproject.toml), [Awpy documentation](https://awpy.readthedocs.io/en/latest/)

## Acknowledgements

This work was prepared for STAT UN3108, Applied Deep Learning and AI, at Columbia University. OpenAI Codex assisted with implementation, debugging, tests, and repository documentation. The project uses a public Awpy fixture and replay sources documented by OpenCS2 and HLTV. [Data attribution](docs/DATA_SOURCES.md)

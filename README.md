# CS2 Round Win Prediction

A Counter-Strike 2 replay pipeline and empirical study for STAT UN3108, Applied Deep Learning and AI. Given a snapshot from a professional Mirage round, the model estimates the probability that the Counter-Terrorist (CT) side wins the round.

[中文项目文档](README.zh-CN.md) · [Data and setup](data/README.md) · [Experiment results](results/README.md)

## Research question

How much do equipment, health, and other observed player states improve win probability estimates over a baseline that uses only the alive-player difference and elapsed time? We also compare added clock, bomb, and weapon context, tree models, and probability calibration.

## Approach

- Awpy, Polars, and PyArrow parse and validate five-second snapshots from professional demos. Each model input uses information available at its sampled tick. The final round winner is stored separately as the training label.
- The main historical corpus contains 38 Mirage maps, 811 rounds, and 14,630 snapshots. The fixed logistic regression model fits 34 maps. Event-level splits keep all snapshots from a match in the same set, and each round receives equal total weight.
- Experiments compare a 47-feature logistic regression (V1), a 96-feature logistic regression (V2), and shallow boosted trees. The comparison with 18 fit maps uses a dedicated Cologne calibration set and a separate Porto validation set.

## Results

In a retrospective evaluation on 50 additional Mirage matches (1,099 rounds), the fixed V1 model achieved **0.4321 log loss**, **0.1448 Brier score**, and **0.8714 ROC AUC**. The alive-player difference and elapsed-time baseline, trained on the same 34 maps, scored **0.5125 log loss**. See the model comparison, calibration results, and match-level uncertainty in the [results summary](results/README.md).

Among models fitted on the same 18 maps, the extra V2 features and the shallow tree models did not improve log loss on this set. These matches have been examined during development. Generalization to a new event remains to be evaluated. Performance here is for Mirage rounds with usable recordings.

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

Expect 18 rounds and 295 sampled ticks, a JSON model, and a Parquet file of win probabilities. The one-match run tests the pipeline. Reproducing the reported scores requires the [multi-match data sources and manifests](data/README.md). The optional CUDA experiments use the `gpu` extra and require a supported NVIDIA GPU and driver.

## Local demo

Run `uv run --frozen --extra ml python -m cs2_pipeline.monitor` and open [the training monitor](http://127.0.0.1:8765). It shows training stages, logs, and metrics. Each preset requires its corresponding local dataset. [Monitor guide (中文)](docs/training_ui.md)

## Repository contents

| Path | Contents |
| --- | --- |
| [`code/`](code/README.md) | Guide to the Python package in `src/`, experiment manifests in `configs/`, and scripts and tests |
| [`data/`](data/README.md) | Source attribution, download and import instructions, and dataset layout |
| [`results/`](results/README.md) | Model comparison and links to the machine-readable results under `reports/` |
| [`docs/`](docs/) | Detailed experiment notes, source provenance, and monitor documentation |
| [`poster/`](poster/) and [`report/`](report/) | Locations for the course presentation and summary PDFs |

## References and acknowledgements

- Xenopoulos, Doraiswamy, and Silva (2020) used changes in win probability to value actions in CS:GO. This project studies the round-probability estimation step on a separate CS2 corpus. [Valuing Player Actions in Counter-Strike: Global Offensive](https://arxiv.org/abs/2011.01324)
- Demo parsing uses Awpy. Thanks to the Awpy maintainers for the public test fixture and to OpenCS2 and HLTV for the documented demo sources. [Awpy documentation](https://awpy.readthedocs.io/en/latest/), [demo attribution](docs/DATA_SOURCES.md)

# CS2 Round Win Prediction

A Counter-Strike 2 replay pipeline and empirical study for STAT UN3108, Applied Deep Learning and AI. Given a snapshot from a professional Mirage round, the model estimates the probability that the Counter-Terrorist (CT) side wins. This lets a replay analyst track how the outlook changes during the round.

[中文项目文档](README.zh-CN.md) · [Data and setup](data/README.md) · [Experiment results](results/README.md)

## Study target

Table II of Xenopoulos, Doraiswamy, and Silva (2020) compares round win probability models in CS:GO. Figure 6 measures predictive performance as rounds progress. We investigate both questions on CS2 Mirage, beginning with a baseline that uses alive-player difference and elapsed time. [Original paper](https://arxiv.org/abs/2011.01324)

## Approach

- Awpy, Polars, and PyArrow parse and validate five-second snapshots from professional demos. Each model input uses information available at its sampled tick. The final round winner is stored separately as the training label.
- The main historical corpus contains 38 Mirage maps, 811 rounds, and 14,630 snapshots. The fixed logistic regression model fits 34 maps. Event-level splits keep all snapshots from a match in the same set, and each round receives equal total weight.
- Experiments compare a 47-feature logistic regression (V1), a 96-feature logistic regression (V2), and shallow boosted trees. The comparison with 18 fit maps uses a dedicated Cologne calibration set and a separate Porto validation set.

## Results

In a retrospective evaluation on 50 additional Mirage matches (1,099 rounds), the fixed V1 model achieved **0.4321 log loss**, **0.1448 Brier score**, and **0.8714 ROC AUC**. The alive-player difference and elapsed-time baseline, trained on the same 34 maps, scored **0.5125 log loss**. See the model comparison, calibration results, and match-level uncertainty in the [results summary](results/README.md).

The paper's XGBoost model led on log loss. On this CS2 dataset, the 34-map logistic model had lower log loss than the 34-map XGBoost model (0.4321 versus 0.4435). The studies use different games, features, and evaluation designs. The [results summary](results/README.md) shows the feature comparisons and the matched analysis by round time.

## Conclusion

Player state helped estimate Mirage round outcomes beyond the alive-player and time baseline. The 50-match set was examined during development. Prospective evaluation on an independent event remains open.

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

Run `uv run --frozen --extra ml python -m cs2_pipeline.monitor` and open [the training monitor](http://127.0.0.1:8765). It shows training stages, logs, and metrics. Each preset requires its corresponding local dataset. [Monitor guide (中文)](docs/training_ui.md)

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
- Demo parsing uses Awpy, Polars, and PyArrow. The supervised baselines use scikit-learn and XGBoost. [Dependencies](pyproject.toml), [Awpy documentation](https://awpy.readthedocs.io/en/latest/)

## Acknowledgements

This work was prepared for STAT UN3108, Applied Deep Learning and AI, at Columbia University. OpenAI Codex assisted with implementation, debugging, tests, and repository documentation. The project uses a public Awpy fixture and replay sources documented by OpenCS2 and HLTV. [Data attribution](docs/DATA_SOURCES.md)

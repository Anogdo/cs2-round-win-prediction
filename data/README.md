# Data

The inputs are professional Counter-Strike 2 replay files (`.dem`). The sample downloader writes to `data/raw/`. For multi-match runs, place manually obtained recordings at the paths specified in the manifests. Parsed matches, extracted context, and labeled datasets are generated locally.

## Public sample

Run `uv run --frozen python scripts/download_sample.py` from the repository root after installing the [dependencies](../README.md#reproduce-the-single-demo-pipeline). The script verifies the file size and SHA-256 for B8 vs Team Vitality, IEM Atlanta 2026, map 1 (Mirage). To place it manually, download the release asset listed in the manifest and save it as `data/raw/match.dem`. [Pinned sample manifest](../docs/demo_source.json), [Awpy fixtures](https://github.com/pnxenopoulos/awpy-fixtures)

## Multi-match experiments

| Corpus | How to obtain and prepare |
| --- | --- |
| Rio pilot | Use the nine pinned OpenCS2 replays to fit and score a logistic regression baseline. [Public benchmark steps](#public-rio-benchmark), [download manifest](../configs/corpus_downloads.json), [earlier V1 and V2 experiments](../docs/phase2_pilot.md) |
| Expanded 38-map corpus | Combine the public Atlanta and Rio demos with the Cologne, Porto, Beijing, and BLAST Rivals recordings. The source manifests list archive names, recording segments, and match pages. The [historical reparse](../docs/phase2_expanded_2026.md#复现与检查) checks exact hashes of the earlier parser outputs before processing the demos. [Cologne manifest](../configs/cologne_local.json), [additional Rio downloads](../configs/rio_additional_downloads.json), [HLTV match sources](../configs/user_hltv_sources.json), [reparse plan](../configs/reparse_expanded_2026.json) |
| 50-match exploration and ESL evaluation | Download the match replays from the listed match pages and place the archives in the local directories in the plans. The import and evaluation steps cover validation, recording segments, and scoring the fixed model. [50-match download list](../docs/holdout_download_list_50.md), [50-match evaluation](../docs/holdout_50_results.md#文件和复现), [ESL plan](../configs/esl24_holdout_plan.json) |

The expanded and evaluation experiments require the recordings specified in their manifests. Use the match pages' Demo download links for HLTV archives. The batch RAR import requires UnRAR.

## Public Rio benchmark

This CPU run uses the locked Awpy 3.3 parser and nine public replays (3.92 GB total). It uses a chronological split of seven training matches, one validation match, and one test match. Start from the repository root with fresh output paths.

```sh
uv sync --frozen --extra ml
uv run --frozen cs2-acquire --manifest configs/corpus_downloads.json --raw data/raw/rio_public --processed data/processed/rio_public --dataset-manifest data/processed/rio_public_sources.json
uv run --frozen cs2-build-dataset --manifest data/processed/rio_public_sources.json --group-by match --out data/datasets/rio_public
uv run --frozen cs2-train --dataset data/datasets/rio_public --out models/rio_public
```

Expect 173 rounds, 3,038 sampled ticks, and a 47-feature model. The run writes metrics to `models/rio_public/report.json`. [Verified benchmark scores](../results/README.md#public-rio-benchmark)

## Archived evaluation

The exact 50-match scores depend on the original 38-map training dataset, the model frozen before scoring, and the audited evaluation dataset. The evaluator checks their hashes. These archived inputs are local to the original run. With those inputs in place, run the following from the repository root and choose a fresh output path.

```sh
uv run --frozen --extra ml --extra context python -m scripts.evaluate_holdout_50 score --dataset data/datasets/mirage_holdout_50_v1 --out models/evaluations/holdout_50_recheck
```

The repository records the source lists, metrics, and audit checks. [Reported scores](../results/README.md), [frozen evaluation plan](../configs/holdout_50_plan.json)

## Generated files

- `data/processed/` holds validated Parquet files for rounds, players and sampled player states, plus source metadata.
- `data/context/` holds optional game clock, weapon and bomb-state features aligned to replay ticks.
- `data/datasets/` holds the derived features, labels, split assignments, and dataset metadata.

The source notes record match links, attribution, archives, and hashes. The OpenCS2 dataset card lists CC BY 4.0 for its mirror. Consult each other source for its own terms. [Data provenance and attribution (中文)](../docs/DATA_SOURCES.md)

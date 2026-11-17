# Data

The inputs are professional Counter-Strike 2 replay files (`.dem`). The sample downloader writes to `data/raw/`. For multi-match runs, place manually obtained recordings at the paths specified in the manifests. Parsed matches, extracted context, and labeled datasets are generated locally.

## Public sample

Run `uv run --frozen python scripts/download_sample.py` from the repository root after installing the [dependencies](../README.md#reproduce-the-single-demo-pipeline). The script verifies the file size and SHA-256 for B8 vs Team Vitality, IEM Atlanta 2026, map 1 (Mirage). To place it manually, download the release asset listed in the manifest and save it as `data/raw/match.dem`. [Pinned sample manifest](../docs/demo_source.json), [Awpy fixtures](https://github.com/pnxenopoulos/awpy-fixtures)

## Multi-match experiments

| Corpus | How to obtain and prepare |
| --- | --- |
| Rio pilot | Run `uv run --frozen --extra context python -m cs2_pipeline.acquire` to fetch and parse the nine pinned OpenCS2 demos. The V2 features also require `uv run --frozen --extra context python scripts/enrich_corpus.py`. [Download manifest](../configs/corpus_downloads.json), [pilot commands](../docs/phase2_pilot.md#重现流程) |
| Expanded 38-map corpus | Combine the public Atlanta and Rio demos with the Cologne, Porto, Beijing, and BLAST Rivals recordings. Get the archive names, map segments, and match pages from the source manifests. Use the [expanded experiment steps](../docs/phase2_expanded_2026.md#复现与检查) to import, reparse and train. [Cologne manifest](../configs/cologne_local.json), [additional Rio downloads](../configs/rio_additional_downloads.json), [HLTV match sources](../configs/user_hltv_sources.json), [reparse plan](../configs/reparse_expanded_2026.json) |
| 50-match exploration and ESL evaluation | Download the match replays from the listed match pages and place the archives in the local directories in the plans. The import and evaluation steps cover validation, recording segments, and scoring the fixed model. [50-match download list](../docs/holdout_download_list_50.md), [50-match evaluation](../docs/holdout_50_results.md#文件和复现), [ESL plan](../configs/esl24_holdout_plan.json) |

The expanded and evaluation experiments require the specific replay files in their manifests. Use the match pages' Demo download links for HLTV archives. The batch RAR import requires UnRAR. The [Atlanta sample](#public-sample) provides a smaller end-to-end run.

## Generated files

- `data/processed/` holds validated Parquet files for rounds, players and sampled player states, plus source metadata.
- `data/context/` holds optional game clock, weapon and bomb-state features aligned to replay ticks.
- `data/datasets/` holds the derived features, labels, split assignments, and dataset metadata.

The source notes record match links, attribution, archives, and hashes. The OpenCS2 dataset card lists CC BY 4.0 for its mirror. Consult each other source for its own terms. [Data provenance and attribution (中文)](../docs/DATA_SOURCES.md)

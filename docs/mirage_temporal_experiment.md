# Does recent in-round history improve Mirage round predictions?

Initial experiment, November 18, 2026. At each sampled tick, the task is to estimate the probability that CT wins the current Mirage round. The controlled comparison asks whether recent snapshots help a small Transformer relative to the same architecture restricted to the current snapshot. Logistic regression establishes the current-state baseline.

## Data and window construction

The experiment reuses the V1 features and targets in `mirage_expanded_2026_v1` and `mirage_holdout_50_v1`. It does not reparse demos. The loader verifies the frozen dataset hashes and joins features and labels one-to-one on `sample_id`. It checks the recording hash, match, map, segment, round number, label, and local tick against the archived round records. A sample must be within `[freeze_end_tick, end_tick)`. The end tick and winner are used to audit the retrospective sample population; the model receives neither of those fields, nor recording IDs, team IDs, or round IDs.

| Role | Events | Matches | Rounds | Targets |
| --- | --- | ---: | ---: | ---: |
| Fit | IEM Rio and IEM Atlanta | 18 | 353 | 6,333 |
| Reserved for calibration | IEM Cologne Major | 14 | 333 | 6,087 |
| Validation and epoch selection | BLAST Open Porto | 2 | 44 | 788 |
| Previously observed exploration | Five later events | 50 | 1,099 | 20,180 |

The partitions are chronological and disjoint by event, match, map and recording. Cologne was reserved; no calibrator was fit and no Cologne scores were used. `pgl-masters-bucharest-2026` remained excluded. The [frozen Phase 2 comparison plan](../configs/phase2_comparison_plan.json) defines the events; the [temporal plan](../configs/phase2_temporal_plan.json) defines the new models, selection rules, and input hashes.

The history model sees the target plus up to seven preceding valid snapshots from the same recording and round. Snapshots are nominally five seconds apart, so eight consecutive samples span 35 seconds; gaps can make the time span longer. Windows are sorted by original tick. A padding mask hides missing positions in short windows, and the current snapshot is always the final token. Neither model sees tokens after its target. The preprocessing uses fit-only medians and scaling, with no extra missingness indicators, so all candidates see the same 47 feature channels. Each target gets inverse round-length weight, giving every round total weight one in fitting and scoring.

## Training and selection

Both Transformers have 70,209 parameters: two encoder layers, hidden size 64, four heads, a 128-wide feed-forward layer, dropout 0.1, and a linear readout of the last token. They use the same initial seed (42), batch order, fixed relative-position encoding, and AdamW settings (batch size 64, learning rate 0.001, weight decay 0.01). The training loop ran 20 epochs per model in FP32 and kept each Transformer’s earliest best Porto log-loss checkpoint. It recorded the winning model and hashes of all checkpoints before loading the 50-match data. The logistic baseline uses `C=1.0` and the same preprocessed current snapshot. All reported comparisons are uncalibrated.

## Results

Scores are round weighted; lower log loss and Brier are better, and higher ROC AUC is better. All three new candidates were scored on the same 788 and 20,180 target IDs, respectively. The existing models are identified separately; the frozen 34-match V1 used Porto to fit, so a Porto score would not be held out.

| Model | Fit matches | Porto log loss | Porto Brier | Porto AUC | Observed 50 log loss | Observed 50 Brier | Observed 50 AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Logistic regression (Porto selection)** | 18 | **0.3761** | **0.1222** | **0.9068** | **0.4420** | **0.1484** | 0.8660 |
| Current-snapshot Transformer (epoch 1) | 18 | 0.3910 | 0.1274 | 0.8934 | 0.4469 | 0.1490 | 0.8664 |
| Up-to-eight-snapshot Transformer (epoch 1) | 18 | 0.3965 | 0.1290 | 0.8911 | 0.4474 | 0.1492 | 0.8665 |
| Prior CPU histogram tree, V1 (reference) | 18 | 0.4216 | 0.1374 | 0.8794 | 0.4487 | 0.1486 | 0.8636 |
| Prior frozen V1 logistic regression (reference) | 34 | — | — | — | 0.4321 | 0.1448 | 0.8714 |

History increased log loss by **0.0055** on Porto and **0.00046** on the observed 50 compared with the same architecture given one snapshot. The paired 95% match-cluster bootstrap interval for that 50-match difference is **[−0.00388, +0.00519]** over 2,000 draws; the Brier difference interval is **[−0.00136, +0.00187]**. The small AUC increase on the 50-match set did not improve either probability score. The round-weighted 10-bin expected calibration error there was 0.0296 for logistic, 0.0396 for the snapshot Transformer, and 0.0444 for the history Transformer. No post-training calibration was applied.

On the 12,512 common targets with a full eight-snapshot window, the log losses of the snapshot and history Transformers were 0.34663 and 0.34637 respectively. On the 6,569 targets with two to seven available snapshots, they were 0.58948 and 0.59174. This compares models only within each common subset. The subsets cover different round states and sometimes different rounds, so their absolute scores cannot be read as the effect of adding more snapshots or waiting longer.

The history network’s fitted loss fell from 0.4356 to 0.1015 between epochs 1 and 20, while its Porto loss rose from 0.3965 to 0.8644. The single-snapshot network showed the same divergence (fit: 0.4394 to 0.1874; Porto: 0.3910 to 0.5925). These are signs of overfitting this small fit/validation setup. This run provides no clear probability benefit from history with these aggregate V1 features and this configuration. It does not rule out a benefit with more training events, better regularization, or other temporal features.

Porto represents two matches from one event, all overlapping windows from each round are dependent, and only one training seed was used. The bootstrap resamples matches with all their rounds and ticks; it does not capture between-event or training-seed uncertainty. The 50-match dataset had already been studied and is retrospective. The next priority is broader event-level validation, followed by a new unseen event after fixing models, seeds and scoring rules.

## CUDA verification and reproduction

The [official PyTorch CUDA 13.0 wheel](https://pytorch.org/get-started/previous-versions/#v2110) is locked as the optional `temporal` extra for Windows/Linux. The run used Python 3.12.14, PyTorch 2.11.0+cu130, NVIDIA driver 621.01, and an RTX 4070 Ti (12 GB, `cuda:0`). It requires CUDA and does not fall back to CPU. The standalone real-data smoke test verified a finite FP32 forward pass, a nonzero CUDA gradient, and a parameter update. Its first full epoch took 1.12 seconds across 99 batches, peaking at 72.1 MiB allocated (92 MiB reserved by PyTorch). The full two-model run peaked at 72.3 MiB allocated. These memory figures describe PyTorch’s allocator, not total GPU memory in use. Reloading both selected checkpoints reproduced the validation probabilities exactly on this machine.

Run from the repository root with the archived datasets and their processed round records in place. Use fresh output paths. [Required files](../data/README.md#archived-temporal-inputs)

```sh
uv sync --frozen --extra dev --extra ml --extra context --extra temporal
uv run --frozen --no-sync python -m cs2_pipeline.phase2_temporal --dataset data/datasets/mirage_expanded_2026_v1 --out models/temporal_smoke_recheck --smoke-test
uv run --frozen --no-sync python -m cs2_pipeline.phase2_temporal --dataset data/datasets/mirage_expanded_2026_v1 --out models/temporal_full_recheck --progress-file runs/temporal_full_recheck/progress.jsonl
uv run --frozen --no-sync python -m pytest -q
uv run --frozen --no-sync ruff check code/src scripts tests
```

Add `--extra gpu` to `uv sync` if retaining the separate XGBoost experiment in the same environment. To launch from the dashboard, run `uv run --frozen --no-sync python -m cs2_pipeline.monitor` and select **Mirage · 时序 Transformer 对照**. The existing dashboard records epochs through its progress stream.

The original local run is in `models/monitor_runs/20261118T051803-7248d43d/`. Its report contains split counts, preprocessing, scores, epoch timings, and hashes. The separate smoke report is in `models/phase2_temporal_cuda_smoke_20261118/`. Verification checked target IDs, labels, ticks, weights and window lengths before recomputing the metrics from the saved predictions. [Public run summary](../results/mirage_temporal_initial.json)

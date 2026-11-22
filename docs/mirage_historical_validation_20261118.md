# Mirage historical validation: BLAST Bounty and Esports World Cup

This follow-up evaluates the three models selected or checkpointed on Porto in the [initial temporal experiment](mirage_temporal_experiment.md). It also checks two additional random seeds and the effect of adding a single bomb-state feature to the V1 logistic model. The new replays were collected for historical development validation. Their July–August dates precede Porto's selection dates in September, so this does not measure performance on games that occurred after model selection.

All models in the new comparisons fit only the original Rio and Atlanta corpus (18 matches, 353 rounds and 6,333 targets). Cologne remained reserved for calibration and was not used. The previously observed 50 matches supplied no targets to this run. `pgl-masters-bucharest-2026` remained untouched. Every target uses the same round weighting and the same evaluation rows within each comparison.

## Data verification

The 11 supplied RAR files contained one byte-identical duplicate. Ten Mirage maps across 11 recording parts yielded **214 rounds, 3,945 targets, and 39,450 player rows**. All 10 map scores matched the public HLTV records linked in the [intake manifest](../configs/phase2_validation_more_20261118.json). JiJieHao vs Aurora required reconstructing rounds 1–8 from the first part. On the companion recording, the same reconstruction matched Awpy on boundaries, winner and reason for rounds 9–19. Each history stays within its recording and round. The dataset audit found no roster, side, team identity, or alive-state violations and no incomplete targets. All rows are marked `historical_validation_only`; the ordinary training entry point refuses this dataset.

## Frozen temporal models

Each cell lists round-weighted Log Loss / Brier score / ROC AUC. The seed-42 checkpoints and preprocessing are unchanged from the initial comparison.

| Historical validation (matches / rounds / targets) | Logistic regression | Single-snapshot Transformer | Up-to-eight-snapshot Transformer |
| --- | ---: | ---: | ---: |
| Pooled (10 / 214 / 3,945) | 0.4459 / 0.1506 / 0.8620 | 0.4504 / 0.1511 / 0.8632 | 0.4449 / 0.1490 / 0.8673 |
| BLAST Bounty (3 / 59 / 1,067) | 0.4520 / 0.1530 / 0.8621 | 0.4592 / 0.1549 / 0.8593 | 0.4533 / 0.1549 / 0.8611 |
| Esports World Cup (7 / 155 / 2,878) | 0.4436 / 0.1497 / 0.8598 | 0.4470 / 0.1497 / 0.8625 | 0.4417 / 0.1467 / 0.8677 |

For seed 42, history reduced pooled log loss by 0.00550 relative to the single snapshot. The paired 95% match-bootstrap interval for history minus current was **[−0.01768, +0.00641]**. The corresponding difference from logistic regression was −0.00102, with an interval of **[−0.01679, +0.01289]**. The pooled 10-bin ECE was 0.0394 for logistic, 0.0485 for the single-snapshot Transformer and 0.0573 for the history model. The history model also improved log loss by about 0.006 on first targets, where it had no prior snapshot. That difference reflects what the two models learned during training; it cannot be attributed to earlier observations available at inference.

## Training-seed sensitivity

Seeds 7 and 2026 independently repeat epoch 1 on the same 18 matches, using the same preprocessing and architecture. Both pairs were saved before scoring the new events. The FP32 fits ran on the RTX 4070 Ti and took 0.80–1.35 seconds each. No seed was chosen using these validation results.

| Seed | Single-snapshot pooled Log Loss / Brier / AUC | History pooled Log Loss / Brier / AUC | History minus single Log Loss (95% match interval) |
| --- | ---: | ---: | ---: |
| 42 | 0.4504 / 0.1511 / 0.8632 | 0.4449 / 0.1490 / 0.8673 | −0.00550 [−0.01768, +0.00641] |
| 7 | 0.4448 / 0.1503 / 0.8622 | 0.4556 / 0.1537 / 0.8559 | +0.01077 [+0.00595, +0.01556] |
| 2026 | 0.4460 / 0.1499 / 0.8616 | 0.4474 / 0.1502 / 0.8611 | +0.00146 [−0.02468, +0.02548] |

The direction of the history effect varied by seed. Resampling whole matches (2,000 draws) preserves within-round dependence. The data cover only two events and do not estimate variability across new events. These results do not justify replacing the logistic reference with a temporal model.

## Bomb-state ablation

Two separately fitted logistic models share the same V1 inputs and missingness indicators. The sole additional input is `bomb_planted`, based on observed plant-completion and defuse events at or before the target tick in the same round. All 163 bomb-ended rounds across the fit, historical validation, and Porto recordings had observed plant events. Round end reasons were used only for that audit.

| Group | V1 Log Loss | V1 + bomb Log Loss |
| --- | ---: | ---: |
| Historical validation, pooled | 0.4460 | 0.4428 |
| BLAST Bounty | 0.4517 | 0.4558 |
| Esports World Cup | 0.4439 | 0.4379 |
| Porto (historical reference) | 0.3759 | 0.3707 |

On the same 3,945 historical validation targets, pooled Brier moved from 0.1507 to 0.1496, AUC from 0.8618 to 0.8648, and raw 10-bin ECE from 0.0397 to 0.0348. No calibration was fitted. The match-bootstrap log loss difference was −0.00319 (95% interval **[−0.01066, +0.00373]**), so the pooled improvement is inconclusive, and the direction reverses across the two events. The planted-only subset also worsened (0.2269 to 0.2514, 577 targets). This subset result is descriptive and applies only to those targets. Keep bomb state as a candidate for future preregistered evaluation.

## Reproduction

Run from the repository root. See the [source plan](../configs/phase2_validation_more_20261118.json) for the exact HLTV replays, original archive names, map indices, published scores and hashes. Preserve the original names under `data/raw/phase2_validation_more/`. The import reads only the pinned Mirage members and verifies maps, recorded rounds, identities, and published scores before writing the V1 validation tables.

```powershell
uv sync --frozen --extra dev --extra ml --extra context --extra temporal
uv run --frozen --no-sync python -m scripts.import_validation_more --out data/datasets/mirage_validation_more_recheck --progress runs/phase2_validation_more_recheck/progress.json
uv run --frozen --no-sync python -m cs2_pipeline.phase2_temporal_validation --dataset data/datasets/mirage_validation_more_recheck --out models/temporal_historical_recheck --progress-file runs/temporal_historical_recheck/progress.jsonl
uv run --frozen --no-sync python -m scripts.temporal_seed_sensitivity fit --out models/temporal_seed_fit_recheck
uv run --frozen --no-sync python -m scripts.temporal_seed_sensitivity score --dataset data/datasets/mirage_validation_more_recheck --frozen models/temporal_seed_fit_recheck --out models/temporal_seed_score_recheck
$validationHash = (Get-FileHash -LiteralPath data/datasets/mirage_validation_more_recheck/dataset.json).Hash.ToLowerInvariant()
uv run --frozen --no-sync python -m cs2_pipeline.phase2_bomb --development data/datasets/mirage_validation_more_recheck --dataset-sha256 $validationHash --out models/bomb_ablation_recheck
```

The [temporal validation plan](../configs/phase2_temporal_validation_plan.json) pins the original Porto selection receipt and this batch's intake manifest. The selection receipt in turn pins the three trained model files and their preprocessing. Scoring uses those unchanged artifacts. Open the existing training monitor to view the **新增 Mirage 历史验证数据** batch and the **Mirage · 冻结模型跨赛事验证** scoring preset.

Local reports and predictions: `models/monitor_runs/20261118T232352-6504cdb7/`, `models/temporal_seed_sensitivity_validation_v1/`, and `models/mirage_validation_more_bomb_20261118/`. Additional-seed weights are in `models/temporal_seed_sensitivity_fit_v1/`, and the intake receipt is `runs/phase2_validation_more_v1/progress.json`.

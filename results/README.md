# Results

## Matched model comparison and follow-ups

The matched 34-map fit produced retrospective log losses of **0.4322** for logistic regression, **0.4348** for the single-snapshot Transformer, **0.4386** for the history Transformer, and **0.4435** for the archived CUDA XGBoost reference. All models use the same 50 previously observed evaluation maps. The Transformer results shown here use seed 42. [Matched results and paired intervals](../docs/mirage_temporal_experiment.md#follow-up-identical-34-map-training-set-november-21-2026)

| Follow-up | Evaluation and evidence |
| --- | --- |
| Training duration | The 200-epoch runs overfit. The development-selected checkpoints scored worse than matched logistic regression on the retrospective cohort across all three seeds. [Results](../docs/mirage_temporal_experiment.md#training-duration) · [Learning curves](figures/mirage-training-duration.png) |
| Training set size | On the 10-map historical development set, the 16-to-32-map comparison showed no consistent improvement in the Transformer gap to logistic regression. [Results](../docs/mirage_temporal_experiment.md#training-set-size-and-event-expansion) · [Learning curves](figures/mirage-training-size.png) |
| Training expansion | Both models improved on the same 10 development maps when trained with the 15 additional Rotterdam and Astana maps. The difference between models varied with seed. [Results](../docs/mirage_temporal_experiment.md#training-set-size-and-event-expansion) |
| Dropout | Dropout 0.20 reduced late overfitting and missed the event-level adoption rule. [Results](../docs/mirage_temporal_experiment.md#dropout-ablation) · [Learning curves](figures/mirage-dropout.png) |
| Singapore Major | Candidate weights and scoring rules are frozen for the 49-map fit. Prospective scoring is pending. [Evaluation design](../docs/mirage_temporal_experiment.md#frozen-singapore-evaluation) · [Freeze receipt](mirage_singapore_freeze_20261121.json) |

The [machine-readable follow-up results](mirage_followup_20261121.json) include data counts, scores, paired intervals, and checksums of the archived reports.

## First temporal comparison

The 18-match logistic baseline had the lowest Porto log loss of the three candidates. The eight-snapshot Transformer scored 0.4474 on the 50 previously observed matches, compared with 0.4469 for the current-snapshot model. Both Transformer checkpoints were selected using Porto before the exploratory data was scored. [Metrics and limitations](../docs/mirage_temporal_experiment.md#results), [machine-readable scores](mirage_temporal_initial.json)

## Historical 50-match evaluation

All models below were scored on 20,180 observations from the same 1,099 Mirage rounds, with equal total weight per round. Each of the 50 matches was disjoint from the fit data. Some events appeared in earlier evaluation. This set has already been examined during development.

| Model | Fit maps | Calibration | Log loss ↓ | Brier ↓ | ROC AUC ↑ | ECE ↓ |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Alive difference and elapsed time | 34 | None | 0.5125 | 0.1758 | 0.7916 | 0.0247 |
| Fixed V1 logistic regression | 34 | None | 0.4321 | 0.1448 | 0.8714 | 0.0213 |
| XGBoost (CUDA) | 34 | None | 0.4435 | 0.1465 | 0.8682 | 0.0179 |
| V1 logistic regression | 18 | None | 0.4420 | 0.1484 | 0.8660 | 0.0293 |
| V1 logistic regression | 18 | Cologne | 0.4395 | 0.1473 | 0.8660 | 0.0191 |
| V2 logistic regression | 18 | Cologne | 0.4615 | 0.1557 | 0.8503 | 0.0263 |
| Histogram gradient boosting (CPU) | 18 | Cologne | 0.4482 | 0.1484 | 0.8636 | 0.0139 |
| XGBoost (CUDA) | 18 | Cologne | 0.4532 | 0.1494 | 0.8613 | 0.0138 |

The 95% match-bootstrap interval for the log-loss reduction of fixed V1 over alive difference and elapsed time was **[0.0646, 0.0962]**. The interval describes variation among these matches. It does not estimate performance variation across new events. The [audit and comparison notes](../docs/phase2_mirage_audit_20261117.md#同一批-50-场上的结果) document the paired comparisons, reliability bins, and performance by time, alive counts, and bomb state.

## Timing and error analysis

Among the same 906 rounds with observations at both 0 and 60 seconds, fixed V1 log loss fell from **0.6392** to **0.4021**. This matches the direction reported in Figure 6 of the CS:GO study and applies to rounds that lasted long enough to be sampled at both times. In the 50-match set, log loss was **0.5970** on snapshots with equal alive counts, which identifies a game state where the model's probabilities were less accurate. [Model and error analysis](../reports/phase2_exploration_final.json), [Original paper](https://arxiv.org/abs/2011.01324)

## Public Rio benchmark

The locked Awpy 3.3 run parsed all nine pinned Rio demos and fit the 47-feature logistic regression on seven matches. The validation and test splits each represent one match from the same tournament. [Reproduction steps](../data/README.md#public-rio-benchmark)

| Split | Rounds | Model log loss ↓ | Training prior log loss ↓ | Brier ↓ | ROC AUC ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 17 | 0.4362 | 0.7380 | 0.1406 | 0.8862 |
| Test | 24 | 0.5407 | 0.7440 | 0.1842 | 0.8214 |

## Detailed results

| Experiment | Documentation | Machine-readable evidence |
| --- | --- | --- |
| Rio pilot | [Experiment notes](../docs/phase2_pilot.md) | [Pilot results](../reports/rio_pilot.json) |
| Cologne event holdout | [Experiment notes](../docs/phase2_cologne.md) | [Cologne results](../reports/cologne_experiment.json) |
| Expanded corpus and supplementary ESL holdout | [Experiment notes](../docs/phase2_expanded_2026.md) | [Model and data audit](../reports/expanded_2026_model_audit.json) |
| 50-match evaluation | [Fixed-model evaluation](../docs/holdout_50_results.md) | [Fixed-model audit](../reports/holdout_50_audit_v2.json) |
| Feature and model comparison | [Data integrity and evaluation notes](../docs/phase2_mirage_audit_20261117.md) | [Comparison report](../reports/phase2_exploration_final.json), [data audit](../reports/phase2_data_audit_v2.json) |
| Current snapshot vs. up to eight snapshots | [Temporal experiment](../docs/mirage_temporal_experiment.md) | [First temporal comparison](mirage_temporal_initial.json) |
| Two additional historical events | [Frozen temporal validation, seed sensitivity, and bomb-state ablation](../docs/mirage_historical_validation_20261118.md) | Local reports and reproduction commands listed in the experiment notes |

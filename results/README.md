# Results

## 50-match exploratory evaluation

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

## Detailed results

| Experiment | Documentation | Machine-readable evidence |
| --- | --- | --- |
| Rio pilot | [Experiment notes](../docs/phase2_pilot.md) | [Pilot results](../reports/rio_pilot.json) |
| Cologne event holdout | [Experiment notes](../docs/phase2_cologne.md) | [Cologne results](../reports/cologne_experiment.json) |
| Expanded corpus and supplementary ESL holdout | [Experiment notes](../docs/phase2_expanded_2026.md) | [Model and data audit](../reports/expanded_2026_model_audit.json) |
| 50-match evaluation | [Fixed-model evaluation](../docs/holdout_50_results.md) | [Fixed-model audit](../reports/holdout_50_audit_v2.json) |
| Feature and model comparison | [Data integrity and evaluation notes](../docs/phase2_mirage_audit_20261117.md) | [Comparison report](../reports/phase2_exploration_final.json), [data audit](../reports/phase2_data_audit_v2.json) |

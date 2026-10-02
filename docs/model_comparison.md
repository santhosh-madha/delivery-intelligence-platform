# Ridge, random forest, and CatBoost comparison

Completed October 2, 2026. Four expanding chronological windows: October 19–22, 2022 (Asia/Shanghai). Total: 274,461 validation orders. No model was promoted or serving artifact replaced.

## Overall results

| Model | MAE min | RMSE min | P90 error min | Total fit seconds (4 folds) | Warm single request ms | Warm batch of 256 ms |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 6.7308 | 8.7697 | 13.8611 | 23.59 | 1.94 | 2.52 |
| Ridge | 6.8340 | 8.8998 | 14.0329 | 1.37 | 6.30 | 6.71 |
| Random forest | 7.1731 | 9.3371 | 14.7147 | 38.28 | 20.37 | 20.75 |
| Training median | 7.9637 | 10.5067 | 16.2000 | 0.01 | not measured | not measured |

## Daily validation MAE

| Day | Ridge | Random forest | CatBoost |
|---|---:|---:|---:|
| 2022-10-19 | 6.7867 | 7.0610 | 6.6650 |
| 2022-10-20 | 6.6619 | 6.9833 | 6.5711 |
| 2022-10-21 | 6.9818 | 7.3110 | 6.8651 |
| 2022-10-22 | 6.8907 | 7.3114 | 6.8051 |

## Courier segments

| Model | Seen MAE min | Unseen MAE min |
|---|---:|---:|
| CatBoost | 6.7152 | 7.4269 |
| Ridge | 6.8258 | 7.2020 |
| Random forest | 7.1671 | 7.4392 |
| Training median | 7.9642 | 7.9432 |

268,467 seen-courier and 5,994 unseen-courier validation orders; membership is based only on each fold’s training data.

## Decision

Keep CatBoost as the v1 serving candidate: it has the lowest MAE on all four days and the lowest measured inference latency. Its pooled improvement over Ridge is modest: approximately 0.103 minutes (6.2 seconds), or 1.51%. Ridge trains much faster and has better unseen-courier MAE. This does not prove CatBoost is universally best or that random forests cannot improve with tuning.

## Reproducibility and fairness

- Same eligible orders, target, six information features, chronological splits, and zero-minute prediction floor for every model. Labels must be available before training/validation boundaries. No random train/test split or subsampling of training rows.
- Ridge and random forest use training-only sparse one-hot encoding: rare IDs grouped at min_frequency=20; acceptance hour is categorical without grouping. Numeric ages are standardized. CatBoost receives native categorical strings.
- Ridge: alpha=10, lsqr, tolerance=1e-4, max_iter=2000; iteration limit checked.
- Random forest: 100 trees, max_depth=16, min_samples_leaf=20, max_features=0.5, bootstrap=True, random_state=42, n_jobs=4. This is one resource-bounded configuration, not a hyperparameter search.
- CatBoost: 300 iterations, depth=6, learning_rate=0.08, RMSE loss, l2_leaf_reg=3, random_seed=42, CPU, 4 threads; matches v1.
- Fit time includes model-specific preprocessing but excludes shared feature creation. Native numerical thread pools capped at four.
- Latency includes shared feature creation, categorical preprocessing, prediction, and flooring, starting from a raw dataframe. It excludes HTTP parsing, serialization, database writes, network, and cold starts. Three warmups; 30 single-row / 10 batch repeats per fold; table reports median of fold medians. Four-worker forest includes worker dispatch overhead. These are local measurements, not cloud SLAs.
- Execution order rotates between folds. Local timings may be affected by background load and thermal effects. Training times are single measurements per fold.
- Source SHA256 verified against the saved model. October 22 CatBoost MAE exactly reproduces the saved bundle’s validation MAE. Serving model checksum remains unchanged.
- Validation periods have already been inspected. These are model-selection results, not a pristine final holdout. Later periods were not evaluated here. Dataset covers a limited time/location and excludes outcomes beyond each validation cutoff.

## Run again

```bash
uv run --locked python -m delivery_intelligence_platform.compare_models
```

Each run creates a new directory under `artifacts/comparisons/` with protocol, source snapshot, split audits, daily results, pooled summary, and per-order validation predictions. Artifacts remain excluded from Git; this aggregate report may be committed.

Run used: `artifacts/comparisons/20261002T184655Z`.

# Delivery Intelligence & ETA Reliability Platform

An end-to-end machine learning system for delivery ETA prediction, delay-risk detection, and ETA reliability analysis.

## Project Overview

Delivery platforms rely on accurate estimated arrival times to improve customer experience and operational efficiency.

This project aims to build a production-oriented machine learning system that:

- Predicts delivery ETA using historical delivery data
- Identifies deliveries at risk of arriving late
- Evaluates ETA reliability
- Serves predictions through a real-time inference API
- Tracks model experiments and versions
- Logs predictions for monitoring
- Detects data and model drift
- Supports automated testing and deployment

## Dataset

The project will use historical operational delivery data from the Meituan-INFORMS TSL Research Challenge.

Raw data is not stored in this repository.

## Project Status

🚧 In development

Current phase: reproducible training, local MLflow tracking, and saved-model inference.

## V1 scope

Predict minutes from courier acceptance to delivery for on-demand orders.
The development cohort contains accepted orders with observed completion; pre-booked
orders and records without usable targets are excluded. The candidate is CatBoost
with courier ID. Its unseen-courier accuracy is weaker than the model without that
feature; the production routing decision remains deferred.

The notebooks remain the experiment history. Package code now owns the reproducible
training and inference path. No notebooks are executed by the training command.

## Run locally

From the project root, install the locked environment and run the checks:

```bash
uv sync --locked --group dev
uv run --locked pytest -q
```

Train the established October 22 validation experiment:

```bash
uv run --locked eta-train
```

Default behavior: train on records accepted before October 22 whose delivery labels
are known before that boundary; evaluate October 22 acceptances completed before
October 23. Later records are counted in the audit but are not evaluated. Boundary
exclusions intentionally preserve the notebook protocol, but underrepresent orders
that span midnight. This is a development evaluation, not a claim of a pristine
holdout or production readiness.

Other existing development folds can be reproduced explicitly:

```bash
uv run --locked eta-train --validation-start 2022-10-21 --validation-end 2022-10-22
```

Each invocation creates a new directory under `artifacts/models/<run-id>/` containing:

- `model.cbm`: the native CatBoost model.
- `manifest.json`: feature contract, data/model checksums, split, versions, settings,
  source-code fingerprints, and Git state.
- `metrics.json`: train/validation metrics, constant and promise baselines, and
  seen/unseen-courier results.
- `data_audit.json`: eligibility counts, exclusions, and split counts.
- `source_snapshot/`: exact package code, project configuration, and dependency lock.

The model is reloaded and checked against in-memory predictions before tracking
completes. All artifacts stay local and are ignored by Git. Raw data is never copied
into the model bundle or uploaded by this workflow.

## Inspect experiments

MLflow uses `artifacts/mlflow/mlflow.db` for local experiment metadata and
`artifacts/mlflow/run_artifacts/` for artifact copies. SQLite is only the local
experiment backend; the future application's prediction/outcome database remains
PostgreSQL. This milestone logs native model bundles as run artifacts; model-registry
promotion and a production alias are not implemented yet.

```bash
uv run --locked mlflow ui --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` and select the `delivery-eta-v1` experiment.

## Predict from the saved artifact

Use the exact bundle path printed by training in place of `<run-id>`:

```bash
uv run --locked eta-predict --model-dir artifacts/models/<run-id> --input docs/example_acceptance.json
```

The sample is a synthetic event, not a source delivery record. JSON accepts one
object or a list of objects. Required fields are `platform_order_time`,
`order_push_time`, `grab_time` (positive Unix seconds), and `poi_id`, `da_id`,
`courier_id` (nonnegative integer identifiers; zero is valid). Times must satisfy
creation <= push <= acceptance. Requests are on-demand by contract; if provided,
`is_prebook` must be zero. Fractional timestamps, booleans, missing identifiers,
and non-finite values are rejected.

The shared feature builder derives ages and acceptance hour in Asia/Shanghai;
targets and actual arrival times are never needed for prediction. Predictions have
a zero-minute floor and include the estimated arrival timestamp and model version.
Unseen IDs use CatBoost's native handling, not a separately trained fallback model.

Python callers can load once and predict repeatedly:

```python
import pandas as pd
from delivery_intelligence_platform.predict import EtaPredictor

predictor = EtaPredictor("artifacts/models/<run-id>")
predictions = predictor.predict(pd.read_json("docs/example_acceptance.json", typ="series").to_frame().T)
```

FastAPI prediction/outcome endpoints and PostgreSQL now run locally through Docker
Compose. Training does not start these services. Performance reporting is available
with `uv run --locked eta-report`; see [the reporting guide](docs/performance_reporting.md)
for demo exclusion, coverage, model segments, and report limitations. GitHub Actions configuration is documented in
[the CI guide](docs/continuous_integration.md). Hosted deployment remains a future milestone.


## Model comparison

See [the controlled Ridge, random forest, and CatBoost comparison](docs/model_comparison.md)
for accuracy, training time, inference latency, and model-selection limitations.

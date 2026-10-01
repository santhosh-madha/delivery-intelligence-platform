# Performance reporting

Run from the project root with PostgreSQL running:

```bash
uv run --locked eta-report
```

The command writes a timestamped Markdown report and machine-readable JSON under
`artifacts/reports/`. It only reads the database. Optional UTC window:

```bash
uv run --locked eta-report --since 2026-10-01T00:00:00+00:00 --until 2026-10-02T00:00:00+00:00
```

The window uses prediction creation time, inclusive start and exclusive end.
Only outcomes recorded before the end are considered available. This is a
reporting window, not a historical replay based on order acceptance time.

## Demo and real requests

`POST /predict` defaults to the query parameter `data_source=demo`. Swagger demos
are therefore excluded. For verified real delivery events use
`POST /predict?data_source=real` with the usual JSON request body. Outcomes inherit
eligibility from their parent prediction; submit only actual measured arrival
 times for real requests. The flag is supplied by the caller, not independently
verified by the system.

The migration labels existing predictions `unknown`, preserving every existing
record. Unknown records are excluded rather than assumed real. No historical
records are automatically relabeled. Changing this classification requires a
separate reviewed data correction.

## Interpreting the report

- Coverage is observed outcomes / eligible predictions. Pending deliveries stay
  in the denominator, including those without any outcomes.
- Errors use valid matched outcomes only. No matched outcomes means unavailable
  accuracy, never zero error. Invalid matched outcomes are counted separately.
- Bias is predicted minus actual minutes: negative means underestimation.
- MAE, RMSE, bias and 90th-percentile absolute error are in the JSON output.
- Results are separated by model version, actual-duration group, courier training
  membership, and prediction creation day in UTC.
- Duration groups are diagnostic: actual duration is unknown when predicting.
- Missing outcomes may bias results toward fast deliveries. Coverage must be read
  alongside error metrics; this report does not implement maturity windows,
  drift alerts, confidence intervals, or automated retraining.
- Courier membership uses each model manifest, a matching source-data SHA256, and
  its recorded chronological training split. If those are unavailable, membership
  is reported as unknown, not unseen. Source data is never needed by the API.
- The command loads the selected window into memory; use narrower date windows
  as the database grows. It does not publish individual requests or courier IDs.

## Deployment and verification

```bash
RUN_INTEGRATION=1 uv run --locked pytest -q
docker compose build api
docker compose up -d --wait db
docker compose run --rm --no-deps api alembic upgrade head
docker compose up -d --wait api
uv run --locked eta-report
```

The report runs on the host so it can optionally inspect local training artifacts.
Keep offline validation results separate from serving metrics: manual examples
and the October 22 validation dataset are not production monitoring evidence.

# API memory-limit check

Passed October 2, 2026 with the saved CatBoost model
`87a38ec8f34d4ed0a47437a3b1e120ee`.

| Measurement | Result |
|---|---:|
| API memory limit | 512 MiB |
| Swap | Disabled |
| Peak measured container memory | 346.5 MiB |
| Memory remaining below limit at peak | 165.5 MiB |
| Startup until healthy | 6.24 seconds |
| Sequential requests | 50 successful |
| Requests at concurrency 4 | 40 successful |
| Persisted test predictions | 90 |
| Out-of-memory events / kills | 0 / 0 |

The API ran with one CPU and a disposable PostgreSQL container. The original
application database was not used. Each response matched the saved model's
expected prediction. All test containers, temporary database storage, and network
were removed afterward. Peak comes from cgroup v2 `memory.peak`, which includes
container memory beyond the Python process and small measurement/health-check
processes. This is not merely Python heap usage.

This supports proceeding to a free-host deployment trial. It does not guarantee
Render performance: this test ran on local Linux aarch64 (Apple Silicon Docker),
with one CPU, a local database, and synthetic repeated requests. Cloud CPU,
architecture, TLS/network delays, cold starts, and sustained traffic differ.
The 512 MiB cap applies only to the API, not image building or the database.

## Repeat

With Docker Desktop running, from the project root:

```bash
uv run --locked python scripts/check_api_memory.py
```

The script uses port 18080 and Compose project `delivery-eta-memory-check`.
Do not run two copies simultaneously. It writes a timestamped JSON report under
`artifacts/memory_checks/`. The model directory can be supplied with `--model-dir`;
the response assertion is specific to the current v1 model and must be updated
when intentionally testing a different model.

Result artifact: `artifacts/memory_checks/20261002T190034Z/result.json`.

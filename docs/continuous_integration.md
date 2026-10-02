# Continuous integration

The `CI` workflow runs on pushes, pull requests, and manual dispatch. It has two
independent checks on Ubuntu 24.04:

- **Python and PostgreSQL tests:** installs Python 3.12 and uv 0.12.19, synchronizes
  `uv.lock`, and runs all tests with `RUN_INTEGRATION=1`. The tests create synthetic
  models and an isolated PostgreSQL 18 container, apply migrations, and remove it.
- **Docker build and import check:** builds the local `Dockerfile` image, imports the installed API
  and report module, and checks that Alembic can discover the migrations. This is
  an image packaging check, not a deployed HTTP smoke test.

Neither check needs your `.env`, dataset, saved models, or application database.
These GitHub Actions jobs do not publish images or deploy the application. Render
handles deployment separately and is configured to wait for checks to pass. The workflow has read-only repository
permissions and pins actions to commit IDs. uv and Python match the Docker setup.

## Check a run

The workflow and application files are committed, and CI has run successfully on
GitHub. Each new change gets its own result; an earlier successful run does not
prove that a later change passes.

Open the repository's **Actions** tab after pushing. The `CI` workflow shows two
job results. Open a failed job to inspect the failing step. Keep credentials,
raw datasets, local model bundles, and private notes out of commits.

## How this relates to deployment

`render.yaml` sets `autoDeployTrigger: checksPass`. Render builds
`Dockerfile.cloud`, downloads the published model, and starts the application
with database migrations. GitHub CI currently builds only `Dockerfile`; it does
not test the cloud image or the live Render service.

For this educational project, I checked the deployed API manually: readiness,
prediction, database persistence, API-key rejection, and outcome recording.
See [the deployment guide](cloud_deployment.md) for those results. A passing CI
run and a healthy deployed application are separate checks.

Required branch checks are a separate GitHub repository setting; this workflow
does not configure branch protection.

## Local equivalent

```bash
uv sync --locked --group dev
RUN_INTEGRATION=1 uv run --locked pytest -q
docker build --tag delivery-eta:ci .
docker run --rm delivery-eta:ci python -c "from delivery_intelligence_platform.api import app; assert '/predict' in app.openapi()['paths']"
docker run --rm delivery-eta:ci alembic heads
```

Sources: https://docs.astral.sh/uv/guides/integration/github/ and
https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions

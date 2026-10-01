# Continuous integration

The `CI` workflow runs on pushes, pull requests, and manual dispatch. It has two
independent checks on Ubuntu 24.04:

- **Python and PostgreSQL tests:** installs Python 3.12 and uv 0.12.19, synchronizes
  `uv.lock`, and runs all tests with `RUN_INTEGRATION=1`. The tests create synthetic
  models and an isolated PostgreSQL 18 container, apply migrations, and remove it.
- **Docker build and import check:** builds the image, imports the installed API
  and report module, and checks that Alembic can discover the migrations. This is
  an image packaging check, not a deployed HTTP smoke test.

Neither check needs your `.env`, dataset, saved models, or database. No image is
published and no application is deployed. The workflow has read-only repository
permissions and pins actions to commit IDs. uv and Python match the Docker setup.

## Activate the checks

The workflow runs once committed and pushed with the application files it tests.
Many project implementation files are currently untracked: pushing only the
workflow will not include them. Include source, tests, migrations, `uv.lock`,
`pyproject.toml`, `README.md`, `Dockerfile`, `.dockerignore`, `alembic.ini`, and
`compose.test.yaml` in the project commit. Review the staged files before pushing.
Do not add ignored credentials, raw datasets, model bundles, or local reports.

Open the repository's Actions tab after pushing. `CI` shows two job results;
open a failed step to read the error. A successful local run does not establish
that a GitHub-hosted run passed. Required branch checks are a separate repository
setting and have not been enabled by this file.

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

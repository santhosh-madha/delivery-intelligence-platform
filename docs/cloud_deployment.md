# Free cloud deployment preparation

Target: a single Render Free Docker web service and a Neon Free PostgreSQL database.
No cloud accounts, resources, releases, or deployments were created by preparation.
Local Docker Compose continues to use `Dockerfile`; Render uses `Dockerfile.cloud`.

## Prepared files

- `render.yaml`: explicitly selects `plan: free`, one service, Virginia region,
  and deploys after GitHub checks pass. No paid database or disk is provisioned.
- `Dockerfile.cloud`: bundles a checksum-verified model; runs migrations then one
  Uvicorn worker on Render's `PORT`.
- `deploy/model/manifest.json`: minimal serving metadata, without local paths,
  source snapshots, dataset records, or database credentials.
- `deploy/model/model.cbm`: local staging copy, excluded from Git. Publish this
  82 MB binary as a versioned release asset rather than adding it to Git history.

## First: publish code and the model asset

Review and commit the cloud preparation code, configuration, minimal manifest,
unit tests, and this guide. Do not add `.env` or the ignored model binary to Git.
Push the commit and confirm CI passes.

Create a GitHub release named `eta-v1` for the deployment commit and attach
`deploy/model/model.cbm`. The expected checksum is already pinned in the minimal
manifest. The public download URL would then be:

```
https://github.com/santhosh-madha/delivery-intelligence-platform/releases/download/eta-v1/model.cbm
```

This URL will not work until the release is actually published. The build uses
public HTTPS downloads: a private repository/private release needs a different
artifact access setup. Keep the versioned asset stable. Check model/dataset sharing
terms before publishing trained artifacts. No raw training data needs uploading.

## Create Neon Free

1. Create a Neon account and select the Free plan.
2. Create one PostgreSQL project in a region near Render Virginia.
3. Copy the **direct** PostgreSQL connection URL into your password manager. It is
   a secret; do not paste it into GitHub files or chat. Use the direct endpoint,
   not the pooled endpoint, because startup also runs migrations.
4. Keep paid upgrades disabled and review the displayed Free quotas.

The application accepts `DATABASE_URL` and uses Psycopg 3. It forces
`sslmode=verify-full` plus `sslrootcert=system`, checking the server certificate and
hostname even if the copied URL specifies weaker TLS. The cloud image sets
`SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt` so Psycopg's bundled OpenSSL
finds Debian's installed CA certificates. Local Compose retains its
existing POSTGRES_* settings. The cloud startup refuses missing cloud credentials.

## Create Render Free

1. Connect GitHub to Render and select **New → Blueprint** for this repository.
2. Inspect `render.yaml`: there must be only the Free web service. Choose no paid
   resources or disks. For a strict zero-cost demo, avoid adding a payment method.
3. Set the prompted `DATABASE_URL` secret to the Neon direct connection URL.
4. Set `MODEL_URL` to the published HTTPS release asset URL above.
5. Render generates `ETA_API_KEY`. Save it privately from the service environment
   settings after creation. Do not put it in a public frontend or README.
6. Deploy and inspect logs. The build downloads and verifies the model. Startup
   applies Alembic migrations before starting the API. Never run migration commands
   concurrently; this setup is intended for one service/worker.

Free Render has no pre-deploy job, so migration runs at startup. Future schema
changes must remain compatible with the previous API version during redeploys.
A failed migration prevents the new instance from starting; rollback of the image
is not rollback of the database schema.

## Verify the public service

- `/live`: model loaded; no database query. Render uses this for health checks.
- `/health`: verifies both model and database; use for a manual readiness check.
- `/docs`: public Swagger documentation. Use **Authorize**, enter the API key,
  then try the known example request.
- `POST /predict` and the outcome endpoint require `X-API-Key` in cloud mode.
  Missing/wrong keys return 401. The API refuses cloud startup with a missing or
  short API key. Existing local behavior is unchanged unless ETA_API_KEY is set.
- Predictions default to demo. Never mark synthetic inputs as real just to fill
  accuracy reports. Only submit measured real outcomes for real predictions.

Confirm the response matches local model predictions, record its prediction ID,
and verify the row in Neon's SQL editor. Test unauthorized requests return 401.
Allow a sleep/wake cycle and verify that model loading and prediction still work.
Do not expose database credentials or API keys in screenshots.

## Local checks completed / remaining

The automated suite checks existing behavior, API key rejection/acceptance, TLS
configuration enforcement, and checksum mismatch handling. A local cloud image
build validates the staged model. Actual Neon TLS connectivity, Render build-time
release download, wake-up behavior, and hosted memory use require real cloud
accounts and are not yet verified. The earlier 512 MiB test was local ARM64,
not Render's CPU/architecture. No paid capacity is enabled by these files.

The API key limits who can write; it is not per-user authorization, a rate limiter,
or a full abuse-protection system. This is a small private-access portfolio demo,
not a production public API. Free services can sleep or hit usage quotas.

## References

- https://render.com/docs/blueprint-spec
- https://render.com/docs/docker
- https://render.com/docs/free
- https://neon.com/blog/avoid-mitm-attacks-with-psql-postgres-16

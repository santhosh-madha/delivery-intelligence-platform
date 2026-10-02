# Cloud deployment

I deployed the API on Render and the PostgreSQL database on Neon using their Free
plans. The API is configured in Virginia; the database is in Ohio (`us-east-2`).
The prediction and outcome flow was verified on October 2, 2026.

- [API documentation](https://delivery-intelligence-api-2z5u.onrender.com/docs)
- [Readiness check](https://delivery-intelligence-api-2z5u.onrender.com/health)

The setup instructions below explain how to reproduce this deployment.
Local Docker Compose continues to use `Dockerfile`; Render uses `Dockerfile.cloud`.

## Deployment files

- `render.yaml`: explicitly selects `plan: free`, one service, Virginia region,
  and deploys after GitHub checks pass. No paid database or disk is provisioned.
- `Dockerfile.cloud`: bundles a checksum-verified model; runs migrations then one
  Uvicorn worker on Render's `PORT`.
- `deploy/model/manifest.json`: minimal serving metadata, without local paths,
  source snapshots, dataset records, or database credentials.
- `deploy/model/model.cbm`: local staging copy, excluded from Git. The
  82 MB binary is published as the `eta-v1` release asset, outside Git history.

## Published model

The [eta-v1 release](https://github.com/santhosh-madha/delivery-intelligence-platform/releases/tag/eta-v1)
contains the serving model. Its checksum matches `deploy/model/manifest.json`.
Set `MODEL_URL` to this public download URL:

```text
https://github.com/santhosh-madha/delivery-intelligence-platform/releases/download/eta-v1/model.cbm
```

The build downloads the model and verifies its checksum. No raw training data is
uploaded. For a future model version, publish a new versioned asset and update the
manifest and deployment settings together; keep the existing release stable.
A private release would need a different artifact access setup.

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

## Verified results and limits

On October 2, 2026, I verified the deployed service with a synthetic demo request:

| Check | Result |
|---|---|
| `/health` | `ready`, with the expected model version |
| Authorized prediction | 32.264926 minutes, matching local inference |
| Prediction persistence | Matching prediction ID stored in Neon as `demo` |
| Missing API key | Request rejected |
| Outcome recording | Simulated 35-minute duration; absolute error 2.735074 minutes |
| Outcome persistence | Stored outcome joined to the correct prediction |

The successful cloud deployment also exercised the release download, model
verification, database migrations, and Neon TLS connection. The first deployment
failed certificate verification; explicitly setting the certificate bundle path
in `Dockerfile.cloud` fixed it without disabling verification.

The cloud-preparation suite passed 61 tests locally. The model also loaded and
predicted in a local container limited to 512 MiB. This was an ARM64 check, not a
measurement of Render's peak memory. Sleep/wake behavior and sustained hosted load
have not been verified. Demo outcomes do not establish real-world model accuracy.

The API key limits who can write; it is not per-user authorization, a rate limiter,
or a full abuse-protection system. This is a small private-access portfolio demo,
not a production public API. Free services can sleep or hit usage quotas.

## References

- https://render.com/docs/blueprint-spec
- https://render.com/docs/docker
- https://render.com/docs/free
- https://neon.com/blog/avoid-mitm-attacks-with-psql-postgres-16

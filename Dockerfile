FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Runtime library used by numerical/ML packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately so Docker can reuse this layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-cache

# Install our application.
COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable --no-cache

# Include the database migration instructions.
COPY alembic.ini ./
COPY migrations ./migrations

# Run the application as an ordinary user.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "delivery_intelligence_platform.api:app", "--host", "0.0.0.0", "--port", "8000"]
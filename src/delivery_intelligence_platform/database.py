"""Database connection and prediction table definition."""

import os
from uuid import uuid4

from dotenv import dotenv_values
from sqlalchemy import (
    BigInteger,
    ForeignKey,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    MetaData,
    String,
    Table,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import URL


metadata = MetaData()

predictions = Table(
    "predictions",
    metadata,
    Column(
        "prediction_id",
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    ),
    Column("data_source", String(16), nullable=False, server_default="unknown"),
    CheckConstraint("data_source IN ('unknown', 'demo', 'real')",
                    name="ck_predictions_data_source"),
    Column(
        "input_payload",
        JSONB,
        nullable=False,
    ),
    Column(
        "predicted_duration_min",
        Double,
        nullable=False,
    ),
    Column(
        "estimated_arrival_unix_s",
        Double,
        nullable=False,
    ),
    Column(
        "model_version",
        String(128),
        nullable=False,
        index=True,
    ),
    CheckConstraint(
        "predicted_duration_min >= 0 "
        "AND predicted_duration_min < 'Infinity'::float8",
        name="ck_predictions_duration",
    ),
    CheckConstraint(
        "estimated_arrival_unix_s > 0 "
        "AND estimated_arrival_unix_s < 'Infinity'::float8",
        name="ck_predictions_arrival",
    ),
)


delivery_outcomes = Table(
    "delivery_outcomes", metadata,
    Column("prediction_id", UUID(as_uuid=True),
           ForeignKey("predictions.prediction_id"), primary_key=True),
    Column("actual_arrival_unix_s", BigInteger, nullable=False),
    Column("recorded_at", DateTime(timezone=True), nullable=False,
           server_default=func.now()),
    CheckConstraint(
        "actual_arrival_unix_s > 0 AND actual_arrival_unix_s < 9007199254740992",
        name="ck_delivery_outcomes_arrival",
    ),
)


def make_engine():
    # Read local settings; actual environment variables take precedence.
    settings = {
        **dotenv_values(".env"),
        **os.environ,
    }

    required = (
        "POSTGRES_DB",
        "POSTGRES_USER",
        "POSTGRES_PASSWORD",
    )
    missing = [name for name in required if not settings.get(name)]

    if missing:
        raise ValueError(
            f"Missing database settings: {', '.join(missing)}"
        )

    url = URL.create(
        drivername="postgresql+psycopg",
        username=settings["POSTGRES_USER"],
        password=settings["POSTGRES_PASSWORD"],
        host=settings.get("POSTGRES_HOST", "127.0.0.1"),
        port=int(settings.get("POSTGRES_PORT", "5432")),
        database=settings["POSTGRES_DB"],
    )

    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )
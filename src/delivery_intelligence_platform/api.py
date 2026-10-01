"""Serve the saved ETA model through HTTP."""

from uuid import UUID, uuid4

from sqlalchemy import insert, select
from sqlalchemy.exc import SQLAlchemyError

from .database import delivery_outcomes, make_engine, predictions
from sqlalchemy.dialects.postgresql import insert as pg_insert

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .features import build_features
from .predict import EtaPredictor


logger = logging.getLogger(__name__)

Timestamp = Annotated[
    int,
    Field(strict=True, gt=0, lt=2**53),
]

Identifier = Annotated[
    int,
    Field(strict=True, ge=0, lt=2**53),
]


class AcceptanceEvent(BaseModel):
    """Information available when a courier accepts an order."""

    model_config = ConfigDict(extra="forbid")

    platform_order_time: Timestamp
    order_push_time: Timestamp
    grab_time: Timestamp

    poi_id: Identifier
    da_id: Identifier
    courier_id: Identifier

    is_prebook: Annotated[
        int,
        Field(strict=True, ge=0, le=0),
    ] = 0

    @model_validator(mode="after")
    def validate_event(self):
        # Use the same validation as training and local prediction.
        build_features(pd.DataFrame([self.model_dump()]))
        return self


class PredictionResponse(BaseModel):
    prediction_id: UUID

    predicted_duration_min: float = Field(
        ge=0, allow_inf_nan=False
    )

    estimated_arrival_unix_s: float = Field(
        gt=0, allow_inf_nan=False
    )

    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str


class DeliveryOutcomeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actual_arrival_unix_s: Timestamp

    @model_validator(mode="after")
    def validate_arrival(self):
        if self.actual_arrival_unix_s > int(time.time()):
            raise ValueError("Actual arrival cannot be in the future.")
        return self


class DeliveryOutcomeResponse(BaseModel):
    prediction_id: UUID
    actual_arrival_unix_s: int
    actual_duration_min: float = Field(ge=0, allow_inf_nan=False)
    predicted_duration_min: float = Field(ge=0, allow_inf_nan=False)
    error_min: float = Field(allow_inf_nan=False)
    absolute_error_min: float = Field(ge=0, allow_inf_nan=False)
    model_version: str


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configured_path = model_dir or os.environ.get("ETA_MODEL_DIR")

        if not configured_path:
            raise RuntimeError(
                "Set ETA_MODEL_DIR to a saved model bundle directory."
            )

        predictor = EtaPredictor(
            Path(configured_path).expanduser().resolve()
        )

        engine = make_engine()

        try:
            # Check that PostgreSQL and the migrated table are available.
            with engine.connect() as connection:
                connection.execute(select(predictions).limit(0))
                connection.execute(select(delivery_outcomes).limit(0))

            app.state.predictor = predictor
            app.state.db_engine = engine

            logger.info(
                "Loaded model %s; database is ready.",
                predictor.manifest["model_version"],
            )

            yield
        finally:
            app.state.predictor = None
            app.state.db_engine = None
            engine.dispose()

    app = FastAPI(
        title="Delivery Intelligence API",
        description=(
            "Predict delivery duration at courier acceptance "
            "and save predictions to PostgreSQL."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    def get_predictor(request: Request) -> EtaPredictor:
        predictor = getattr(request.app.state, "predictor", None)

        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail="The prediction model is not ready.",
            )

        return predictor

    def get_engine(request: Request):
        engine = getattr(request.app.state, "db_engine", None)

        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="The database is not ready.",
            )

        return engine

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request):
        predictor = get_predictor(request)
        engine = get_engine(request)

        try:
            with engine.connect() as connection:
                connection.execute(select(predictions).limit(0))
                connection.execute(select(delivery_outcomes).limit(0))
        except SQLAlchemyError:
            logger.error("Database readiness check failed.")
            raise HTTPException(
                status_code=503,
                detail="The database is unavailable.",
            ) from None

        return HealthResponse(
            status="ready",
            model_version=predictor.manifest["model_version"],
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(
        event: AcceptanceEvent, request: Request,
        data_source: Literal["demo", "real"] = "demo",
    ):
        predictor = get_predictor(request)
        engine = get_engine(request)

        try:
            events = pd.DataFrame([event.model_dump()])
            result = predictor.predict(events).iloc[0]

            response = PredictionResponse(
                prediction_id=uuid4(),
                predicted_duration_min=float(
                    result["predicted_duration_min"]
                ),
                estimated_arrival_unix_s=float(
                    result["estimated_arrival_unix_s"]
                ),
                model_version=str(result["model_version"]),
            )
        except Exception:
            logger.exception("Prediction failed.")
            raise HTTPException(
                status_code=500,
                detail="Prediction failed.",
            ) from None

        try:
            # Commit the row before returning a successful response.
            with engine.begin() as connection:
                connection.execute(
                    insert(predictions),
                    {
                        "prediction_id": response.prediction_id,
                        "input_payload": event.model_dump(mode="json"),
                        "data_source": data_source,
                        "predicted_duration_min": (
                            response.predicted_duration_min
                        ),
                        "estimated_arrival_unix_s": (
                            response.estimated_arrival_unix_s
                        ),
                        "model_version": response.model_version,
                    },
                )
        except SQLAlchemyError:
            logger.error("Saving the prediction failed.")
            raise HTTPException(
                status_code=503,
                detail="Could not save the prediction.",
            ) from None

        return response

    @app.post(
        "/predictions/{prediction_id}/outcome",
        response_model=DeliveryOutcomeResponse,
        status_code=201,
    )
    def record_outcome(
        prediction_id: UUID, event: DeliveryOutcomeEvent, request: Request,
    ):
        engine = get_engine(request)
        try:
            with engine.begin() as connection:
                prediction = connection.execute(
                    select(predictions).where(
                        predictions.c.prediction_id == prediction_id
                    )
                ).mappings().one_or_none()
                if prediction is None:
                    raise HTTPException(404, "Prediction not found.")

                accepted_at = int(prediction["input_payload"]["grab_time"])
                arrived_at = event.actual_arrival_unix_s
                if arrived_at < accepted_at:
                    raise HTTPException(
                        422, "Actual arrival cannot precede courier acceptance."
                    )

                actual_duration = (arrived_at - accepted_at) / 60
                predicted_duration = float(prediction["predicted_duration_min"])
                error = predicted_duration - actual_duration
                response = DeliveryOutcomeResponse(
                    prediction_id=prediction_id,
                    actual_arrival_unix_s=arrived_at,
                    actual_duration_min=actual_duration,
                    predicted_duration_min=predicted_duration,
                    error_min=error,
                    absolute_error_min=abs(error),
                    model_version=prediction["model_version"],
                )
                inserted_id = connection.execute(
                    pg_insert(delivery_outcomes).values(
                        prediction_id=prediction_id,
                        actual_arrival_unix_s=arrived_at,
                    ).on_conflict_do_nothing(
                        index_elements=["prediction_id"]
                    ).returning(delivery_outcomes.c.prediction_id)
                ).scalar_one_or_none()
                if inserted_id is None:
                    raise HTTPException(
                        409, "An outcome already exists for this prediction."
                    )
            return response
        except SQLAlchemyError:
            logger.error("Saving the delivery outcome failed.")
            raise HTTPException(503, "Could not save the delivery outcome.") from None

    return app


app = create_app()
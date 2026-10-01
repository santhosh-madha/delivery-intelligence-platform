"""Test the API against a real, disposable PostgreSQL database."""

import json
import os
import time
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
import pytest
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from catboost import CatBoostRegressor
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, text

from delivery_intelligence_platform.api import create_app
from delivery_intelligence_platform.database import (
    delivery_outcomes,
    make_engine,
    predictions,
)
from delivery_intelligence_platform.features import (
    CATEGORICAL_FEATURES,
    CONTRACT_VERSION,
    FEATURE_COLUMNS,
    build_features,
)
from delivery_intelligence_platform.predict import EtaPredictor, sha256


ROOT = Path(__file__).resolve().parents[1]

# Docker tests run only when explicitly enabled.
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="Set RUN_INTEGRATION=1 to run PostgreSQL integration tests.",
)

COMPOSE = [
    "docker",
    "compose",
    "-p",
    "delivery-eta-integration",
    "-f",
    str(ROOT / "compose.test.yaml"),
]


def example_event():
    return {
        "platform_order_time": 1666407600,
        "order_push_time": 1666407660,
        "grab_time": 1666407720,
        "poi_id": 1,
        "da_id": 0,
        "courier_id": 2,
        "is_prebook": 0,
    }


@pytest.fixture(scope="module")
def integration_model(tmp_path_factory):
    """Create a small real model without using project artifacts."""
    bundle = tmp_path_factory.mktemp("integration_model")

    events = pd.DataFrame([
        {
            **example_event(),
            "poi_id": i % 2,
            "courier_id": i % 3,
            "grab_time": 1666407720 + i * 60,
        }
        for i in range(20)
    ])

    model = CatBoostRegressor(
        iterations=5,
        depth=2,
        random_seed=42,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    )

    model.fit(
        build_features(events),
        np.arange(20, dtype=float) + 10,
        cat_features=CATEGORICAL_FEATURES,
    )

    model_path = bundle / "model.cbm"
    model.save_model(str(model_path))

    manifest = {
        "feature_contract": CONTRACT_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "model_sha256": sha256(model_path),
        "model_version": "integration-test-model",
    }

    (bundle / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    return bundle


@pytest.fixture(scope="module")
def integration_database():
    """Start an isolated database and apply actual migrations."""
    test_settings = {
        "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "55432",
        "POSTGRES_DB": "delivery_integration_test",
        "POSTGRES_USER": "integration_user",
        "POSTGRES_PASSWORD": "integration_test_only",
    }

    # These changes are restored when the fixture finishes.
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(ROOT)

        for key, value in test_settings.items():
            patch.setenv(key, value)

        engine = None

        try:
            subprocess.run(
                COMPOSE + [
                    "up", "-d", "--wait",
                    "--wait-timeout", "60",
                    "--force-recreate",
                ],
                cwd=ROOT,
                check=True,
                timeout=120,
            )

            engine = make_engine()

            # Confirm the connection targets the dedicated test database.
            with engine.connect() as connection:
                database_name = connection.scalar(
                    text("SELECT current_database()")
                )
                assert database_name == "delivery_integration_test"

            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option(
                "script_location",
                str(ROOT / "migrations"),
            )

            command.upgrade(config, "head")

            # Confirm all repository migrations were applied.
            expected_heads = set(
                ScriptDirectory.from_config(config).get_heads()
            )
            with engine.connect() as connection:
                actual_heads = set(
                    MigrationContext.configure(
                        connection
                    ).get_current_heads()
                )

            assert actual_heads == expected_heads

            yield engine

        finally:
            if engine is not None:
                engine.dispose()

            subprocess.run(
                COMPOSE + ["down", "--volumes"],
                cwd=ROOT,
                check=True,
                timeout=60,
            )


@pytest.fixture
def integration_client(integration_database, integration_model):
    # Each test starts with an empty predictions table.
    with integration_database.begin() as connection:
        connection.execute(delete(delivery_outcomes))
        connection.execute(delete(predictions))

    with TestClient(create_app(integration_model)) as client:
        yield client


def count_predictions(engine):
    with engine.connect() as connection:
        return connection.scalar(
            select(func.count()).select_from(predictions)
        )


def test_health_with_real_database(integration_client):
    response = integration_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": "integration-test-model",
    }


def test_prediction_is_committed(
    integration_client,
    integration_database,
    integration_model,
):
    event = example_event()

    expected = EtaPredictor(integration_model).predict(
        pd.DataFrame([event])
    ).iloc[0]

    response = integration_client.post("/predict", json=event)

    assert response.status_code == 200
    body = response.json()
    prediction_id = UUID(body["prediction_id"])

    # Read using a separate connection to verify the API committed.
    with integration_database.connect() as connection:
        saved = connection.execute(
            select(predictions).where(
                predictions.c.prediction_id == prediction_id
            )
        ).mappings().one()

    assert count_predictions(integration_database) == 1
    assert saved["input_payload"] == event
    assert saved["created_at"].tzinfo is not None
    assert saved["model_version"] == body["model_version"]
    assert saved["model_version"] == "integration-test-model"

    assert saved["predicted_duration_min"] == (
        body["predicted_duration_min"]
    )
    assert saved["estimated_arrival_unix_s"] == (
        body["estimated_arrival_unix_s"]
    )

    assert body["predicted_duration_min"] == pytest.approx(
        expected["predicted_duration_min"],
        rel=1e-10,
    )
    assert body["estimated_arrival_unix_s"] == pytest.approx(
        expected["estimated_arrival_unix_s"],
        rel=0,
        abs=1e-6,
    )


def test_invalid_request_creates_no_record(
    integration_client,
    integration_database,
):
    before = count_predictions(integration_database)

    response = integration_client.post(
        "/predict",
        json={**example_event(), "grab_time": 0},
    )

    assert response.status_code == 422
    assert count_predictions(integration_database) == before

def count_outcomes(engine):
    with engine.connect() as connection:
        return connection.scalar(select(func.count()).select_from(delivery_outcomes))


def create_test_prediction(client):
    response = client.post("/predict", json=example_event())
    assert response.status_code == 200
    return response.json()


def test_outcome_is_saved_and_error_is_correct(integration_client, integration_database):
    prediction = create_test_prediction(integration_client)
    pid = prediction["prediction_id"]
    arrival = example_event()["grab_time"] + 40 * 60
    response = integration_client.post(
        f"/predictions/{pid}/outcome", json={"actual_arrival_unix_s": arrival}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["prediction_id"] == pid
    assert body["actual_arrival_unix_s"] == arrival
    assert body["actual_duration_min"] == 40
    assert body["predicted_duration_min"] == prediction["predicted_duration_min"]
    assert body["error_min"] == pytest.approx(prediction["predicted_duration_min"] - 40)
    assert body["absolute_error_min"] == pytest.approx(abs(body["error_min"]))
    assert body["model_version"] == prediction["model_version"]
    with integration_database.connect() as connection:
        saved = connection.execute(select(delivery_outcomes).where(
            delivery_outcomes.c.prediction_id == UUID(pid)
        )).mappings().one()
        original = connection.execute(select(predictions).where(
            predictions.c.prediction_id == UUID(pid)
        )).mappings().one()
    assert saved["actual_arrival_unix_s"] == arrival
    assert saved["recorded_at"].tzinfo is not None
    assert original["predicted_duration_min"] == prediction["predicted_duration_min"]
    assert count_outcomes(integration_database) == 1


def test_outcome_for_unknown_prediction_returns_404(integration_client, integration_database):
    response = integration_client.post(
        f"/predictions/{uuid4()}/outcome",
        json={"actual_arrival_unix_s": example_event()["grab_time"] + 2400},
    )
    assert response.status_code == 404
    assert count_outcomes(integration_database) == 0


@pytest.mark.parametrize("second_offset", [0, 60])
def test_repeated_outcome_does_not_overwrite(integration_client, integration_database, second_offset):
    prediction = create_test_prediction(integration_client)
    pid = prediction["prediction_id"]
    arrival = example_event()["grab_time"] + 2400
    endpoint = f"/predictions/{pid}/outcome"
    assert integration_client.post(endpoint, json={"actual_arrival_unix_s": arrival}).status_code == 201
    assert integration_client.post(endpoint, json={"actual_arrival_unix_s": arrival + second_offset}).status_code == 409
    with integration_database.connect() as connection:
        saved = connection.scalar(select(delivery_outcomes.c.actual_arrival_unix_s).where(
            delivery_outcomes.c.prediction_id == UUID(pid)
        ))
    assert saved == arrival
    assert count_outcomes(integration_database) == 1


@pytest.mark.parametrize("invalid_arrival", [0, True, 1666410120.5, 1666407719, 2**53, int(time.time()) + 3600])
def test_invalid_outcome_creates_no_record(integration_client, integration_database, invalid_arrival):
    prediction = create_test_prediction(integration_client)
    response = integration_client.post(
        f"/predictions/{prediction['prediction_id']}/outcome",
        json={"actual_arrival_unix_s": invalid_arrival},
    )
    assert response.status_code == 422
    assert count_outcomes(integration_database) == 0


def test_reporting_excludes_demo_unknown_and_keeps_pending(
    integration_client, integration_database,
):
    from sqlalchemy import insert
    from delivery_intelligence_platform.reporting import build_report

    # Default requests are demos; explicit real requests participate in metrics.
    demo = integration_client.post('/predict', json=example_event()).json()
    real = integration_client.post('/predict?data_source=real', json=example_event()).json()
    pending = integration_client.post('/predict?data_source=real', json={
        **example_event(), 'courier_id': 999999,
    })
    assert pending.status_code == 200
    for item in (demo, real):
        result = integration_client.post(
            f"/predictions/{item['prediction_id']}/outcome",
            json={'actual_arrival_unix_s': example_event()['grab_time'] + 3600},
        )
        assert result.status_code == 201
    # Old clients / pre-migration records default to unknown.
    with integration_database.begin() as connection:
        connection.execute(insert(predictions), {
            'input_payload': example_event(), 'predicted_duration_min': 10,
            'estimated_arrival_unix_s': 1666408320,
            'model_version': 'legacy-model',
        })
    report = build_report(integration_database, memberships={'integration-test-model': {2}})
    assert report['excluded'] == {'demo': 1, 'unknown': 1}
    assert report['overall']['predictions'] == 2
    assert report['overall']['outcomes'] == 1
    assert report['overall']['outcome_coverage'] == .5
    assert report['overall']['metrics']['mae_min'] == pytest.approx(abs(real['predicted_duration_min']-60))
    model = report['by_model']['integration-test-model']
    assert model['by_courier']['seen']['predictions'] == 1
    assert model['by_courier']['unseen']['predictions'] == 1
    assert model['by_actual_duration']['60_plus']['valid_outcomes'] == 1
    assert sum(x['predictions'] for x in model['by_day_utc'].values()) == 2
    unknown_report = build_report(integration_database)
    assert unknown_report['by_model']['integration-test-model']['by_courier']['unknown']['predictions'] == 2
    assert integration_client.post('/predict?data_source=unknown', json=example_event()).status_code == 422


def test_report_window_uses_prediction_and_outcome_availability(integration_client, integration_database):
    from datetime import datetime, timezone
    from sqlalchemy import insert
    from delivery_intelligence_platform.reporting import build_report
    def date(day):
        return datetime(2026, 1, day, tzinfo=timezone.utc)
    with integration_database.begin() as connection:
        for day in (1, 2, 3):
            pid = uuid4()
            connection.execute(insert(predictions), {
                'prediction_id': pid, 'created_at': date(day),
                'data_source': 'real', 'input_payload': example_event(),
                'predicted_duration_min': 30, 'estimated_arrival_unix_s': 1666409520,
                'model_version': 'window-model',
            })
            connection.execute(insert(delivery_outcomes), {
                'prediction_id': pid, 'actual_arrival_unix_s': 1666409520,
                'recorded_at': date(3),
            })
    report = build_report(integration_database, since=date(2), until=date(3))
    assert report['overall']['predictions'] == 1
    assert report['overall']['outcomes'] == 0
    assert report['overall']['metrics'] is None
    with pytest.raises(ValueError):
        build_report(integration_database, since=date(3), until=date(2))
    with pytest.raises(ValueError):
        build_report(integration_database, since=datetime(2026, 1, 1))

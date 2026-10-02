"""Test request validation and saved-model HTTP predictions."""

import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor
from fastapi.testclient import TestClient

from unittest.mock import MagicMock
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

import delivery_intelligence_platform.api as api_module


from delivery_intelligence_platform.api import create_app
from delivery_intelligence_platform.features import (
    CATEGORICAL_FEATURES,
    CONTRACT_VERSION,
    FEATURE_COLUMNS,
    build_features,
)
from delivery_intelligence_platform.predict import EtaPredictor, sha256


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
def model_bundle(tmp_path_factory):
    # A small synthetic model keeps tests independent of real artifacts.
    bundle = tmp_path_factory.mktemp("api_model")

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
        "model_version": "api-test-model",
    }

    (bundle / "manifest.json").write_text(json.dumps(manifest))

    return bundle


@pytest.fixture(autouse=True)
def fake_db_engine(monkeypatch):
    engine = MagicMock()

    monkeypatch.setattr(
        api_module,
        "make_engine",
        lambda: engine,
    )

    return engine

@pytest.fixture
def client(model_bundle):
    with TestClient(create_app(model_bundle)) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": "api-test-model",
    }


def test_api_matches_saved_model_prediction(client, model_bundle):
    event = example_event()

    expected = EtaPredictor(model_bundle).predict(
        pd.DataFrame([event])
    ).iloc[0]

    response = client.post("/predict", json=event)

    assert response.status_code == 200

    actual = response.json()

    assert actual["model_version"] == expected["model_version"]

    assert actual["predicted_duration_min"] == pytest.approx(
        expected["predicted_duration_min"],
        rel=1e-10,
    )

    assert actual["estimated_arrival_unix_s"] == pytest.approx(
        expected["estimated_arrival_unix_s"],
        rel=0,
        abs=1e-6,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"grab_time": 0},
        {"grab_time": 1666407500},
        {"grab_time": 1666407720.5},
        {"courier_id": True},
        {"courier_id": None},
        {"poi_id": -1},
        {"is_prebook": 1},
        {"unexpected_field": "value"},
    ],
)
def test_invalid_requests_return_422(client, change):
    response = client.post(
        "/predict",
        json={**example_event(), **change},
    )

    assert response.status_code == 422


def test_missing_field_returns_422(client):
    event = example_event()
    del event["order_push_time"]

    response = client.post("/predict", json=event)

    assert response.status_code == 422


def test_unseen_categories_return_valid_prediction(client):
    event = {
        **example_event(),
        "poi_id": 999999,
        "courier_id": 999999,
    }

    response = client.post("/predict", json=event)

    assert response.status_code == 200
    assert np.isfinite(response.json()["predicted_duration_min"])
    assert response.json()["predicted_duration_min"] >= 0


def test_missing_model_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / "missing-model")):
            pass



def test_prediction_saves_matching_record(client, fake_db_engine):
    event = example_event()

    response = client.post("/predict", json=event)

    assert response.status_code == 200
    body = response.json()

    connection = (
        fake_db_engine.begin.return_value.__enter__.return_value
    )
    connection.execute.assert_called_once()

    saved = connection.execute.call_args.args[1]

    assert saved["prediction_id"] == UUID(body["prediction_id"])
    assert saved["input_payload"] == event
    assert saved["model_version"] == body["model_version"]
    assert saved["predicted_duration_min"] == (
        body["predicted_duration_min"]
    )
    assert saved["estimated_arrival_unix_s"] == (
        body["estimated_arrival_unix_s"]
    )


def test_invalid_request_does_not_save(client, fake_db_engine):
    response = client.post(
        "/predict",
        json={**example_event(), "grab_time": 0},
    )

    assert response.status_code == 422
    fake_db_engine.begin.assert_not_called()


def test_commit_failure_returns_503(client, fake_db_engine):
    # A failure when exiting the transaction simulates commit failure.
    transaction = fake_db_engine.begin.return_value
    transaction.__exit__.side_effect = SQLAlchemyError(
        "Simulated commit failure"
    )

    response = client.post("/predict", json=example_event())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Could not save the prediction."
    }

@pytest.mark.parametrize('supplied', [None, 'wrong-key'])
def test_api_key_protects_both_write_endpoints(model_bundle, monkeypatch, fake_db_engine, supplied):
    monkeypatch.setenv('ETA_API_KEY', 'a'*40)
    headers = {} if supplied is None else {'X-API-Key': supplied}
    with TestClient(create_app(model_bundle)) as client:
        assert client.get('/live').status_code == 200
        assert client.get('/health').status_code == 200
        assert client.post('/predict', json=example_event(), headers=headers).status_code == 401
        assert client.post('/predictions/18ebf362-723f-45dc-9baf-aca229ca4e32/outcome',
                           json={'actual_arrival_unix_s':1666410120}, headers=headers).status_code == 401
    fake_db_engine.begin.assert_not_called()


def test_valid_api_key_allows_prediction(model_bundle, monkeypatch):
    monkeypatch.setenv('ETA_API_KEY', 'a'*40)
    with TestClient(create_app(model_bundle)) as client:
        assert client.post('/predict', json=example_event(), headers={'X-API-Key':'a'*40}).status_code == 200


def test_cloud_app_requires_key(model_bundle, monkeypatch):
    monkeypatch.setenv('ETA_ENV', 'cloud')
    monkeypatch.delenv('ETA_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='ETA_API_KEY'):
        with TestClient(create_app(model_bundle)):
            pass

import json

import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostRegressor

from delivery_intelligence_platform.data import chronological_split, prepare_cohort
from delivery_intelligence_platform.features import CATEGORICAL_FEATURES, CONTRACT_VERSION, FEATURE_COLUMNS, build_features
from delivery_intelligence_platform.predict import EtaPredictor, sha256


def event():
    return {"platform_order_time": 1666407600, "order_push_time": 1666407660,
            "grab_time": 1666407720, "poi_id": 1, "da_id": 0, "courier_id": 2}


def test_raw_event_features_are_identical_to_training_row():
    raw = event()
    training_row = {**raw, "arrive_time": 1666408800, "target_duration_min": 18, "order_id": 40}
    features = build_features(pd.DataFrame([raw], index=[9]))
    pd.testing.assert_frame_equal(features, build_features(pd.DataFrame([training_row], index=[9])))
    assert features.iloc[0].to_dict() == {
        "order_age_at_acceptance_min": 2.0, "queue_age_at_acceptance_min": 1.0,
        "poi_id": "1", "da_id": "0", "courier_id": "2", "acceptance_hour": "11",
    }


@pytest.mark.parametrize("change", [
    {"grab_time": 0}, {"grab_time": 1666407500}, {"grab_time": 1.5},
    {"courier_id": None}, {"courier_id": True}, {"poi_id": -1}, {"da_id": float("inf")},
])
def test_invalid_events_are_rejected(change):
    with pytest.raises(ValueError):
        build_features(pd.DataFrame([{**event(), **change}]))


def test_missing_inputs_are_rejected():
    row = event()
    del row["order_push_time"]
    with pytest.raises(ValueError, match="Missing"):
        build_features(pd.DataFrame([row]))


def test_cohort_keeps_accepted_attempt_and_reports_exclusions():
    accepted = {**event(), "order_id": 1, "waybill_id": 1, "is_prebook": 0,
                "is_courier_grabbed": 1, "arrive_time": 1666408800, "estimate_arrived_time": 1666408900}
    rejected = {**accepted, "waybill_id": 2, "is_courier_grabbed": 0, "grab_time": 0, "arrive_time": 0}
    missing_arrival = {**accepted, "order_id": 2, "waybill_id": 3, "arrive_time": 0}
    prebook = {**accepted, "order_id": 3, "waybill_id": 4, "is_prebook": 1}
    cohort, audit = prepare_cohort(pd.DataFrame([accepted, rejected, missing_arrival, prebook]))
    assert cohort.order_id.tolist() == [1]
    assert cohort.target_duration_min.iloc[0] == 18
    assert sum(row["excluded"] for row in audit) == 3


def test_split_excludes_labels_not_observed_before_boundary():
    s = int(pd.Timestamp("2022-10-22", tz="Asia/Shanghai").timestamp())
    e = s+86400
    cohort = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5],
        "grab_time": [s-100, s-50, s, e-20, e],
        "arrive_time": [s-1, s, s+60, e, e+60],
    })
    train, validation, report = chronological_split(cohort, "2022-10-22", "2022-10-23")
    assert train.order_id.tolist() == [1]
    assert validation.order_id.tolist() == [3]
    assert report["train_boundary_exclusions"] == report["validation_boundary_exclusions"] == 1
    assert report["later_records_not_evaluated"] == 1


def test_saved_model_roundtrip_and_unseen_categories(tmp_path):
    events = pd.DataFrame([{**event(), "poi_id": i % 2, "courier_id": i % 3,
                            "grab_time": event()["grab_time"] + i*60} for i in range(20)])
    X = build_features(events)
    model = CatBoostRegressor(iterations=5, depth=2, verbose=False, allow_writing_files=False, thread_count=1)
    model.fit(X, np.arange(20)+10, cat_features=CATEGORICAL_FEATURES)
    model.save_model(str(tmp_path/"model.cbm"))
    manifest = {"feature_contract": CONTRACT_VERSION, "feature_columns": FEATURE_COLUMNS,
                "model_sha256": sha256(tmp_path/"model.cbm"), "model_version": "test-run"}
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    predictor = EtaPredictor(tmp_path)
    np.testing.assert_allclose(predictor.predict(events).predicted_duration_min, model.predict(X))
    unseen = pd.DataFrame([{**event(), "courier_id": 999, "poi_id": 999}])
    assert np.isfinite(predictor.predict(unseen).predicted_duration_min).all()
    with pytest.raises(ValueError, match="on-demand"):
        predictor.predict(pd.DataFrame([{**event(), "is_prebook": 1}]))
    manifest["model_sha256"] = "wrong"
    (tmp_path/"manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum"):
        EtaPredictor(tmp_path)


def test_training_logs_finished_mlflow_run_and_artifacts(tmp_path):
    import mlflow
    from delivery_intelligence_platform.train import run_training

    records = []
    for i in range(40):
        s = int(pd.Timestamp("2022-10-21" if i < 20 else "2022-10-22", tz="Asia/Shanghai").timestamp()) + 36000 + i*60
        records.append({"platform_order_time": s-120, "order_push_time": s-60, "grab_time": s,
                        "arrive_time": s+600+i*5, "estimate_arrived_time": s+1800,
                        "order_id": i, "waybill_id": i, "poi_id": i%2, "da_id": 0,
                        "courier_id": i%3, "is_courier_grabbed": 1, "is_prebook": 0})
    source = tmp_path/"data.csv"
    pd.DataFrame(records).to_csv(source, index=False)
    bundle = run_training(source, tmp_path/"models", tmp_path/"tracking", tmp_path,
                          settings={"iterations": 3, "depth": 2, "thread_count": 1})
    manifest = json.loads((bundle/"manifest.json").read_text())
    assert manifest["split"]["train_rows"] == manifest["split"]["validation_rows"] == 20
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(manifest["model_version"])
    assert run.info.status == "FINISHED"
    assert "validation_mae_min" in run.data.metrics
    artifacts = client.list_artifacts(run.info.run_id, "bundle")
    assert any(item.path == "bundle/model.cbm" for item in artifacts)
    assert EtaPredictor(bundle).predict(pd.DataFrame([event()])).shape == (1, 3)

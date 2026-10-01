"""Reproducible CatBoost training with local MLflow experiment tracking."""

import argparse
import importlib.metadata
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import mlflow
import numpy as np
from catboost import CatBoostRegressor

from .data import chronological_split, load_cohort
from .features import CATEGORICAL_FEATURES, CONTRACT_VERSION, FEATURE_COLUMNS, INPUT_COLUMNS, build_features
from .predict import EtaPredictor, sha256

MODEL_SETTINGS = {
    "iterations": 300, "depth": 6, "learning_rate": 0.08, "loss_function": "RMSE",
    "l2_leaf_reg": 3.0, "random_seed": 42, "thread_count": 4,
    "task_type": "CPU", "allow_writing_files": False, "verbose": False,
}


def metrics(actual, predicted):
    a, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if a.shape != p.shape or a.size == 0 or not np.isfinite(a).all() or not np.isfinite(p).all():
        raise ValueError("Metrics require nonempty, finite, aligned observations.")
    error = p-a
    return {
        "rows": int(a.size), "mae_min": float(np.abs(error).mean()),
        "rmse_min": float(np.sqrt(np.mean(error**2))), "bias_min": float(error.mean()),
        "median_absolute_error_min": float(np.median(np.abs(error))),
        "p90_absolute_error_min": float(np.quantile(np.abs(error), .9)),
    }


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def source_provenance(project: Path) -> dict:
    def git(*args):
        result = subprocess.run(["git", "-C", str(project), *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain")}


def run_training(data: Path, output_root: Path, tracking_root: Path, project: Path,
                 validation_start="2022-10-22", validation_end="2022-10-23",
                 experiment="delivery-eta-v1", settings=None) -> Path:
    """Create a new immutable run bundle; never overwrite a previous run."""
    data, output_root, tracking_root = data.resolve(), output_root.resolve(), tracking_root.resolve()
    params = MODEL_SETTINGS.copy() if settings is None else {**MODEL_SETTINGS, **settings}
    print("Loading source data and checking eligibility...", flush=True)
    fingerprint = sha256(data)
    cohort, audit = load_cohort(data)
    train, validation, split = chronological_split(cohort, validation_start, validation_end)
    X_train, X_validation = build_features(train), build_features(validation)
    y_train, y_validation = train.target_duration_min, validation.target_duration_min
    print(f"Training: {len(train):,} orders; validation: {len(validation):,} orders", flush=True)

    tracking_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_root / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    existing = client.get_experiment_by_name(experiment)
    experiment_id = existing.experiment_id if existing else client.create_experiment(
        experiment, artifact_location=(tracking_root / "run_artifacts").as_uri()
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"catboost-{validation_start}") as run:
        version = run.info.run_id
        bundle = output_root / version
        bundle.mkdir(parents=True, exist_ok=False)
        mlflow.log_params({**params, "feature_contract": CONTRACT_VERSION,
                           "validation_start": validation_start, "validation_end": validation_end})
        mlflow.set_tags({"target": "acceptance_to_arrival_minutes", "population": "on_demand_completed",
                         "data_sha256": fingerprint, "release_status": "development_candidate"})
        started = perf_counter()
        model = CatBoostRegressor(**params, cat_features=CATEGORICAL_FEATURES)
        model.fit(X_train, y_train)
        fit_seconds = perf_counter()-started
        raw = model.predict(X_validation)
        predictions = np.maximum(raw, 0)
        results = {
            "train": metrics(y_train, np.maximum(model.predict(X_train), 0)),
            "validation": metrics(y_validation, predictions),
            "median_baseline": metrics(y_validation, np.full(len(validation), float(y_train.median()))),
            "fit_seconds": fit_seconds, "negative_raw_predictions": int((raw < 0).sum()),
        }
        known = validation.courier_id.isin(set(train.courier_id)).to_numpy()
        results["courier_segments"] = {
            label: metrics(y_validation.to_numpy()[mask], predictions[mask])
            for label, mask in [("seen", known), ("unseen", ~known)] if mask.any()
        }
        promise_valid = validation.estimate_arrived_time.gt(0)
        if promise_valid.any():
            results["promise_baseline"] = metrics(y_validation[promise_valid], validation.loc[promise_valid, "promise_remaining_min"])
        results["promise_baseline_exclusions"] = int((~promise_valid).sum())

        model.save_model(str(bundle / "model.cbm"))
        versions = {name: importlib.metadata.version(name) for name in ["catboost", "pandas", "numpy", "scikit-learn", "mlflow"]}
        source_folder = bundle / "source_snapshot"
        source_folder.mkdir()
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, source_folder / path.name)
        for name in ["pyproject.toml", "uv.lock"]:
            if (project/name).exists():
                shutil.copy2(project/name, source_folder/name)
        manifest = {
            "model_version": version, "feature_contract": CONTRACT_VERSION,
            "feature_columns": FEATURE_COLUMNS, "categorical_features": CATEGORICAL_FEATURES,
            "input_columns": INPUT_COLUMNS, "model_sha256": sha256(bundle/"model.cbm"),
            "source_data_sha256": fingerprint, "source_data_path": str(data),
            "source_code_sha256": {p.name: sha256(p) for p in source_folder.iterdir()},
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "packages": versions,
            "settings": params, "split": split, "tracking_uri": tracking_uri,
            "release_status": "development_candidate", "prediction_floor_min": 0,
            "limitations": ["Only on-demand accepted/completed orders", "Eight source dates in one city",
                            "Validation omits outcomes completed after period end", "Unseen-courier performance is weaker",
                            "Not evaluated for deployment or approved for production"],
            **source_provenance(project),
        }
        dump_json(bundle/"manifest.json", manifest)
        dump_json(bundle/"metrics.json", results)
        dump_json(bundle/"data_audit.json", {"eligibility": audit, "split": split})
        # Verify the real saved artifact through the serving path, with raw event inputs.
        sample = validation.iloc[:64]
        reloaded = EtaPredictor(bundle).predict(sample).predicted_duration_min.to_numpy()
        np.testing.assert_allclose(reloaded, predictions[:len(sample)], rtol=1e-10, atol=1e-10)
        mlflow.log_metrics({f"validation_{k}": v for k, v in results["validation"].items() if k != "rows"})
        mlflow.log_metrics({"train_rows": len(train), "validation_rows": len(validation), "fit_seconds": fit_seconds,
                            "median_baseline_mae_min": results["median_baseline"]["mae_min"]})
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
    print(json.dumps({"bundle": str(bundle), "model_version": version, "validation": results["validation"],
                      "mlflow_tracking_uri": tracking_uri}, indent=2))
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tracking-root", type=Path)
    parser.add_argument("--validation-start", default="2022-10-22")
    parser.add_argument("--validation-end", default="2022-10-23")
    args = parser.parse_args()
    project = args.project_root.resolve()
    run_training(
        args.data or project/"data/raw/meituan/all_waybill_info_meituan_0322.csv",
        args.output_root or project/"artifacts/models",
        args.tracking_root or project/"artifacts/mlflow", project,
        args.validation_start, args.validation_end,
    )


if __name__ == "__main__":
    main()

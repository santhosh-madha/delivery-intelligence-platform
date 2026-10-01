"""Load one immutable model bundle and predict from acceptance events."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .features import CONTRACT_VERSION, FEATURE_COLUMNS, build_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EtaPredictor:
    """Load once, reuse for many requests. Only on-demand orders are supported."""

    def __init__(self, bundle: str | Path):
        self.bundle = Path(bundle)
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())
        if self.manifest.get("feature_contract") != CONTRACT_VERSION:
            raise ValueError("Model feature-contract version is incompatible with this code.")
        if self.manifest.get("feature_columns") != FEATURE_COLUMNS:
            raise ValueError("Model feature columns do not match this code.")
        model_path = self.bundle / "model.cbm"
        if sha256(model_path) != self.manifest["model_sha256"]:
            raise ValueError("Model checksum does not match the manifest.")
        self.model = CatBoostRegressor()
        self.model.load_model(str(model_path))
        if self.model.feature_names_ != FEATURE_COLUMNS:
            raise ValueError("Saved model feature order is incompatible.")

    def predict(self, events: pd.DataFrame) -> pd.DataFrame:
        if "is_prebook" in events and not events.is_prebook.eq(0).all():
            raise ValueError("This model supports on-demand orders only (is_prebook=0).")
        features = build_features(events)
        raw = self.model.predict(features)
        if not np.isfinite(raw).all():
            raise ValueError("Model returned non-finite predictions.")
        duration = np.maximum(raw, 0.0)
        return pd.DataFrame({
            "predicted_duration_min": duration,
            "estimated_arrival_unix_s": pd.to_numeric(events.grab_time).to_numpy() + duration*60,
            "model_version": self.manifest["model_version"],
        }, index=events.index)

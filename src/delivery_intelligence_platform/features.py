"""The acceptance-time feature contract shared by training and prediction."""

import numpy as np
import pandas as pd

CONTRACT_VERSION = "acceptance_eta_v1"
TIMEZONE = "Asia/Shanghai"
INPUT_COLUMNS = [
    "platform_order_time", "order_push_time", "grab_time",
    "poi_id", "da_id", "courier_id",
]
FEATURE_COLUMNS = [
    "order_age_at_acceptance_min", "queue_age_at_acceptance_min",
    "poi_id", "da_id", "courier_id", "acceptance_hour",
]
CATEGORICAL_FEATURES = ["poi_id", "da_id", "courier_id", "acceptance_hour"]


def integer_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Validate identifiers and Unix seconds without silently rounding inputs."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    if frame.empty:
        raise ValueError("At least one record is required.")
    values = frame[columns].copy()
    for col in columns:
        if values[col].map(lambda value: isinstance(value, (bool, np.bool_))).any():
            raise ValueError(f"{col} must contain integers, not booleans.")
        try:
            numeric = pd.to_numeric(values[col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{col} must contain integers.") from exc
        numbers = numeric.to_numpy(dtype=float, na_value=np.nan)
        if (not np.isfinite(numbers).all() or (numbers < 0).any()
                or (numbers >= 2**53).any() or (numbers % 1 != 0).any()):
            raise ValueError(f"{col} must contain finite, nonnegative integers below 2**53.")
        values[col] = numeric.astype("int64")
    return values


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    values = integer_columns(frame, INPUT_COLUMNS)
    times = values[["platform_order_time", "order_push_time", "grab_time"]]
    if times.le(0).any().any():
        raise ValueError("Creation, push, and acceptance times must be positive Unix seconds.")
    if (values.platform_order_time > values.order_push_time).any() or (
        values.order_push_time > values.grab_time
    ).any():
        raise ValueError("Expected creation <= push <= acceptance.")
    try:
        local = pd.to_datetime(values.grab_time, unit="s", utc=True).dt.tz_convert(TIMEZONE)
    except (ValueError, OverflowError) as exc:
        raise ValueError("Acceptance timestamp is outside the supported date range.") from exc
    features = pd.DataFrame(index=frame.index)
    features["order_age_at_acceptance_min"] = (values.grab_time - values.platform_order_time) / 60
    features["queue_age_at_acceptance_min"] = (values.grab_time - values.order_push_time) / 60
    for col in ["poi_id", "da_id", "courier_id"]:
        features[col] = values[col].astype(str)
    features["acceptance_hour"] = local.dt.hour.astype(str)
    return features[FEATURE_COLUMNS]

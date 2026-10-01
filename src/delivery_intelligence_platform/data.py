"""Read-only source loading, audited eligibility, and chronological partitions."""

from pathlib import Path

import pandas as pd

from .features import INPUT_COLUMNS, TIMEZONE, integer_columns

REQUIRED_COLUMNS = list(dict.fromkeys(INPUT_COLUMNS + [
    "order_id", "waybill_id", "is_courier_grabbed", "is_prebook", "arrive_time",
    "estimate_arrived_time",
]))


def prepare_cohort(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    validated = integer_columns(frame, REQUIRED_COLUMNS)
    for flag in ["is_courier_grabbed", "is_prebook"]:
        if not validated[flag].isin([0, 1]).all():
            raise ValueError(f"{flag} must be 0 or 1.")
    if not validated.waybill_id.is_unique:
        raise ValueError("waybill_id must be unique.")
    rules = [
        ("accepted", validated.is_courier_grabbed.eq(1)),
        ("on_demand", validated.is_prebook.eq(0)),
        ("positive_acceptance", validated.grab_time.gt(0)),
        ("positive_arrival", validated.arrive_time.gt(0)),
        ("arrival_after_acceptance", validated.arrive_time.gt(validated.grab_time)),
        ("positive_creation_and_push", validated.platform_order_time.gt(0) & validated.order_push_time.gt(0)),
        ("ordered_inputs", validated.platform_order_time.le(validated.order_push_time)
         & validated.order_push_time.le(validated.grab_time)),
    ]
    keep = pd.Series(True, index=frame.index)
    audit = [{"step": "all_waybills", "remaining_rows": len(frame), "excluded": 0}]
    for name, rule in rules:
        before = int(keep.sum())
        keep &= rule
        audit.append({"step": name, "remaining_rows": int(keep.sum()), "excluded": before-int(keep.sum())})
    cohort = validated.loc[keep].copy()
    if cohort.empty or not cohort.order_id.is_unique:
        raise ValueError("Eligible cohort must be nonempty with one accepted record per order.")
    cohort["target_duration_min"] = (cohort.arrive_time-cohort.grab_time)/60
    cohort["promise_remaining_min"] = (cohort.estimate_arrived_time-cohort.grab_time)/60
    return cohort, audit


def load_cohort(path: Path) -> tuple[pd.DataFrame, list[dict]]:
    return prepare_cohort(pd.read_csv(path))


def chronological_split(cohort: pd.DataFrame, validation_start: str, validation_end: str):
    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    if start.tzinfo is not None or end.tzinfo is not None:
        raise ValueError("Use local dates/times without an offset; timezone is Asia/Shanghai.")
    start = start.tz_localize(TIMEZONE)
    end = end.tz_localize(TIMEZONE)
    if start >= end:
        raise ValueError("Validation end must be later than validation start.")
    s, e = int(start.timestamp()), int(end.timestamp())
    train_candidates = cohort.grab_time.lt(s)
    val_candidates = cohort.grab_time.ge(s) & cohort.grab_time.lt(e)
    train_mask = train_candidates & cohort.arrive_time.lt(s)
    val_mask = val_candidates & cohort.arrive_time.lt(e)
    train, validation = cohort.loc[train_mask].copy(), cohort.loc[val_mask].copy()
    if train.empty or validation.empty:
        raise ValueError("Training and validation must both contain records.")
    if set(train.order_id) & set(validation.order_id):
        raise ValueError("Orders overlap across training and validation.")
    report = {
        "timezone": TIMEZONE, "validation_start": start.isoformat(), "validation_end": end.isoformat(),
        "train_rows": len(train), "validation_rows": len(validation),
        "train_boundary_exclusions": int((train_candidates & ~train_mask).sum()),
        "validation_boundary_exclusions": int((val_candidates & ~val_mask).sum()),
        "later_records_not_evaluated": int(cohort.grab_time.ge(e).sum()),
        "order_overlap": 0,
    }
    return train, validation, report

"""Read-only performance reporting for explicitly real prediction requests."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import and_, select

from .database import delivery_outcomes, make_engine, predictions
from .data import chronological_split, load_cohort
from .predict import sha256


def summarize(rows):
    """Coverage uses all requests; errors use only valid matched outcomes."""
    matched = [r for r in rows if r['actual_arrival_unix_s'] is not None]
    valid = []
    for row in matched:
        actual = (row['actual_arrival_unix_s'] - row['input_payload']['grab_time']) / 60
        predicted = row['predicted_duration_min']
        if actual >= 0 and np.isfinite([actual, predicted]).all() and predicted >= 0:
            valid.append((actual, predicted))
    result = {
        'predictions': len(rows), 'outcomes': len(matched),
        'outcome_coverage': len(matched) / len(rows) if rows else None,
        'valid_outcomes': len(valid), 'invalid_outcomes': len(matched)-len(valid),
        'metrics': None,
    }
    if valid:
        a, p = np.asarray(valid).T
        error = p-a
        result['metrics'] = {
            'mae_min': float(abs(error).mean()),
            'rmse_min': float(np.sqrt((error**2).mean())),
            'bias_min': float(error.mean()),
            'p90_absolute_error_min': float(np.quantile(abs(error), .9)),
            'actual_mean_min': float(a.mean()),
            'predicted_mean_min': float(p.mean()),
        }
    return result


def training_couriers(model_root, version):
    """Reconstruct membership only from verified data and recorded split."""
    if Path(version).name != version or version in ('.', '..'):
        return None, 'Invalid model version path'
    bundle = model_root / version
    if not (bundle / 'manifest.json').exists():
        return None, 'Model manifest unavailable'
    m = json.loads((bundle / 'manifest.json').read_text())
    if m['model_version'] != version:
        return None, 'Model manifest version mismatch'
    source = Path(m['source_data_path'])
    if not source.exists():
        return None, 'Training source unavailable'
    if sha256(source) != m['source_data_sha256']:
        return None, 'Training source fingerprint mismatch'
    cohort, _ = load_cohort(source)
    start = pd.Timestamp(m['split']['validation_start']).tz_localize(None).isoformat()
    end = pd.Timestamp(m['split']['validation_end']).tz_localize(None).isoformat()
    train, _, split = chronological_split(cohort, start, end)
    if split['train_rows'] != m['split']['train_rows']:
        return None, 'Reconstructed training row count mismatch'
    return set(train.courier_id.astype(int)), None


def build_report(engine, *, since=None, until=None, model_root=None, memberships=None):
    until = until or datetime.now(timezone.utc)
    if until.utcoffset() is None or (since is not None and since.utcoffset() is None):
        raise ValueError('Report boundaries must include a timezone.')
    if since is not None and since >= until:
        raise ValueError('Start must precede end.')
    statement = select(
        predictions,
        delivery_outcomes.c.actual_arrival_unix_s,
    ).select_from(predictions.outerjoin(
        delivery_outcomes,
        and_(predictions.c.prediction_id == delivery_outcomes.c.prediction_id,
             delivery_outcomes.c.recorded_at < until),
    )).where(predictions.c.created_at < until)
    if since is not None:
        statement = statement.where(predictions.c.created_at >= since)
    with engine.connect() as connection:
        rows = [dict(r) for r in connection.execute(statement).mappings()]
    eligible = [r for r in rows if r['data_source'] == 'real']
    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'window': {'since': since.isoformat() if since else None, 'until': until.isoformat()},
        'window_basis': 'Prediction creation time; start inclusive, end exclusive. Outcomes available before end.',
        'excluded': {source: sum(r['data_source'] == source for r in rows) for source in ('demo', 'unknown')},
        'overall': summarize(eligible), 'by_model': {},
        'limitations': [
            'Metrics describe only explicitly real requests with observed outcomes.',
            'Pending outcomes can bias error estimates, especially for slow deliveries.',
            'Duration segments use actual outcomes and cannot identify slow orders in advance.',
            'Caller-supplied provenance is not independent verification of a real delivery.',
            'This local report loads the selected window into memory.',
        ],
    }
    for version in sorted({r['model_version'] for r in eligible}):
        group = [r for r in eligible if r['model_version'] == version]
        known = (memberships or {}).get(version)
        reason = None
        if known is None:
            if model_root is not None:
                known, reason = training_couriers(Path(model_root), version)
            else:
                reason = 'Training membership not supplied'
        model = {'summary': summarize(group), 'courier_membership_note': reason,
                 'by_courier': {}, 'by_actual_duration': {}, 'by_day_utc': {}}
        for label in ('seen', 'unseen', 'unknown'):
            selected = [r for r in group if (
                'unknown' if known is None else
                'seen' if int(r['input_payload']['courier_id']) in known else 'unseen'
            ) == label]
            model['by_courier'][label] = summarize(selected)
        for label, lo, hi in [('under_15', 0, 15), ('15_to_30', 15, 30),
                              ('30_to_45', 30, 45), ('45_to_60', 45, 60),
                              ('60_plus', 60, float('inf'))]:
            selected = [r for r in group if r['actual_arrival_unix_s'] is not None
                        and lo <= (r['actual_arrival_unix_s']-r['input_payload']['grab_time'])/60 < hi]
            model['by_actual_duration'][label] = summarize(selected)
        days = sorted({r['created_at'].astimezone(timezone.utc).date().isoformat() for r in group})
        for day in days:
            model['by_day_utc'][day] = summarize([r for r in group if
                r['created_at'].astimezone(timezone.utc).date().isoformat() == day])
        report['by_model'][version] = model
    return report


def render_markdown(report):
    summary = report['overall']
    def value(x):
        return 'N/A' if x is None else f'{x:.2f}'
    def line(label, item):
        m = item['metrics'] or {}
        coverage = item['outcome_coverage']
        return (f"| {label} | {item['predictions']} | {item['valid_outcomes']} | "
                f"{value(coverage*100 if coverage is not None else None)} | "
                f"{value(m.get('mae_min'))} | {value(m.get('bias_min'))} | "
                f"{value(m.get('p90_absolute_error_min'))} |")
    header = ['| Group | Requests | Valid outcomes | Coverage % | MAE min | Bias min | P90 error min |',
              '|---|---:|---:|---:|---:|---:|---:|']
    lines = ['# Delivery performance report', '',
             f"Generated: {report['generated_at_utc']}", '',
             f"Window: {report['window']['since'] or 'all history'} to {report['window']['until']}",
             report['window_basis'], '',
             f"Excluded: {report['excluded']['demo']} demo requests; {report['excluded']['unknown']} unknown requests.", '',
             'Negative bias means the model underestimated duration.', '', *header, line('Overall', summary)]
    if not summary['valid_outcomes']:
        lines += ['', '**No eligible real outcomes yet; accuracy is unavailable, not zero.**']
    lines += ['', f"Invalid matched outcomes excluded from errors: {summary['invalid_outcomes']}"]
    for version, model in report['by_model'].items():
        lines += ['', f'## Model {version}', '', *header, line('All', model['summary'])]
        for section in ('by_courier', 'by_actual_duration', 'by_day_utc'):
            lines += ['', f'### {section}', '', *header]
            lines += [line(label, item) for label, item in model[section].items()]
        if model['courier_membership_note']:
            lines += ['', model['courier_membership_note']]
    lines += ['', '## Interpretation limits', ''] + [f'- {s}' for s in report['limitations']]
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--since', type=datetime.fromisoformat)
    parser.add_argument('--until', type=datetime.fromisoformat)
    parser.add_argument('--model-root', type=Path, default=Path('artifacts/models'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts/reports'))
    args = parser.parse_args()
    engine = make_engine()
    try:
        report = build_report(engine, since=args.since, until=args.until, model_root=args.model_root)
    finally:
        engine.dispose()
    output = args.output_dir / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output/'performance.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    (output/'performance.md').write_text(render_markdown(report))
    print(json.dumps({'report': str((output/'performance.md').resolve()), 'overall': report['overall'], 'excluded': report['excluded']}, indent=2))


if __name__ == '__main__':
    main()

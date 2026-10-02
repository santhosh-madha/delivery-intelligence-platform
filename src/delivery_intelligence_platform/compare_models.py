"""Fixed-configuration rolling comparison; never promotes a serving model."""
import argparse
import importlib.metadata
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from .data import load_cohort, chronological_split
from .features import build_features, CATEGORICAL_FEATURES, FEATURE_COLUMNS
from .predict import sha256
from .train import MODEL_SETTINGS, metrics

DAYS = ['2022-10-19', '2022-10-20', '2022-10-21', '2022-10-22']
RF_SETTINGS = dict(n_estimators=100, max_depth=16, min_samples_leaf=20,
                   max_features=0.5, n_jobs=4, random_state=42)


def make_model(name):
    if name == 'CatBoost':
        return CatBoostRegressor(**MODEL_SETTINGS, cat_features=CATEGORICAL_FEATURES)
    pre = ColumnTransformer([
        ('numeric', StandardScaler(), FEATURE_COLUMNS[:2]),
        ('ids', OneHotEncoder(handle_unknown='ignore', min_frequency=20,
                             sparse_output=True, dtype=np.float64),
         ['poi_id', 'da_id', 'courier_id']),
        ('hour', OneHotEncoder(handle_unknown='ignore', sparse_output=True,
                              dtype=np.float64), ['acceptance_hour']),
    ], sparse_threshold=1.0)
    if name == 'Ridge':
        estimator = Ridge(alpha=10, solver='lsqr', tol=1e-4, max_iter=2000)
    elif name == 'Random forest':
        estimator = RandomForestRegressor(**RF_SETTINGS)
    else:
        raise ValueError(name)
    return Pipeline([('preprocess', pre), ('model', estimator)])


def latency(model, raw, repeats):
    """Raw dataframe through features, model preprocessing, prediction and floor."""
    for _ in range(3):
        np.maximum(model.predict(build_features(raw)), 0)
    elapsed = []
    for _ in range(repeats):
        start = perf_counter()
        np.maximum(model.predict(build_features(raw)), 0)
        elapsed.append((perf_counter()-start)*1000)
    return {'rows': len(raw), 'repeats': repeats,
            'median_ms': float(np.median(elapsed)),
            'p95_ms': float(np.quantile(elapsed, .95))}


def run(project, output):
    bundle = project/'artifacts/models/87a38ec8f34d4ed0a47437a3b1e120ee'
    manifest = json.loads((bundle/'manifest.json').read_text())
    data = Path(manifest['source_data_path'])
    assert sha256(data) == manifest['source_data_sha256'], 'Source dataset changed'
    output.mkdir(parents=True, exist_ok=False)
    cohort, audit = load_cohort(data)
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'days': DAYS, 'features': FEATURE_COLUMNS,
        'source_data_sha256': manifest['source_data_sha256'],
        'random_forest_settings': RF_SETTINGS, 'catboost_settings': MODEL_SETTINGS,
        'ridge_settings': {'alpha': 10, 'solver': 'lsqr', 'tol': 1e-4, 'max_iter': 2000},
        'preprocessing': 'Ridge/RF: standardized numeric, sparse one-hot IDs min_frequency=20, ungrouped categorical hour; fitted on training only. CatBoost: native categories.',
        'selection': 'Lowest pooled out-of-fold MAE; all predictions floored at zero.',
        'timing': '4 worker/thread cap; fit includes model-specific preprocessing, excludes shared feature builder. Warm latency includes shared feature builder and fitted preprocessing, excludes HTTP/database/serialization. 30 single-row and 10 batch-of-256 repeats per fold.',
        'host': platform.platform(),
        'packages': {p: importlib.metadata.version(p) for p in ['scikit-learn','catboost','pandas','numpy']},
        'limitations': ['Fixed configurations, not an exhaustive tuning competition.',
                       'Previously examined validation periods, not an untouched final test.',
                       'Validation excludes deliveries completed after each day ends.',
                       'Timing is local and subject to load and thermal effects; no production SLA.',
                       'Same information features, model-appropriate categorical representations.'],
        'eligibility': audit,
    }
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copy2(Path(__file__), output/'comparison_source.py')
    results, audits, predictions = [], [], []
    for day in DAYS:
        end = (pd.Timestamp(day)+pd.Timedelta(days=1)).date().isoformat()
        train, val, split = chronological_split(cohort, day, end)
        audits.append(split)
        X, V = build_features(train), build_features(val)
        y, a = train.target_duration_min, val.target_duration_min.to_numpy()
        known = val.courier_id.isin(train.courier_id).to_numpy()
        # Rotate execution order deterministically to avoid one model always running last.
        names = ['Ridge', 'Random forest', 'CatBoost']
        shift = DAYS.index(day) % 3
        names = names[shift:] + names[:shift]
        for name in ['Training median'] + names:
            print(f'{day} {name}: {len(train):,} training / {len(val):,} validation', flush=True)
            if name == 'Training median':
                start = perf_counter(); median = float(y.median()); fit = perf_counter()-start
                p = np.full(len(a), median); one = batch = None
            else:
                model = make_model(name)
                start = perf_counter(); model.fit(X, y); fit = perf_counter()-start
                if name == 'Ridge' and np.max(model['model'].n_iter_) >= 2000:
                    raise RuntimeError('Ridge iteration limit reached')
                raw = model.predict(V)
                if not np.isfinite(raw).all(): raise ValueError('Nonfinite predictions')
                p = np.maximum(raw, 0)
                sample = val.sample(n=min(256, len(val)), random_state=42)
                one = latency(model, sample.iloc[:1], 30)
                batch = latency(model, sample, 10)
                del model
            result = {'day':day, 'model':name, 'fit_seconds':fit,
                      **metrics(a,p), 'single_latency':one, 'batch_latency':batch,
                      'couriers':{label:metrics(a[mask],p[mask]) for label,mask in [('seen',known),('unseen',~known)] if mask.any()},
                      'duration':{label:metrics(a[mask],p[mask]) for label,mask in [('under_15',a<15),('15_to_30',(a>=15)&(a<30)),('30_to_45',(a>=30)&(a<45)),('45_plus',a>=45)] if mask.any()}}
            results.append(result)
            predictions.append(pd.DataFrame({'order_id':val.order_id.to_numpy(), 'day':day,
                'model':name, 'actual':a, 'predicted':p, 'seen_courier':known}))
            (output/'daily_results.json').write_text(json.dumps(results,indent=2))
            print(f"  MAE {result['mae_min']:.4f}; fit {fit:.2f}s", flush=True)
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_parquet(output/'validation_predictions.parquet', index=False)
    summary=[]
    for name, group in all_predictions.groupby('model'):
        folds=[r for r in results if r['model']==name]
        item={'model':name, **metrics(group.actual,group.predicted),
              'total_fit_seconds':sum(r['fit_seconds'] for r in folds),
              'worst_day_mae_min':max(r['mae_min'] for r in folds)}
        if name!='Training median':
            item['single_median_ms']=float(np.median([r['single_latency']['median_ms'] for r in folds]))
            item['batch256_median_ms']=float(np.median([r['batch_latency']['median_ms'] for r in folds]))
        item['couriers']={str(label):metrics(g.actual,g.predicted) for label,g in group.groupby('seen_courier')}
        summary.append(item)
    summary.sort(key=lambda r:r['mae_min'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'split_audit.json').write_text(json.dumps(audits,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=Path.cwd())
    args=parser.parse_args()
    project=args.project_root.resolve()
    output=project/'artifacts/comparisons'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    print(f'Output: {output}',flush=True)
    with threadpool_limits(limits=4):
        run(project,output)


if __name__=='__main__': main()

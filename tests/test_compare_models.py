import numpy as np
import pandas as pd
import pytest

from delivery_intelligence_platform.compare_models import make_model
from delivery_intelligence_platform.features import FEATURE_COLUMNS


@pytest.mark.parametrize('name', ['Ridge', 'Random forest', 'CatBoost'])
def test_models_accept_unseen_categories_without_fitting_validation(name):
    train = pd.DataFrame({
        'order_age_at_acceptance_min': np.arange(40)/10,
        'queue_age_at_acceptance_min': np.arange(40)/20,
        'poi_id': ['a']*20+['b']*20,
        'da_id': ['x']*40,
        'courier_id': ['c']*40,
        'acceptance_hour': ['11']*40,
    })[FEATURE_COLUMNS]
    validation = train.iloc[:2].copy()
    validation['courier_id']='validation-only'
    model=make_model(name)
    if name=='Random forest': model.set_params(model__n_estimators=3, model__n_jobs=1)
    if name=='CatBoost': model.set_params(iterations=3, thread_count=1)
    model.fit(train, np.arange(40)+10)
    result=model.predict(validation)
    assert result.shape==(2,)
    assert np.isfinite(result).all()
    if name!='CatBoost':
        encoder=model['preprocess'].named_transformers_['ids']
        assert 'validation-only' not in encoder.categories_[2]
        assert model['preprocess'].transform(validation).shape[1] == model['preprocess'].transform(train).shape[1]

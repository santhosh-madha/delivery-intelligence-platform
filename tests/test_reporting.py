from datetime import datetime, timezone

import pytest

from delivery_intelligence_platform.reporting import summarize, render_markdown, training_couriers


def row(actual, prediction):
    return {'actual_arrival_unix_s': None if actual is None else 1000+60*actual,
            'input_payload': {'grab_time': 1000}, 'predicted_duration_min': prediction}


def test_metrics_and_missing_outcome_denominator():
    result = summarize([row(20, 30), row(60, 40), row(None, 35)])
    assert result['predictions'] == 3
    assert result['outcome_coverage'] == pytest.approx(2/3)
    assert result['metrics']['mae_min'] == 15
    assert result['metrics']['bias_min'] == -5
    assert result['metrics']['rmse_min'] == pytest.approx(250**.5)
    assert result['metrics']['p90_absolute_error_min'] == 19


def test_empty_and_unlabeled_do_not_imply_zero_error():
    assert summarize([])['outcome_coverage'] is None
    assert summarize([row(None, 10)])['metrics'] is None
    assert summarize([row(None, 10)])['outcome_coverage'] == 0


def test_invalid_outcome_is_counted_but_not_scored():
    result = summarize([row(-1, 10), row(20, 30)])
    assert result['invalid_outcomes'] == 1
    assert result['valid_outcomes'] == 1
    assert result['metrics']['mae_min'] == 10


def test_missing_membership_is_not_called_unseen(tmp_path):
    known, reason = training_couriers(tmp_path, 'not-here')
    assert known is None
    assert 'unavailable' in reason
    assert training_couriers(tmp_path, '../escape')[0] is None

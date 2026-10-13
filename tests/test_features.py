"""Tests for the feature engineering and the model pipeline."""
import numpy as np
import pandas as pd

from src.features import FraudFeatureBuilder, signed_log1p
from tests.conftest import make_tiny_dataset
from src.pipeline import build_pipeline


def _row(**overrides):
    row = {"step": 1, "type": "TRANSFER", "amount": 100.0, "oldbalanceOrg": 500.0,
           "newbalanceOrig": 400.0, "oldbalanceDest": 50.0, "newbalanceDest": 150.0}
    row.update(overrides)
    return pd.DataFrame([row])


def test_balance_errors_are_zero_for_consistent_transaction():
    out = FraudFeatureBuilder().transform(_row())
    assert out.loc[0, "errorBalanceOrig"] == 0
    assert out.loc[0, "errorBalanceDest"] == 0


def test_balance_error_detects_mismatch():
    out = FraudFeatureBuilder().transform(_row(newbalanceOrig=500.0))
    assert out.loc[0, "errorBalanceOrig"] == -100.0


def test_hour_and_night_flag():
    out = FraudFeatureBuilder().transform(pd.concat([_row(step=1), _row(step=13), _row(step=25)]))
    assert out["hour_of_day"].tolist() == [0.0, 12.0, 0.0]
    assert out["is_night"].tolist() == [1, 0, 1]


def test_origin_emptied_flag():
    out = FraudFeatureBuilder().transform(_row(oldbalanceOrg=100.0, newbalanceOrig=0.0))
    assert out.loc[0, "origin_emptied"] == 1


def test_signed_log_keeps_sign_and_zero():
    result = signed_log1p([-9.0, 0.0, 9.0])
    assert result[0] < 0 and result[1] == 0 and result[2] > 0


def test_pipeline_handles_unseen_category_without_crashing():
    X, y = make_tiny_dataset()
    pipeline = build_pipeline("decision_tree", "class_weight").fit(X, y)
    new_row = _row(type="NEW_TYPE")
    assert pipeline.predict(new_row).shape == (1,)


def test_pipeline_predictions_are_probabilities():
    X, y = make_tiny_dataset()
    pipeline = build_pipeline("logistic_regression", "class_weight").fit(X, y)
    probabilities = pipeline.predict_proba(X)[:, 1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))

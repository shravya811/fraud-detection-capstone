"""Test fixtures.

The tests need a model file, but the real trained model is too large for Git and does not
exist on the CI server. So the tests train a TINY model on a few made-up rows that have the
same 7 columns. This checks the API behaviour only - it says nothing about real performance.
"""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import RAW_FEATURES, TRANSACTION_TYPES
from src.pipeline import build_pipeline


def make_tiny_dataset(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "step": rng.integers(1, 744, n),
        "type": rng.choice(TRANSACTION_TYPES, n),
        "amount": rng.uniform(0, 5e5, n),
        "oldbalanceOrg": rng.uniform(0, 1e6, n),
        "newbalanceOrig": rng.uniform(0, 1e6, n),
        "oldbalanceDest": rng.uniform(0, 1e6, n),
        "newbalanceDest": rng.uniform(0, 1e6, n),
    })
    y = (df["newbalanceOrig"] < 1e5).astype(int).to_numpy()  # simple made-up rule
    return df[RAW_FEATURES], y


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    X, y = make_tiny_dataset()
    pipeline = build_pipeline("decision_tree", "class_weight")
    pipeline.fit(X, y)
    path = tmp_path_factory.mktemp("model") / "tiny_pipeline.pkl"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture()
def client(model_path, monkeypatch):
    from api.main import app
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(model_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client_without_model(tmp_path, monkeypatch):
    from api.main import app
    monkeypatch.setenv("FRAUD_MODEL_PATH", str(tmp_path / "does_not_exist.pkl"))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_payload():
    return {
        "step": 1, "type": "TRANSFER", "amount": 181.0, "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0,
    }

"""API tests: valid input, missing fields, wrong types, out-of-range values, unseen categories."""
import pytest


def test_health_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_when_model_missing(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False


def test_predict_valid_input(client, valid_payload):
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "label", "fraud_probability"}
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("fraud", "legitimate")
    assert 0.0 <= body["fraud_probability"] <= 1.0


@pytest.mark.parametrize("missing_field", [
    "step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest",
    "newbalanceDest",
])
def test_predict_missing_field(client, valid_payload, missing_field):
    del valid_payload[missing_field]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == missing_field


@pytest.mark.parametrize("field, bad_value", [
    ("amount", "not a number"),
    ("step", "first"),
    ("oldbalanceOrg", None),
    ("newbalanceDest", [1, 2]),
])
def test_predict_wrong_data_type(client, valid_payload, field, bad_value):
    valid_payload[field] = bad_value
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


@pytest.mark.parametrize("field, bad_value", [
    ("amount", -5.0),
    ("step", 0),
    ("step", 100000),
    ("oldbalanceOrg", -1.0),
    ("newbalanceDest", 1e12),
])
def test_predict_out_of_range(client, valid_payload, field, bad_value):
    valid_payload[field] = bad_value
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_predict_unseen_category(client, valid_payload):
    valid_payload["type"] = "BITCOIN"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["field"] == "type"


def test_predict_unknown_extra_field_rejected(client, valid_payload):
    valid_payload["is_fraud"] = 1
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_validation_errors_do_not_echo_input(client, valid_payload):
    valid_payload["type"] = "SECRET_VALUE_123"
    response = client.post("/predict", json=valid_payload)
    assert "SECRET_VALUE_123" not in response.text


def test_predict_returns_503_without_model(client_without_model, valid_payload):
    response = client_without_model.post("/predict", json=valid_payload)
    assert response.status_code == 503


def test_azure_route_returns_503_when_not_configured(client, valid_payload, monkeypatch):
    monkeypatch.delenv("AZURE_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("AZURE_ENDPOINT_KEY", raising=False)
    response = client.post("/predict/azure", json=valid_payload)
    assert response.status_code == 503

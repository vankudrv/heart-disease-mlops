"""API tests. They are skipped automatically if FastAPI/httpx are not installed."""
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from src.api.schemas import EXAMPLE_PATIENT  # noqa: E402


@pytest.fixture()
def client(model_path):
    os.environ["MODEL_PATH"] = str(model_path)
    from src.api.main import app
    with TestClient(app) as test_client:        # the `with` runs the startup (model loading)
        yield test_client


@pytest.fixture()
def client_without_model(tmp_path):
    os.environ["MODEL_PATH"] = str(tmp_path / "does_not_exist.joblib")
    from src.api.main import app
    with TestClient(app) as test_client:
        yield test_client


def test_health_is_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_prediction_and_confidence(client):
    response = client.post("/predict", json=EXAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_predict_works_without_optional_ca_and_thal(client):
    payload = {k: v for k, v in EXAMPLE_PATIENT.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_missing_field(client):
    payload = {k: v for k, v in EXAMPLE_PATIENT.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("field,value", [
    ("age", 500), ("sex", 2), ("cp", 9), ("thal", 5), ("oldpeak", -1), ("chol", "high"),
])
def test_predict_rejects_invalid_values(client, field, value):
    payload = dict(EXAMPLE_PATIENT, **{field: value})
    assert client.post("/predict", json=payload).status_code == 422


def test_model_info_endpoint(client):
    assert client.get("/model-info").status_code == 200


def test_metrics_endpoint_exposes_prometheus_metrics(client):
    client.post("/predict", json=EXAMPLE_PATIENT)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "predictions_total" in text
    assert "api_request_latency_seconds" in text


def test_health_and_predict_report_503_when_model_missing(client_without_model):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=EXAMPLE_PATIENT).status_code == 503

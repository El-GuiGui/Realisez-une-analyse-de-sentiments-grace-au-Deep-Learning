from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_predict_endpoint_basic():
    payload = {"text": "I love this airline"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "label" in data
    assert "label_str" in data
    assert "proba" in data

    assert data["label"] in (0, 1)
    assert data["label_str"] in ("negative", "positive")
    assert 0.0 <= data["proba"] <= 1.0


def test_feedback_endpoint_incorrect_prediction():
    payload = {
        "text": "Worst flight ever",
        "prediction": 1,
        "proba": 0.9,
        "is_correct": False,
    }
    response = client.post("/feedback", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "received"

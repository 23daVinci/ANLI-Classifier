from unittest.mock import patch
from models.prediction import PredictionResponse

_MOCK_RESPONSE = PredictionResponse(
    label="entailment",
    confidence=0.95,
    probabilities={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
    inference_time_ms=45.2,
    model="base",
    routed_to_llm=False,
    deberta_label="entailment",
    deberta_time_ms=45.2,
)


def test_predict_503_without_model(client):
    response = client.post("/v1/predict", json={"premise": "The sky is blue.", "hypothesis": "The sky has color."})
    assert response.status_code == 503


def test_predict_returns_valid_response(client, loaded_model):
    with patch("routers.predict.predict_single", return_value=_MOCK_RESPONSE):
        response = client.post(
            "/v1/predict", json={"premise": "The sky is blue.", "hypothesis": "The sky has color."}
        )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "entailment"
    assert data["confidence"] == 0.95
    assert data["routed_to_llm"] is False


def test_predict_missing_premise(client):
    response = client.post("/v1/predict", json={"hypothesis": "The sky has color."})
    assert response.status_code == 422


def test_batch_predict_503_without_model(client):
    response = client.post("/v1/predict/batch", json={"pairs": [{"premise": "A", "hypothesis": "B"}]})
    assert response.status_code == 503


def test_batch_predict_rejects_over_64(client, loaded_model):
    pairs = [{"premise": "A", "hypothesis": "B"}] * 65
    response = client.post("/v1/predict/batch", json={"pairs": pairs})
    assert response.status_code == 422


def test_batch_predict_rejects_empty(client, loaded_model):
    response = client.post("/v1/predict/batch", json={"pairs": []})
    assert response.status_code == 422

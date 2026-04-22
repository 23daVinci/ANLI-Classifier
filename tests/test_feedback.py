import pytest


@pytest.fixture
def tmp_feedback(tmp_path, monkeypatch):
    path = str(tmp_path / "feedback.json")
    from core.config import settings
    monkeypatch.setattr(settings, "feedback_file", path)
    yield path


def test_submit_feedback_correct(client, tmp_feedback):
    response = client.post(
        "/v1/feedback",
        json={"premise": "The sky is blue.", "hypothesis": "The sky has color.", "predicted_label": "entailment", "is_correct": True},
    )
    assert response.status_code == 200
    data = response.json()
    assert "feedback_id" in data
    assert "Thank you" in data["message"]


def test_submit_feedback_with_correction(client, tmp_feedback):
    response = client.post(
        "/v1/feedback",
        json={
            "premise": "A", "hypothesis": "B",
            "predicted_label": "entailment", "is_correct": False, "correct_label": "contradiction",
        },
    )
    assert response.status_code == 200
    assert "contradiction" in response.json()["message"]


def test_submit_feedback_invalid_label(client):
    response = client.post(
        "/v1/feedback",
        json={"premise": "A", "hypothesis": "B", "predicted_label": "entailment", "is_correct": False, "correct_label": "nonsense"},
    )
    assert response.status_code == 400


def test_feedback_stats_empty(client, tmp_path, monkeypatch):
    from core.config import settings
    monkeypatch.setattr(settings, "feedback_file", str(tmp_path / "none.json"))
    response = client.get("/v1/feedback/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["accuracy"] is None


def test_feedback_export(client, tmp_feedback):
    response = client.get("/v1/feedback/export")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "entries" in data

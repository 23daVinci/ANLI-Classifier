def test_liveness_always_200(client):
    response = client.get("/v1/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_readiness_503_without_model(client):
    response = client.get("/v1/health/ready")
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "not_ready"
    assert data["model_loaded"] is False


def test_readiness_200_with_model(client, loaded_model):
    response = client.get("/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    assert data["active_model"] == "base"

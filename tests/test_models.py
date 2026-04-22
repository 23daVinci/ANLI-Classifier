def test_list_models_returns_both(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    keys = {m["key"] for m in data["models"]}
    assert keys == {"base", "large"}


def test_list_models_active_field(client, loaded_model):
    response = client.get("/v1/models")
    data = response.json()
    active = [m for m in data["models"] if m["active"]]
    assert len(active) == 1
    assert active[0]["key"] == "base"


def test_switch_unknown_model(client):
    response = client.post("/v1/models/switch", json={"model": "xlarge"})
    assert response.status_code == 400
    assert "xlarge" in response.json()["detail"]


def test_switch_already_active(client, loaded_model):
    response = client.post("/v1/models/switch", json={"model": "base"})
    assert response.status_code == 200
    data = response.json()
    assert data["load_time_seconds"] == 0.0
    assert "already active" in data["message"]

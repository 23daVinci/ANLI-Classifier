import pytest
from fastapi.testclient import TestClient
from main import app
from services.model_service import state as model_state
from services.llm_service import state as llm_state


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def loaded_model(client):
    """Patches in a mock loaded model so routes don't return 503."""
    from unittest.mock import MagicMock

    original = (model_state.model, model_state.tokenizer, model_state.active_model_key, model_state.remap)

    model_state.model = MagicMock()
    model_state.tokenizer = MagicMock()
    model_state.active_model_key = "base"
    model_state.remap = {0: 0, 1: 1, 2: 2}

    yield model_state

    model_state.model, model_state.tokenizer, model_state.active_model_key, model_state.remap = original

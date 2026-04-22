import time
from fastapi import APIRouter, HTTPException
from models.registry import ModelInfo, ModelsResponse, SwitchRequest, SwitchResponse
from services.model_service import state as model_state, get_available_models, load_model
from services.llm_service import state as llm_state
from core.config import settings, MODEL_REGISTRY

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=ModelsResponse)
def list_models():
    available = get_available_models()
    models_list = [
        ModelInfo(
            key=key,
            name=info["name"],
            params=info["params"],
            description=info["description"],
            downloaded=info["downloaded"],
            active=(key == model_state.active_model_key),
        )
        for key, info in available.items()
    ]
    return ModelsResponse(
        active_model=model_state.active_model_key,
        models=models_list,
        hybrid_available=llm_state.client is not None,
        llm_model=settings.llm_model if llm_state.client is not None else None,
    )


@router.post("/models/switch", response_model=SwitchResponse)
def switch_model(request: SwitchRequest):
    if request.model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Choose from: {list(MODEL_REGISTRY.keys())}",
        )
    if request.model == model_state.active_model_key:
        return SwitchResponse(
            message=f"Model '{request.model}' is already active.",
            active_model=model_state.active_model_key,
            load_time_seconds=0.0,
        )

    start = time.perf_counter()
    try:
        load_model(request.model)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return SwitchResponse(
        message=f"Switched to '{request.model}' successfully.",
        active_model=model_state.active_model_key,
        load_time_seconds=round(time.perf_counter() - start, 2),
    )

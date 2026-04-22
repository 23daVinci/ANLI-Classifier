from fastapi import APIRouter, Response
from models.health import LivenessResponse, ReadinessResponse
from services.model_service import state as model_state
from services.llm_service import state as llm_state
from core.config import settings

router = APIRouter(prefix="/v1", tags=["health"])


@router.get("/health/live", response_model=LivenessResponse)
def liveness():
    return LivenessResponse(status="alive")


@router.get("/health/ready", response_model=ReadinessResponse)
def readiness(response: Response):
    ready = model_state.model is not None
    if not ready:
        response.status_code = 503
    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        model_loaded=ready,
        active_model=model_state.active_model_key,
        device=settings.device,
        model_dir=settings.model_dir,
        hybrid_available=llm_state.client is not None,
        llm_model=settings.llm_model if llm_state.client is not None else None,
    )

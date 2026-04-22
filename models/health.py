from pydantic import BaseModel


class LivenessResponse(BaseModel):
    status: str  # always "alive"


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "not_ready"
    model_loaded: bool
    active_model: str | None
    device: str
    model_dir: str
    hybrid_available: bool
    llm_model: str | None

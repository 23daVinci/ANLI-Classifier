from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    key: str
    name: str
    params: str
    description: str
    downloaded: bool
    active: bool


class ModelsResponse(BaseModel):
    active_model: str | None
    models: list[ModelInfo]
    hybrid_available: bool
    llm_model: str | None


class SwitchRequest(BaseModel):
    model: str = Field(..., examples=["large"], description="Model key: 'base' or 'large'")


class SwitchResponse(BaseModel):
    message: str
    active_model: str
    load_time_seconds: float

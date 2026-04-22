from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    premise: str = Field(..., min_length=1, examples=["The Parma trolleybus system comprises four urban routes."])
    hypothesis: str = Field(..., min_length=1, examples=["The trolleybus system has over 2 urban routes."])
    hybrid: bool = Field(False, description="Route low-confidence predictions to reasoning LLM")
    confidence_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Override default confidence threshold (default: 0.90)"
    )


class PredictionResponse(BaseModel):
    label: str = Field(..., examples=["entailment"])
    confidence: float = Field(..., examples=[0.95])
    probabilities: dict = Field(..., examples=[{"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}])
    inference_time_ms: float = Field(..., examples=[45.2])
    model: str = Field(..., examples=["base"])
    routed_to_llm: bool = Field(False)
    llm_model: str | None = Field(None)
    llm_reasoning: str | None = Field(None)
    deberta_label: str | None = Field(None)
    deberta_time_ms: float | None = Field(None)
    llm_label: str | None = Field(None)
    llm_time_ms: float | None = Field(None)
    confidence_threshold: float | None = Field(None)


class BatchRequest(BaseModel):
    pairs: list[PredictionRequest] = Field(..., min_length=1, max_length=64)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_inference_time_ms: float
    model: str

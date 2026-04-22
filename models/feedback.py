from pydantic import BaseModel, Field


class FeedbackRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    hypothesis: str = Field(..., min_length=1)
    predicted_label: str = Field(..., description="The label the system predicted")
    is_correct: bool = Field(..., description="Whether the prediction was correct")
    correct_label: str | None = Field(None, description="User-provided correct label (if is_correct=False)")
    confidence: float | None = Field(None)
    model: str | None = Field(None)
    routed_to_llm: bool = Field(False)
    llm_label: str | None = Field(None)


class FeedbackResponse(BaseModel):
    message: str
    feedback_id: str


class FeedbackStatsResponse(BaseModel):
    total: int
    correct: int
    incorrect: int
    accuracy: float | None
    corrections_by_label: dict
    corrections_by_model: dict
    avg_confidence_correct: float | None
    avg_confidence_incorrect: float | None

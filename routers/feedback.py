import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from models.feedback import FeedbackRequest, FeedbackResponse, FeedbackStatsResponse
from services.feedback_service import append_feedback, load_feedback, compute_stats
from core.config import LABEL_NAMES

router = APIRouter(prefix="/v1", tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    if request.correct_label and request.correct_label not in LABEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid correct_label '{request.correct_label}'. Must be one of: {LABEL_NAMES}",
        )

    entry = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "premise": request.premise,
        "hypothesis": request.hypothesis,
        "predicted_label": request.predicted_label,
        "is_correct": request.is_correct,
        "correct_label": request.correct_label,
        "confidence": request.confidence,
        "model": request.model,
        "routed_to_llm": request.routed_to_llm,
        "llm_label": request.llm_label,
    }

    append_feedback(entry)

    message = (
        "Thank you! Feedback recorded."
        if request.is_correct
        else f"Feedback recorded. Correct label: {request.correct_label or 'not provided'}."
    )
    return FeedbackResponse(message=message, feedback_id=entry["id"])


@router.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats():
    return FeedbackStatsResponse(**compute_stats(load_feedback()))


@router.get("/feedback/export")
def export_feedback():
    entries = load_feedback()
    return {
        "count": len(entries),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }

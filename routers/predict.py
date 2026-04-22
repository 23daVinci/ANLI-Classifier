import time
import logging
from fastapi import APIRouter, HTTPException
from models.prediction import PredictionRequest, PredictionResponse, BatchRequest, BatchResponse
from services.model_service import state as model_state
from services.inference_service import predict_single

router = APIRouter(prefix="/v1", tags=["predictions"])
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info(f"Predict: hybrid={request.hybrid}, threshold={request.confidence_threshold}")
    return predict_single(
        request.premise,
        request.hypothesis,
        hybrid=request.hybrid,
        confidence_threshold=request.confidence_threshold,
    )


@router.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if model_state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    predictions = [
        predict_single(
            pair.premise,
            pair.hypothesis,
            hybrid=pair.hybrid,
            confidence_threshold=pair.confidence_threshold,
        )
        for pair in request.pairs
    ]
    total_ms = (time.perf_counter() - start) * 1000

    return BatchResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_ms, 2),
        model=model_state.active_model_key,
    )

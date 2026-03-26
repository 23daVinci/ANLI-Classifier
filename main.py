"""
FastAPI inference server for ANLI Round 2 NLI classification.

Model: DeBERTa-v3-base fine-tuned on MNLI + Fever-NLI + ANLI
       (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli)

Endpoints:
    GET  /           → API info and documentation links
    GET  /health     → Health check (model loaded & ready)
    POST /predict    → NLI prediction for a premise-hypothesis pair
    POST /predict/batch → Batch NLI predictions
"""

import os
import time
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2Tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model / tokenizer (loaded once at startup)
# ---------------------------------------------------------------------------
model = None
tokenizer = None

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_model():
    """Load model and tokenizer from MODEL_DIR."""
    global model, tokenizer

    logger.info(f"Loading model from {MODEL_DIR} on {DEVICE}...")

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            f"Mount your best_model/ folder to this path."
        )

    # Tokenizer — try AutoTokenizer first, fall back to explicit vocab_file
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        logger.info("Loaded tokenizer via AutoTokenizer")
    except Exception:
        spm_path = os.path.join(MODEL_DIR, "spm.model")
        if os.path.exists(spm_path):
            tokenizer = DebertaV2Tokenizer(vocab_file=spm_path, do_lower_case=False)
            logger.info("Loaded tokenizer via DebertaV2Tokenizer (spm.model)")
        else:
            raise RuntimeError(
                f"Could not load tokenizer from {MODEL_DIR}. "
                f"Ensure tokenizer files or spm.model are present."
            )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()

    logger.info(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} params)")


# ---------------------------------------------------------------------------
# Lifespan: load model on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ANLI R2 NLI Classifier",
    description=(
        "3-way Natural Language Inference classifier (entailment / neutral / contradiction) "
        "using DeBERTa-v3-base fine-tuned on MNLI + Fever-NLI + ANLI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    premise: str = Field(..., min_length=1, examples=["The Parma trolleybus system comprises four urban routes."])
    hypothesis: str = Field(..., min_length=1, examples=["The trolleybus system has over 2 urban routes."])

class PredictionResponse(BaseModel):
    label: str = Field(..., examples=["entailment"])
    confidence: float = Field(..., examples=[0.95])
    probabilities: dict = Field(..., examples=[{"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}])
    inference_time_ms: float = Field(..., examples=[45.2])

class BatchRequest(BaseModel):
    pairs: list[PredictionRequest] = Field(..., min_length=1, max_length=64)

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_dir: str


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------
def predict_single(premise: str, hypothesis: str) -> PredictionResponse:
    """Run inference on a single premise-hypothesis pair."""
    start = time.perf_counter()

    inputs = tokenizer(
        premise,
        hypothesis,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
    pred_idx = int(torch.argmax(outputs.logits, dim=-1).item())

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        label=LABEL_MAP[pred_idx],
        confidence=round(max(probs), 4),
        probabilities={
            LABEL_MAP[i]: round(p, 4) for i, p in enumerate(probs)
        },
        inference_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "ANLI R2 NLI Classifier",
        "model": "DeBERTa-v3-base-mnli-fever-anli",
        "task": "3-way NLI (entailment / neutral / contradiction)",
        "endpoints": {
            "/predict": "POST — classify a premise-hypothesis pair",
            "/predict/batch": "POST — classify up to 64 pairs",
            "/health": "GET — health check",
            "/docs": "GET — interactive Swagger UI",
        },
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        device=DEVICE,
        model_dir=MODEL_DIR,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predict_single(request.premise, request.hypothesis)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    predictions = [
        predict_single(pair.premise, pair.hypothesis)
        for pair in request.pairs
    ]
    total_ms = (time.perf_counter() - start) * 1000

    return BatchResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_ms, 2),
    )
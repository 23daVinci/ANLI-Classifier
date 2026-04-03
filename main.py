"""
FastAPI inference server for ANLI Round 2 NLI classification.

Supports two models:
  - DeBERTa-v3-base (86M params, ~380ms inference)
  - DeBERTa-v3-large (304M params, ~1.2s inference, higher accuracy)

Hybrid mode: routes low-confidence DeBERTa predictions to a reasoning LLM
(via HuggingFace Inference API) for improved accuracy.

Endpoints:
    GET  /               → Web UI
    GET  /presentation   → Project presentation
    GET  /health         → Health check
    GET  /models         → List available and active models
    POST /models/switch  → Switch active model
    POST /predict        → NLI prediction (with optional hybrid routing)
    POST /predict/batch  → Batch NLI predictions
"""

import os
import re
import json
import time
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2Tokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hybrid routing config
HF_TOKEN = os.getenv("HF_TOKEN", None)
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.90"))
FEEDBACK_FILE = os.getenv("FEEDBACK_FILE", "feedback.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "base": {
        "name": "DeBERTa-v3-base",
        "hf_id": "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "dir": "base",
        "params": "86M",
        "description": "Fast inference (~380ms). 54.6% on ANLI R2.",
    },
    "large": {
        "name": "DeBERTa-v3-large",
        "hf_id": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        "dir": "large",
        "params": "304M",
        "description": "Higher accuracy (68.2% on ANLI R2). Slower inference (~1.2s).",
    },
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
tokenizer = None
active_model_key = None
llm_client = None

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES = ["entailment", "neutral", "contradiction"]
REMAP = {}


def get_model_path(model_key):
    path = os.path.join(MODEL_DIR, MODEL_REGISTRY[model_key]["dir"])
    if os.path.isdir(path):
        return path
    if model_key == "base" and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        return MODEL_DIR
    return path


def get_available_models():
    available = {}
    for key, info in MODEL_REGISTRY.items():
        path = get_model_path(key)
        config_exists = os.path.isfile(os.path.join(path, "config.json"))
        available[key] = {**info, "path": path, "downloaded": config_exists}
    return available


def load_model(model_key):
    global model, tokenizer, active_model_key, REMAP

    path = get_model_path(model_key)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(
            f"Model '{model_key}' not found at {path}. "
            f"Run: python download_model.py --model {model_key}"
        )

    logger.info(f"Loading model '{model_key}' from {path} on {DEVICE}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info("Loaded tokenizer via AutoTokenizer")
    except Exception:
        spm_path = os.path.join(path, "spm.model")
        if os.path.exists(spm_path):
            tokenizer = DebertaV2Tokenizer(vocab_file=spm_path, do_lower_case=False)
            logger.info("Loaded tokenizer via DebertaV2Tokenizer (spm.model)")
        else:
            raise RuntimeError(f"Could not load tokenizer from {path}")

    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()

    model_id2label = model.config.id2label
    REMAP = {}
    for model_idx, label_str in model_id2label.items():
        model_idx = int(model_idx)
        label_lower = label_str.lower()
        if "entail" in label_lower:
            REMAP[model_idx] = 0
        elif "neutral" in label_lower:
            REMAP[model_idx] = 1
        elif "contra" in label_lower:
            REMAP[model_idx] = 2

    active_model_key = model_key
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model '{model_key}' loaded ({params:,} params). Remap: {REMAP}")


def init_llm_client():
    """Initialize the HuggingFace Inference API client for hybrid routing."""
    global llm_client
    try:
        from huggingface_hub import InferenceClient
        llm_client = InferenceClient(model=LLM_MODEL, token=HF_TOKEN)
        logger.info(f"LLM client initialized: {LLM_MODEL}")
    except ImportError:
        logger.warning(
            "huggingface_hub not installed — hybrid routing disabled. "
            "Install with: pip install huggingface_hub"
        )
        llm_client = None
    except Exception as e:
        logger.warning(f"Failed to init LLM client: {e} — hybrid routing disabled.")
        llm_client = None


# ---------------------------------------------------------------------------
# Feedback storage
# ---------------------------------------------------------------------------
def load_feedback() -> list[dict]:
    """Load feedback entries from JSON file."""
    if os.path.isfile(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_feedback(entries: list[dict]):
    """Save feedback entries to JSON file."""
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def append_feedback(entry: dict):
    """Append a single feedback entry and persist."""
    entries = load_feedback()
    entries.append(entry)
    save_feedback(entries)
    logger.info(f"Feedback saved: id={entry['id']} correct={entry['is_correct']}")


# ---------------------------------------------------------------------------
# LLM reasoning
# ---------------------------------------------------------------------------
NLI_SYSTEM_PROMPT = (
    "You are an expert at Natural Language Inference. Given a premise and hypothesis, "
    "determine the relationship.\n\n"
    "**Definitions:**\n"
    "- **entailment**: The hypothesis is definitely true given the premise.\n"
    "- **neutral**: The hypothesis might or might not be true; the premise doesn't give enough information.\n"
    "- **contradiction**: The hypothesis is definitely false given the premise.\n\n"
    "Think step-by-step, then state your final answer as exactly one of: entailment, neutral, contradiction."
)

NLI_USER_TEMPLATE = (
    "**Premise:** {premise}\n\n"
    "**Hypothesis:** {hypothesis}\n\n"
    "Think step-by-step, then give your **Final Answer:**"
)


def extract_label(text):
    """Extract NLI label from LLM output."""
    text_lower = text.lower().strip()

    final_match = re.search(
        r'final\s*answer[:\s]*\*{0,2}\s*(entailment|neutral|contradiction)', text_lower
    )
    if final_match:
        return LABEL_NAMES.index(final_match.group(1))

    last_pos = -1
    last_label = None
    for i, lbl in enumerate(LABEL_NAMES):
        pos = text_lower.rfind(lbl)
        if pos > last_pos:
            last_pos = pos
            last_label = i

    if last_label is not None:
        return last_label

    return 1  # fallback: neutral


def llm_classify(premise: str, hypothesis: str) -> tuple[int, str, float]:
    """
    Classify using the reasoning LLM via HF Inference API.
    Returns (label_index, reasoning_text, elapsed_ms).
    """
    messages = [
        {"role": "system", "content": NLI_SYSTEM_PROMPT},
        {"role": "user", "content": NLI_USER_TEMPLATE.format(premise=premise, hypothesis=hypothesis)},
    ]

    start = time.perf_counter()
    try:
        response = llm_client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        text = response.choices[0].message.content
        label = extract_label(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return label, text, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"LLM call failed ({elapsed_ms:.0f}ms): {e}")
        raise


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    available = get_available_models()
    if available["large"]["downloaded"]:
        load_model("large")
    elif available["base"]["downloaded"]:
        load_model("base")
    else:
        logger.error(f"No models found in {MODEL_DIR}. Run download_model.py first.")

    # Initialize LLM client for hybrid routing
    init_llm_client()

    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ANLI R2 NLI Classifier",
    description=(
        "3-way Natural Language Inference classifier (entailment / neutral / contradiction) "
        "using DeBERTa-v3 with optional hybrid routing to a reasoning LLM for low-confidence predictions."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

figures_dir = Path(__file__).parent / "figures"
if figures_dir.exists():
    app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    premise: str = Field(..., min_length=1, examples=["The Parma trolleybus system comprises four urban routes."])
    hypothesis: str = Field(..., min_length=1, examples=["The trolleybus system has over 2 urban routes."])
    hybrid: bool = Field(False, description="Enable hybrid routing: route low-confidence predictions to reasoning LLM")
    confidence_threshold: float | None = Field(None, ge=0.0, le=1.0, description="Override default confidence threshold (default: 0.90)")

class PredictionResponse(BaseModel):
    label: str = Field(..., examples=["entailment"])
    confidence: float = Field(..., examples=[0.95])
    probabilities: dict = Field(..., examples=[{"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02}])
    inference_time_ms: float = Field(..., examples=[45.2])
    model: str = Field(..., examples=["base"])
    routed_to_llm: bool = Field(False, description="Whether this prediction was routed to the reasoning LLM")
    llm_model: str | None = Field(None, description="Name of the LLM used, if routed")
    llm_reasoning: str | None = Field(None, description="Chain-of-thought reasoning from LLM, if routed")
    deberta_label: str | None = Field(None, description="DeBERTa's original prediction (before routing)")
    deberta_time_ms: float | None = Field(None, description="DeBERTa inference time in ms")
    llm_label: str | None = Field(None, description="LLM's prediction, if routed")
    llm_time_ms: float | None = Field(None, description="LLM inference time in ms, if routed")
    confidence_threshold: float | None = Field(None, description="Confidence threshold used for routing")

class BatchRequest(BaseModel):
    pairs: list[PredictionRequest] = Field(..., min_length=1, max_length=64)

class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]
    total_inference_time_ms: float
    model: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_model: str | None
    device: str
    model_dir: str
    hybrid_available: bool
    llm_model: str | None

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


class FeedbackRequest(BaseModel):
    premise: str = Field(..., min_length=1)
    hypothesis: str = Field(..., min_length=1)
    predicted_label: str = Field(..., description="The label the system predicted")
    is_correct: bool = Field(..., description="Whether the prediction was correct")
    correct_label: str | None = Field(None, description="User-provided correct label (if is_correct=False)")
    confidence: float | None = Field(None, description="Model confidence for the prediction")
    model: str | None = Field(None, description="Which model made the prediction")
    routed_to_llm: bool = Field(False, description="Whether hybrid routing was used")
    llm_label: str | None = Field(None, description="LLM's prediction, if routed")

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_single(
    premise: str,
    hypothesis: str,
    hybrid: bool = False,
    confidence_threshold: float | None = None,
) -> PredictionResponse:
    start = time.perf_counter()

    # Step 1: DeBERTa prediction
    inputs = tokenizer(
        premise, hypothesis,
        max_length=MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    probs_raw = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
    pred_model_idx = int(torch.argmax(outputs.logits, dim=-1).item())
    pred_anli_idx = REMAP[pred_model_idx]
    confidence = max(probs_raw)

    anli_probs = {LABEL_MAP[REMAP[i]]: round(probs_raw[i], 4) for i in range(3)}

    deberta_ms = (time.perf_counter() - start) * 1000

    # Step 2: Hybrid routing (if enabled and confidence below threshold)
    threshold = confidence_threshold if confidence_threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
    routed = False
    llm_model_name = None
    llm_reasoning = None

    deberta_label = LABEL_MAP[pred_anli_idx]
    llm_label_str = None
    llm_ms = 0.0

    should_route = confidence < threshold or threshold >= 1.0

    if hybrid and llm_client is not None and should_route:
        try:
            llm_label, reasoning, llm_ms = llm_classify(premise, hypothesis)
            routed = True
            pred_anli_idx = llm_label
            llm_model_name = LLM_MODEL
            llm_reasoning = reasoning
            llm_label_str = LABEL_MAP[llm_label]
            total_ms = deberta_ms + llm_ms
            logger.info(
                f"Routed to LLM (conf={confidence:.3f} < {threshold}): "
                f"DeBERTa={deberta_label} → LLM={llm_label_str} "
                f"({llm_ms:.0f}ms)"
            )
        except Exception as e:
            logger.warning(f"LLM fallback to DeBERTa: {e}")
            total_ms = deberta_ms
    else:
        total_ms = deberta_ms

    return PredictionResponse(
        label=LABEL_MAP[pred_anli_idx],
        confidence=round(confidence, 4),
        probabilities=anli_probs,
        inference_time_ms=round(total_ms, 2),
        model=active_model_key,
        routed_to_llm=routed,
        llm_model=llm_model_name,
        llm_reasoning=llm_reasoning,
        deberta_label=deberta_label,
        deberta_time_ms=round(deberta_ms, 2),
        llm_label=llm_label_str,
        llm_time_ms=round(llm_ms, 2) if routed else None,
        confidence_threshold=threshold if hybrid else None,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return HTMLResponse(
        "<h3>ANLI R2 NLI Classifier</h3>"
        "<p>API is running. Visit <a href='/docs'>/docs</a> for Swagger UI.</p>"
    )


@app.get("/presentation", response_class=HTMLResponse)
def presentation():
    html_path = Path(__file__).parent / "static" / "presentation.html"
    if html_path.exists():
        return html_path.read_text()
    raise HTTPException(status_code=404, detail="Presentation not found")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        active_model=active_model_key,
        device=DEVICE,
        model_dir=MODEL_DIR,
        hybrid_available=llm_client is not None,
        llm_model=LLM_MODEL if llm_client is not None else None,
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    available = get_available_models()
    models_list = [
        ModelInfo(
            key=key,
            name=info["name"],
            params=info["params"],
            description=info["description"],
            downloaded=info["downloaded"],
            active=(key == active_model_key),
        )
        for key, info in available.items()
    ]
    return ModelsResponse(
        active_model=active_model_key,
        models=models_list,
        hybrid_available=llm_client is not None,
        llm_model=LLM_MODEL if llm_client is not None else None,
    )


@app.post("/models/switch", response_model=SwitchResponse)
def switch_model(request: SwitchRequest):
    if request.model not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{request.model}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    if request.model == active_model_key:
        return SwitchResponse(
            message=f"Model '{request.model}' is already active.",
            active_model=active_model_key,
            load_time_seconds=0.0,
        )

    start = time.perf_counter()
    try:
        load_model(request.model)
    except RuntimeError as e:
        raise HTTPException(status_code=404, detail=str(e))

    elapsed = time.perf_counter() - start
    return SwitchResponse(
        message=f"Switched to '{request.model}' successfully.",
        active_model=active_model_key,
        load_time_seconds=round(elapsed, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    logger.info(f"Predict request: hybrid={request.hybrid}, threshold={request.confidence_threshold}")
    return predict_single(
        request.premise,
        request.hypothesis,
        hybrid=request.hybrid,
        confidence_threshold=request.confidence_threshold,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if model is None:
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
        model=active_model_key,
    )


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------
@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    """Submit user feedback on a prediction (correct/incorrect + optional correction)."""
    if request.correct_label and request.correct_label not in LABEL_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid correct_label '{request.correct_label}'. Must be one of: {LABEL_NAMES}"
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

    return FeedbackResponse(
        message="Thank you! Feedback recorded." if request.is_correct
                else f"Feedback recorded. Correct label: {request.correct_label or 'not provided'}.",
        feedback_id=entry["id"],
    )


@app.get("/feedback/stats", response_model=FeedbackStatsResponse)
def feedback_stats():
    """Get summary statistics of collected feedback."""
    entries = load_feedback()

    if not entries:
        return FeedbackStatsResponse(
            total=0, correct=0, incorrect=0, accuracy=None,
            corrections_by_label={}, corrections_by_model={},
            avg_confidence_correct=None, avg_confidence_incorrect=None,
        )

    correct = [e for e in entries if e.get("is_correct")]
    incorrect = [e for e in entries if not e.get("is_correct")]

    # Count corrections by true label
    corrections_by_label = {}
    for e in incorrect:
        cl = e.get("correct_label", "unknown") or "not_provided"
        corrections_by_label[cl] = corrections_by_label.get(cl, 0) + 1

    # Count corrections by model
    corrections_by_model = {}
    for e in incorrect:
        m = e.get("model", "unknown") or "unknown"
        corrections_by_model[m] = corrections_by_model.get(m, 0) + 1

    # Average confidence
    correct_confs = [e["confidence"] for e in correct if e.get("confidence") is not None]
    incorrect_confs = [e["confidence"] for e in incorrect if e.get("confidence") is not None]

    return FeedbackStatsResponse(
        total=len(entries),
        correct=len(correct),
        incorrect=len(incorrect),
        accuracy=round(len(correct) / len(entries), 4) if entries else None,
        corrections_by_label=corrections_by_label,
        corrections_by_model=corrections_by_model,
        avg_confidence_correct=round(sum(correct_confs) / len(correct_confs), 4) if correct_confs else None,
        avg_confidence_incorrect=round(sum(incorrect_confs) / len(incorrect_confs), 4) if incorrect_confs else None,
    )


@app.get("/feedback/export")
def export_feedback():
    """Export all feedback as JSON (for training data or analysis)."""
    entries = load_feedback()
    return {
        "count": len(entries),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "entries": entries,
    }
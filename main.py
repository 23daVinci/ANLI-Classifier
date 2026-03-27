"""
FastAPI inference server for ANLI Round 2 NLI classification.

Supports two models:
  - DeBERTa-v3-base (86M params, ~380ms inference)
  - DeBERTa-v3-large (304M params, ~1.2s inference, higher accuracy)

Endpoints:
    GET  /               → Web UI
    GET  /presentation   → Project presentation
    GET  /health         → Health check
    GET  /models         → List available and active models
    POST /models/switch  → Switch active model
    POST /predict        → NLI prediction
    POST /predict/batch  → Batch NLI predictions
"""

import os
import time
import logging
from contextlib import asynccontextmanager
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
        "description": "Higher accuracy (~58-62% on ANLI R2). Slower inference (~1.2s).",
    },
}

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
model = None
tokenizer = None
active_model_key = None

LABEL_MAP = {0: "entailment", 1: "neutral", 2: "contradiction"}
REMAP = {}


def get_model_path(model_key):
    """Resolve the path for a given model key."""
    model_dir_name = MODEL_REGISTRY[model_key]["dir"]
    # Check for models/<key> structure first
    path = os.path.join(MODEL_DIR, model_dir_name)
    if os.path.isdir(path):
        return path
    # Fall back to MODEL_DIR itself (single-model setup / backward compat)
    if model_key == "base" and os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        return MODEL_DIR
    return path


def get_available_models():
    """Check which models are downloaded."""
    available = {}
    for key, info in MODEL_REGISTRY.items():
        path = get_model_path(key)
        config_exists = os.path.isfile(os.path.join(path, "config.json"))
        available[key] = {**info, "path": path, "downloaded": config_exists}
    return available


def load_model(model_key):
    """Load a specific model and tokenizer."""
    global model, tokenizer, active_model_key, REMAP

    path = get_model_path(model_key)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(
            f"Model '{model_key}' not found at {path}. "
            f"Run: python download_model.py --model {model_key}"
        )

    logger.info(f"Loading model '{model_key}' from {path} on {DEVICE}...")

    # Tokenizer
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

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.to(DEVICE)
    model.eval()

    # Build label remap
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


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the best available model at startup
    available = get_available_models()
    # Prefer large if downloaded, otherwise base
    if available["large"]["downloaded"]:
        load_model("large")
    elif available["base"]["downloaded"]:
        load_model("base")
    else:
        logger.error(f"No models found in {MODEL_DIR}. Run download_model.py first.")
    yield
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ANLI R2 NLI Classifier",
    description=(
        "3-way Natural Language Inference classifier (entailment / neutral / contradiction) "
        "using DeBERTa-v3 fine-tuned on MNLI + Fever-NLI + ANLI. "
        "Supports base (86M) and large (304M) model variants."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Serve static files
figures_dir = Path(__file__).parent / "figures"
if figures_dir.exists():
    app.mount("/figures", StaticFiles(directory=str(figures_dir)), name="figures")


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
    model: str = Field(..., examples=["base"])

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

class SwitchRequest(BaseModel):
    model: str = Field(..., examples=["large"], description="Model key: 'base' or 'large'")

class SwitchResponse(BaseModel):
    message: str
    active_model: str
    load_time_seconds: float


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict_single(premise: str, hypothesis: str) -> PredictionResponse:
    start = time.perf_counter()

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

    # Remap probabilities to ANLI ordering
    anli_probs = {LABEL_MAP[REMAP[i]]: round(probs_raw[i], 4) for i in range(3)}

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictionResponse(
        label=LABEL_MAP[pred_anli_idx],
        confidence=round(max(probs_raw), 4),
        probabilities=anli_probs,
        inference_time_ms=round(elapsed_ms, 2),
        model=active_model_key,
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
    )


@app.get("/models", response_model=ModelsResponse)
def list_models():
    available = get_available_models()
    models = [
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
    return ModelsResponse(active_model=active_model_key, models=models)


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
    return predict_single(request.premise, request.hypothesis)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    predictions = [
        predict_single(pair.premise, pair.hypothesis) for pair in request.pairs
    ]
    total_ms = (time.perf_counter() - start) * 1000

    return BatchResponse(
        predictions=predictions,
        total_inference_time_ms=round(total_ms, 2),
        model=active_model_key,
    )
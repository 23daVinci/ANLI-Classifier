# ANLI Round 2 — NLI Classifier

3-way Natural Language Inference (entailment / neutral / contradiction) on the Adversarial NLI Round 2 benchmark using DeBERTa-v3 with optional hybrid routing to a reasoning LLM.

## Results

| Model | Params | Test Accuracy | Test Macro F1 |
|-------|:---:|:---:|:---:|
| Random Uniform (50-seed avg) | — | 33.5% | 0.335 |
| Majority Class (neutral) | — | 33.3% | 0.167 |
| TF-IDF + LogReg (concat) | — | 35.0% | 0.335 |
| LinearSVC (sep TF-IDF + NLI feats) | — | 42.5% | 0.421 |
| BERT-base (fine-tuned) | 110M | 43.5% | 0.431 |
| DeBERTa-v3-base (MNLI+FEVER+ANLI) | 86M | 54.6% | 0.546 |
| DeBERTa-v3-large (MNLI+FEVER+ANLI) | 304M | 68.2% | 0.682 |
| **Hybrid: DeBERTa-v3-large + Qwen-72B** | **304M + API** | **72.1%** | **0.722** |

> ANLI R2 is adversarially constructed — annotators wrote hypotheses specifically designed to fool transformer models. The hybrid system routes low-confidence DeBERTa predictions to a reasoning LLM, achieving 72.1% accuracy — 38.8 points above random.

## Project Structure

```
anli-nli-classifier/
├── main.py                           # App entry point — creates FastAPI app, mounts routers
├── core/
│   ├── config.py                     # Pydantic Settings — all env vars and constants
│   └── lifespan.py                   # Startup/shutdown (model load, LLM client init)
├── models/                           # Pydantic request/response schemas
│   ├── prediction.py                 # PredictionRequest/Response, BatchRequest/Response
│   ├── health.py                     # LivenessResponse, ReadinessResponse
│   ├── registry.py                   # ModelInfo, ModelsResponse, SwitchRequest/Response
│   └── feedback.py                   # FeedbackRequest/Response, FeedbackStatsResponse
├── services/                         # ML and business logic
│   ├── model_service.py              # ModelState, load_model, get_available_models
│   ├── llm_service.py                # LLMState, init_llm_client, llm_classify
│   ├── inference_service.py          # predict_single (DeBERTa + hybrid routing)
│   └── feedback_service.py           # load/save/append/compute_stats
├── routers/                          # API endpoints (all versioned under /v1)
│   ├── health.py                     # GET /v1/health/live, GET /v1/health/ready
│   ├── models.py                     # GET /v1/models, POST /v1/models/switch
│   ├── predict.py                    # POST /v1/predict, POST /v1/predict/batch
│   ├── feedback.py                   # POST /v1/feedback, GET /v1/feedback/stats|export
│   └── ui.py                         # GET /, GET /presentation
├── tests/
│   ├── conftest.py                   # Shared fixtures (client, loaded_model)
│   ├── test_health.py
│   ├── test_predict.py
│   ├── test_models.py
│   └── test_feedback.py
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory Data Analysis
│   ├── Baseline.ipynb                # Baseline models
│   └── Phase_3.ipynb                 # DeBERTa-v3 evaluation + error analysis
├── figures/                          # Plots from EDA and evaluation
├── best_model/                       # Model weights (not in repo — see Setup)
│   ├── base/                         # DeBERTa-v3-base (~360MB)
│   └── large/                        # DeBERTa-v3-large (~1.2GB, optional)
├── static/
│   ├── index.html                    # Web UI with hybrid toggle + threshold slider
│   └── presentation.html             # Interactive project presentation
├── download_model.py                 # Model download script (base / large / both)
├── Dockerfile
├── docker-compose.yaml
├── pyrightconfig.json                # Pyright/Pylance type checker config
├── pytest.ini                        # Test runner config
├── .env                              # Local env vars — MODEL_DIR, HF_TOKEN (not in repo)
├── requirements.txt                  # API dependencies (CPU-only torch)
└── README.md
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Model weights in `best_model/` directory (see below)
- (Optional) HuggingFace token for hybrid routing

### Model Setup

The model weights are not included in the repository due to size. Download them with the included script:

```bash
# Download base model only (~360MB) — fast inference, 54.6% accuracy
python download_model.py

# Download large model only (~1.2GB) — 68.2% accuracy
python download_model.py --model large

# Download both models
python download_model.py --all
```

### Hybrid Routing Setup (Optional)

To enable hybrid mode (DeBERTa + reasoning LLM), create a `.env` file with your HuggingFace token:

```bash
HF_TOKEN=hf_your_token_here
```

Get a free token at https://huggingface.co/settings/tokens. Without this, the server runs normally with DeBERTa only.

### Run with Docker Compose (Recommended)

```bash
# Create feedback file before first run (Docker mounts it as a volume)
echo "[]" > feedback.json

# Build and start
docker compose up --build

# Or run detached
docker compose up --build -d
```

The server auto-detects available models and loads the best one at startup (prefers large if downloaded).

- **Web UI**: `http://localhost:8000` — interactive classifier with hybrid toggle
- **Presentation**: `http://localhost:8000/presentation` — full project walkthrough
- **Swagger UI**: `http://localhost:8000/docs` — API documentation
- **Liveness**: `http://localhost:8000/v1/health/live`
- **Readiness**: `http://localhost:8000/v1/health/ready`

### Run with Docker Directly

```bash
docker build -t anli-nli-classifier .
docker run -p 8000:8000 -v ./best_model:/app/model:ro -e HF_TOKEN=$HF_TOKEN anli-nli-classifier
```

### Run without Docker

```bash
pip install -r requirements.txt
# Set MODEL_DIR and HF_TOKEN in .env (see Hybrid Routing Setup above)
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Hybrid Routing

The system combines DeBERTa-v3's NLI-tuned precision with a reasoning LLM's world knowledge and chain-of-thought inference.

### How It Works

1. DeBERTa-v3 classifies the input (~0.4-1.2s)
2. If confidence is below the threshold (default 90%), the prediction is routed to Qwen2.5-72B-Instruct via HuggingFace Inference API
3. The LLM reasons step-by-step and returns its classification (~3-8s)
4. The LLM's answer replaces DeBERTa's for that prediction
5. If the LLM call fails, DeBERTa's answer is returned as fallback

### Results by Threshold

| Threshold | Routed | Hybrid Acc | Delta | LLM Acc on Routed |
|:-:|:-:|:-:|:-:|:-:|
| < 0.70 | 7.7% | 69.1% | +0.9% | 57.1% |
| < 0.80 | 12.6% | 70.4% | +2.2% | 62.7% |
| < 0.90 | 19.2% | 71.2% | +3.0% | 64.1% |
| **< 0.95** | **27.5%** | **72.1%** | **+3.9%** | **65.5%** |

The LLM consistently outperforms DeBERTa on routed examples (65.5% vs 51.3%), validating the complementary strengths hypothesis.

### Via Web UI

Toggle "Hybrid mode" in the UI and adjust the confidence threshold slider (50-100%). The result card shows both DeBERTa's and the LLM's predictions side-by-side with reasoning.

### Via API

```bash
# Standard prediction (DeBERTa only)
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "The Parma trolleybus system comprises four urban routes.",
    "hypothesis": "The trolleybus system has over 2 urban routes."
  }'

# Hybrid prediction (routes to LLM if confidence < 0.90)
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "The Parma trolleybus system comprises four urban routes.",
    "hypothesis": "The trolleybus system has over 2 urban routes.",
    "hybrid": true
  }'

# Hybrid with custom threshold (force all predictions through LLM)
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "Muccan Station is a pastoral lease that operates as a cattle station.",
    "hypothesis": "Muccan Station is on the radio.",
    "hybrid": true,
    "confidence_threshold": 1.0
  }'
```

Hybrid response includes additional fields:

```json
{
  "label": "contradiction",
  "confidence": 0.731,
  "probabilities": {"entailment": 0.731, "neutral": 0.082, "contradiction": 0.187},
  "inference_time_ms": 15823.0,
  "model": "large",
  "routed_to_llm": true,
  "llm_model": "Qwen/Qwen2.5-72B-Instruct",
  "llm_reasoning": "Step 1: The premise states...",
  "deberta_label": "entailment",
  "deberta_time_ms": 7205.0,
  "llm_label": "contradiction",
  "llm_time_ms": 8618.0,
  "confidence_threshold": 0.9
}
```

## Model Switching

The server supports live model switching without restart.

```bash
# List available models
curl http://localhost:8000/v1/models

# Switch to large model
curl -X POST http://localhost:8000/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "large"}'
```

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI with hybrid toggle and threshold slider |
| `GET` | `/presentation` | Interactive project presentation |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/v1/health/live` | Liveness — is the process running? (always 200) |
| `GET` | `/v1/health/ready` | Readiness — is the model loaded? (200 or 503) |
| `GET` | `/v1/models` | List available models and active model |
| `POST` | `/v1/models/switch` | Switch active model (base/large) without restart |
| `POST` | `/v1/predict` | Classification (with optional hybrid routing) |
| `POST` | `/v1/predict/batch` | Batch classification (up to 64 pairs) |
| `POST` | `/v1/feedback` | Submit correctness feedback on a prediction |
| `GET` | `/v1/feedback/stats` | Aggregated feedback statistics |
| `GET` | `/v1/feedback/export` | Export all feedback as JSON |

## Methodology

### Dataset

ANLI Round 2 (Nie et al., 2020) — 45,460 training / 1,000 dev / 1,000 test examples. Created via an adversarial human-and-model-in-the-loop process (HAMLET) where annotators wrote hypotheses to fool RoBERTa ensembles. Premises sourced from Wikipedia via HotpotQA.

### Models

| Variant | Architecture | Training Data | Parameters |
|---|---|---|---|
| **Base** | DeBERTa-v3-base | MNLI + Fever-NLI + ANLI (763K pairs) | 86M |
| **Large** | DeBERTa-v3-large | MNLI + Fever-NLI + ANLI + LingNLI + WANLI (885K pairs) | 304M |
| **Hybrid** | DeBERTa-v3-large + Qwen2.5-72B-Instruct | Confidence-based routing | 304M + API |

### Key Findings

- **72.1% accuracy** with hybrid routing (DeBERTa-v3-large + Qwen-72B), +3.9% over DeBERTa alone
- **68.2% accuracy** with DeBERTa-v3-large standalone (34.9 points above random)
- BERT-base fine-tuned reaches only 43.5%, barely above LinearSVC (42.5%) — ANLI was built to fool MLM models
- DeBERTa-v3-large is 3.4x better calibrated than base (confidence gap: 0.054 vs 0.016), making routing more effective
- LLM outperforms DeBERTa on routed examples (65.5% vs 51.3%), confirming complementary strengths
- 58% of DeBERTa errors are overconfident (>0.9) and bypass routing — the main limitation
- Quantifier reasoning has the highest error rate (50.9%), followed by surface-level reasoning (48.2%)

## Configuration

All variables can be set in `.env` (local) or as real environment variables (Docker). Validated at startup by Pydantic Settings.

| Environment Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `/app/model` | Path to model weights directory (`best_model` locally) |
| `MAX_LENGTH` | `256` | Maximum token sequence length |
| `DEVICE` | auto-detected | `cuda` or `cpu` — override to force one |
| `HF_TOKEN` | `None` | HuggingFace token for hybrid routing |
| `LLM_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | Reasoning LLM model ID |
| `CONFIDENCE_THRESHOLD` | `0.90` | Default routing threshold |
| `FEEDBACK_FILE` | `feedback.json` | Path to feedback storage file |

## References

1. Nie et al. (2020) — *Adversarial NLI: A New Benchmark for Natural Language Understanding*
2. He et al. (2023) — *DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*
3. Laurer et al. (2022) — *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI*
# ANLI Round 2 — NLI Classifier

3-way Natural Language Inference (entailment / neutral / contradiction) on the Adversarial NLI Round 2 benchmark using DeBERTa-v3.

## Results

| Model | Params | Test Accuracy | Test Macro F1 |
|-------|:---:|:---:|:---:|
| Random Uniform (50-seed avg) | — | 33.5% | 0.335 |
| Majority Class (neutral) | — | 33.3% | 0.167 |
| TF-IDF + LogReg (concat) | — | 35.0% | 0.335 |
| LinearSVC (sep TF-IDF + NLI feats) | — | 42.5% | 0.421 |
| BERT-base (fine-tuned) | 110M | 43.5% | 0.431 |
| **DeBERTa-v3-base (MNLI+FEVER+ANLI)** | **86M** | **54.6%** | **0.546** |
| **DeBERTa-v3-large (MNLI+FEVER+ANLI)** | **304M** | **68.2%** | **0.682** |

> ANLI R2 is adversarially constructed — annotators wrote hypotheses specifically designed to fool transformer models. DeBERTa-v3-large achieves 68.2% accuracy, 34.9 points above random and 24.7 points above fine-tuned BERT-base.

## Project Structure

```
anli-nli-classifier/
├── notebooks/
│   ├── phase1_eda.ipynb              # Exploratory Data Analysis
│   ├── phase2_baselines.ipynb        # Baseline models
│   └── phase3_deberta.ipynb          # DeBERTa-v3 evaluation + error analysis
├── figures/                          # Plots from EDA and evaluation
├── best_model/                       # Model weights (not in repo — see Setup)
│   ├── base/                         # DeBERTa-v3-base (~360MB)
│   └── large/                        # DeBERTa-v3-large (~1.2GB, optional)
├── main.py                           # FastAPI inference server
├── static/
│   ├── index.html                    # Web UI with model switching
│   └── presentation.html             # Interactive project presentation
├── download_model.py                 # Model download script (base / large / both)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                  # API dependencies (CPU-only torch)
├── results_summary.json              # Evaluation metrics
├── presentation/
└── README.md
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Model weights in `best_model/` directory (see below)

### Model Setup

The model weights are not included in the repository due to size. Download them with the included script:

```bash
# Download base model only (~360MB) — fast inference, 54.6% accuracy
python download_model.py

# Download large model only (~1.2GB) — higher accuracy (68.2%)
python download_model.py --model large

# Download both models
python download_model.py --all
```

The script downloads from HuggingFace Hub, auto-installs `huggingface_hub` if needed, and skips models that already exist.

### Run with Docker Compose (Recommended)

```bash
# Build and start
docker compose up --build

# Or run detached
docker compose up --build -d
```

The server auto-detects available models and loads the best one at startup (prefers large if downloaded).

- **Web UI**: `http://localhost:8000` — interactive classifier with model switching
- **Presentation**: `http://localhost:8000/presentation` — full project walkthrough with figures
- **Swagger UI**: `http://localhost:8000/docs` — API documentation with test interface
- **Health Check**: `http://localhost:8000/health`

### Run with Docker Directly

```bash
docker build -t anli-nli-classifier .
docker run -p 8000:8000 -v ./best_model:/app/model:ro anli-nli-classifier
```

### Run without Docker

```bash
pip install -r requirements.txt
export MODEL_DIR=./best_model
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Model Switching

The server supports live model switching without restart. Both the Web UI and API support this.

### Via Web UI

Click the "Switch to DeBERTa-v3-large" button in the header badge row. The model loads in a few seconds and all subsequent predictions use the new model.

### Via API

```bash
# List available models
curl http://localhost:8000/models

# Switch to large model
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "large"}'

# Switch back to base
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "base"}'
```

## API Reference

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_model": "base",
  "device": "cpu",
  "model_dir": "/app/model"
}
```

### `GET /models` — List Models

```bash
curl http://localhost:8000/models
```

```json
{
  "active_model": "base",
  "models": [
    {"key": "base", "name": "DeBERTa-v3-base", "params": "86M", "downloaded": true, "active": true},
    {"key": "large", "name": "DeBERTa-v3-large", "params": "304M", "downloaded": true, "active": false}
  ]
}
```

### `POST /predict` — Single Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "premise": "The Parma trolleybus system comprises four urban routes.",
    "hypothesis": "The trolleybus system has over 2 urban routes."
  }'
```

```json
{
  "label": "entailment",
  "confidence": 0.9883,
  "probabilities": {
    "entailment": 0.9883,
    "neutral": 0.0107,
    "contradiction": 0.0008
  },
  "inference_time_ms": 472.23,
  "model": "base"
}
```

### `POST /predict/batch` — Batch Prediction (up to 64 pairs)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "pairs": [
      {
        "premise": "The Parma trolleybus system comprises four urban routes.",
        "hypothesis": "The trolleybus system has over 2 urban routes."
      },
      {
        "premise": "She moved to Paris in 2010.",
        "hypothesis": "She has never lived in France."
      }
    ]
  }'
```

### `POST /models/switch` — Switch Active Model

```bash
curl -X POST http://localhost:8000/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "large"}'
```

```json
{
  "message": "Switched to 'large' successfully.",
  "active_model": "large",
  "load_time_seconds": 4.32
}
```

### Interactive Docs

Swagger UI at `http://localhost:8000/docs` with pre-filled examples for all endpoints.

## Methodology

### Dataset

ANLI Round 2 (Nie et al., 2020) — 45,460 training / 1,000 dev / 1,000 test examples. Created via an adversarial human-and-model-in-the-loop process (HAMLET) where annotators wrote hypotheses to fool RoBERTa ensembles. Premises sourced from Wikipedia via HotpotQA.

### Models

| Variant | Architecture | Training Data | Parameters |
|---|---|---|---|
| **Base** | DeBERTa-v3-base | MNLI + Fever-NLI + ANLI (763K pairs) | 86M |
| **Large** | DeBERTa-v3-large | MNLI + Fever-NLI + ANLI + LingNLI + WANLI (885K pairs) | 304M |

DeBERTa-v3 uses disentangled attention and replaced token detection (RTD) pre-training, giving it a structural advantage over MLM-based models like BERT and RoBERTa on adversarial NLI data.

### Key Findings

- **68.2% accuracy** with DeBERTa-v3-large on ANLI R2 test (34.9 points above random)
- **54.6% accuracy** with DeBERTa-v3-base — scaling to large adds +13.6%
- BERT-base fine-tuned reaches only 43.5%, barely above LinearSVC (42.5%) — ANLI was built to fool MLM models
- Contradiction is the hardest class: recall of 50.2% (base) / 66.1% (large)
- The dominant error pattern is contradiction → entailment (29.4% of true contradictions), where the model misses logical reversals
- 58.1% of base model errors are high-confidence (>0.9), indicating poor calibration on adversarial examples
- Quantifier reasoning has the highest error rate (50.9%), followed by surface-level reasoning (48.2%)

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `/app/model` | Path to model weights directory (contains `base/` and/or `large/` subdirs) |
| `MAX_LENGTH` | `256` | Maximum token sequence length |

## References

1. Nie et al. (2020) — *Adversarial NLI: A New Benchmark for Natural Language Understanding*
2. He et al. (2023) — *DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*
3. Laurer et al. (2022) — *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI*
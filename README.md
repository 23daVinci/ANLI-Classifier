# ANLI Round 2 — NLI Classifier

3-way Natural Language Inference (entailment / neutral / contradiction) on the Adversarial NLI Round 2 benchmark using DeBERTa-v3-base.

## Results

| Model | Test Accuracy | Test Macro F1 |
|-------|:---:|:---:|
| Random Baseline | 33.3% | 0.333 |
| TF-IDF + Logistic Regression | — | — |
| BERT-base | — | — |
| **DeBERTa-v3-base (MNLI+FEVER+ANLI)** | **54.6%** | **0.546** |

> ANLI R2 is adversarially constructed — annotators wrote hypotheses specifically designed to fool transformer models. 54.6% represents a 21+ point improvement over random chance on this challenging benchmark.

## Project Structure

```
anli-nli-classifier/
├── notebooks/
│   ├── phase1_eda.ipynb              # Exploratory Data Analysis
│   ├── phase2_baselines.ipynb        # Baseline models
│   └── phase3_deberta.ipynb          # DeBERTa-v3 evaluation + error analysis
├── figures/                          # Plots from EDA and evaluation
├── best_model/                       # Model weights (not in repo — see Setup)
│   ├── config.json
│   ├── model.safetensors
│   ├── spm.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── main.py                           # FastAPI inference server
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
- Model weights in `best_model/` directory (see [Model Setup](#model-setup))

### Model Setup

The model weights are not included in the repository due to size (~360MB). Download them with the included script:

```bash
python download_model.py
```

This downloads `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` from HuggingFace Hub into `best_model/`. The script auto-installs `huggingface_hub` if needed and skips the download if the model already exists.

### Run with Docker Compose (Recommended)

```bash
# Build and start
docker compose up --build

# Or run detached
docker compose up --build -d
```

The API will be available at `http://localhost:8000`.

### Run with Docker Directly

```bash
# Build the image
docker build -t anli-nli-classifier .

# Run with model volume mount
docker run -p 8000:8000 -v ./best_model:/app/model:ro anli-nli-classifier
```

### Run without Docker

```bash
pip install -r requirements.txt

# Set model path
export MODEL_DIR=./best_model

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Reference

### `GET /` — API Info

```bash
curl http://localhost:8000/
```

### `GET /health` — Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cpu",
  "model_dir": "/app/model"
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
  "inference_time_ms": 472.23
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

### Interactive Docs

Swagger UI is available at `http://localhost:8000/docs` with pre-filled example requests.

## Methodology

### Dataset

ANLI Round 2 (Nie et al., 2020) — 45,460 training / 1,000 dev / 1,000 test examples. Created via an adversarial human-and-model-in-the-loop process (HAMLET) where annotators wrote hypotheses to fool RoBERTa ensembles. Premises sourced from Wikipedia via HotpotQA.

### Model

**DeBERTa-v3-base** (He et al., 2023) fine-tuned on MNLI + Fever-NLI + ANLI (763,913 NLI pairs) by [Laurer et al.](https://huggingface.co/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli). DeBERTa-v3 uses disentangled attention and replaced token detection (RTD) pre-training, giving it a structural advantage over MLM-based models like BERT and RoBERTa on adversarial NLI data.

### Key Findings

- **54.6% accuracy** on ANLI R2 test (21+ points above random baseline)
- Contradiction is the hardest class (50.15% recall) — the model defaults to neutral when uncertain
- 58.1% of errors are high-confidence (>0.9), indicating poor calibration on adversarial examples
- Quantifier reasoning has the highest error rate (50.9%), followed by surface-level reasoning (48.2%)
- The dominant error pattern is contradiction → entailment (98 examples), where the model misses logical reversals

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MODEL_DIR` | `/app/model` | Path to model weights directory |
| `MAX_LENGTH` | `256` | Maximum token sequence length |

## References

1. Nie et al. (2020) — *Adversarial NLI: A New Benchmark for Natural Language Understanding*
2. He et al. (2023) — *DeBERTa V3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing*
3. Laurer et al. (2022) — *Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI*
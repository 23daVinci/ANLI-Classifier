# ============================================================
# ANLI R2 NLI Classifier — Docker Image
# ============================================================
# Lightweight inference container for DeBERTa-v3-base NLI model.
# Model weights are mounted as a volume at runtime (not bundled).
#
# Build:  docker build -t anli-nli-classifier .
# Run:    docker run -p 8000:8000 -v ./best_model:/app/model anli-nli-classifier
# ============================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY static/ ./static/
COPY figures/ ./figures/

# Model directory — mounted as volume at runtime
# Default path the app expects; override with MODEL_DIR env var
ENV MODEL_DIR=/app/model

# Expose the API port
EXPOSE 8000

# Health check — container orchestrators use this
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
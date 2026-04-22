import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from core.lifespan import lifespan
from routers import health, models, predict, feedback, ui

logging.basicConfig(level=logging.INFO)

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

app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(feedback.router)
app.include_router(ui.router)

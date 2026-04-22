import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from services.model_service import get_available_models, load_model
from services.llm_service import init_llm_client
from core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    available = get_available_models()
    if available["large"]["downloaded"]:
        load_model("large")
    elif available["base"]["downloaded"]:
        load_model("base")
    else:
        logger.error(f"No models found in {settings.model_dir}. Run download_model.py first.")

    init_llm_client()
    yield
    logger.info("Shutting down...")

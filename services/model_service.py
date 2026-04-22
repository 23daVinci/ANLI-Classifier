import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DebertaV2Tokenizer
from core.config import settings, MODEL_REGISTRY

logger = logging.getLogger(__name__)


class ModelState:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.active_model_key: str | None = None
        self.remap: dict[int, int] = {}


state = ModelState()


def get_model_path(model_key: str) -> str:
    path = os.path.join(settings.model_dir, MODEL_REGISTRY[model_key]["dir"])
    if os.path.isdir(path):
        return path
    if model_key == "base" and os.path.isfile(os.path.join(settings.model_dir, "config.json")):
        return settings.model_dir
    return path


def get_available_models() -> dict:
    available = {}
    for key, info in MODEL_REGISTRY.items():
        path = get_model_path(key)
        config_exists = os.path.isfile(os.path.join(path, "config.json"))
        available[key] = {**info, "path": path, "downloaded": config_exists}
    return available


def load_model(model_key: str):
    path = get_model_path(model_key)
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise RuntimeError(
            f"Model '{model_key}' not found at {path}. "
            f"Run: python download_model.py --model {model_key}"
        )

    logger.info(f"Loading model '{model_key}' from {path} on {settings.device}...")

    try:
        state.tokenizer = AutoTokenizer.from_pretrained(path)
        logger.info("Loaded tokenizer via AutoTokenizer")
    except Exception:
        spm_path = os.path.join(path, "spm.model")
        if os.path.exists(spm_path):
            state.tokenizer = DebertaV2Tokenizer(vocab_file=spm_path, do_lower_case=False)
            logger.info("Loaded tokenizer via DebertaV2Tokenizer (spm.model)")
        else:
            raise RuntimeError(f"Could not load tokenizer from {path}")

    state.model = AutoModelForSequenceClassification.from_pretrained(path)
    state.model.to(settings.device)
    state.model.eval()

    model_id2label = state.model.config.id2label
    state.remap = {}
    for model_idx, label_str in model_id2label.items():
        model_idx = int(model_idx)
        label_lower = label_str.lower()
        if "entail" in label_lower:
            state.remap[model_idx] = 0
        elif "neutral" in label_lower:
            state.remap[model_idx] = 1
        elif "contra" in label_lower:
            state.remap[model_idx] = 2

    state.active_model_key = model_key
    params = sum(p.numel() for p in state.model.parameters())
    logger.info(f"Model '{model_key}' loaded ({params:,} params). Remap: {state.remap}")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_dir: str = Field("/app/model")
    max_length: int = Field(256)
    # Reads DEVICE env var if set; falls back to hardware detection
    device: str = Field(default_factory=_detect_device)
    hf_token: str | None = Field(None)
    llm_model: str = Field("Qwen/Qwen2.5-72B-Instruct")
    confidence_threshold: float = Field(0.90, ge=0.0, le=1.0)
    feedback_file: str = Field("feedback.json")


settings = Settings()

# Pure constants — not env-configurable
MODEL_REGISTRY: dict[str, dict] = {
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

LABEL_MAP: dict[int, str] = {0: "entailment", 1: "neutral", 2: "contradiction"}
LABEL_NAMES: list[str] = ["entailment", "neutral", "contradiction"]

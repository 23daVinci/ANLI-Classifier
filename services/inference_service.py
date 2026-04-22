import time
import logging
import torch
from models.prediction import PredictionResponse
from services import model_service, llm_service
from services.llm_service import llm_classify
from core.config import settings, LABEL_MAP

logger = logging.getLogger(__name__)


def predict_single(
    premise: str,
    hypothesis: str,
    hybrid: bool = False,
    confidence_threshold: float | None = None,
) -> PredictionResponse:
    ms = model_service.state
    ls = llm_service.state

    start = time.perf_counter()

    inputs = ms.tokenizer(
        premise, hypothesis,
        max_length=settings.max_length,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(settings.device)

    with torch.no_grad():
        outputs = ms.model(**inputs)

    probs_raw = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
    pred_model_idx = int(torch.argmax(outputs.logits, dim=-1).item())
    pred_anli_idx = ms.remap[pred_model_idx]
    confidence = max(probs_raw)

    anli_probs = {LABEL_MAP[ms.remap[i]]: round(probs_raw[i], 4) for i in range(3)}
    deberta_ms = (time.perf_counter() - start) * 1000
    deberta_label = LABEL_MAP[pred_anli_idx]

    threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
    routed = False
    llm_model_name = None
    llm_reasoning = None
    llm_label_str = None
    llm_ms = 0.0
    total_ms = deberta_ms

    if hybrid and ls.client is not None and (confidence < threshold or threshold >= 1.0):
        try:
            llm_label, reasoning, llm_ms = llm_classify(premise, hypothesis)
            routed = True
            pred_anli_idx = llm_label
            llm_model_name = settings.llm_model
            llm_reasoning = reasoning
            llm_label_str = LABEL_MAP[llm_label]
            total_ms = deberta_ms + llm_ms
            logger.info(
                f"Routed to LLM (conf={confidence:.3f} < {threshold}): "
                f"DeBERTa={deberta_label} → LLM={llm_label_str} ({llm_ms:.0f}ms)"
            )
        except Exception as e:
            logger.warning(f"LLM fallback to DeBERTa: {e}")

    return PredictionResponse(
        label=LABEL_MAP[pred_anli_idx],
        confidence=round(confidence, 4),
        probabilities=anli_probs,
        inference_time_ms=round(total_ms, 2),
        model=ms.active_model_key,
        routed_to_llm=routed,
        llm_model=llm_model_name,
        llm_reasoning=llm_reasoning,
        deberta_label=deberta_label,
        deberta_time_ms=round(deberta_ms, 2),
        llm_label=llm_label_str,
        llm_time_ms=round(llm_ms, 2) if routed else None,
        confidence_threshold=threshold if hybrid else None,
    )

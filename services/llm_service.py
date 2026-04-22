import re
import time
import logging
from typing import Any
from core.config import settings, LABEL_NAMES

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an expert at Natural Language Inference. Given a premise and hypothesis, "
    "determine the relationship.\n\n"
    "**Definitions:**\n"
    "- **entailment**: The hypothesis is definitely true given the premise.\n"
    "- **neutral**: The hypothesis might or might not be true; the premise doesn't give enough information.\n"
    "- **contradiction**: The hypothesis is definitely false given the premise.\n\n"
    "Think step-by-step, then state your final answer as exactly one of: entailment, neutral, contradiction."
)

_USER_TEMPLATE = (
    "**Premise:** {premise}\n\n"
    "**Hypothesis:** {hypothesis}\n\n"
    "Think step-by-step, then give your **Final Answer:**"
)


class LLMState:
    # Typed Any because InferenceClient is an optional dependency
    client: Any = None


state = LLMState()


def init_llm_client():
    try:
        from huggingface_hub import InferenceClient
        state.client = InferenceClient(model=settings.llm_model, token=settings.hf_token)
        logger.info(f"LLM client initialized: {settings.llm_model}")
    except ImportError:
        logger.warning("huggingface_hub not installed — hybrid routing disabled.")
        state.client = None
    except Exception as e:
        logger.warning(f"Failed to init LLM client: {e} — hybrid routing disabled.")
        state.client = None


def extract_label(text: str) -> int:
    text_lower = text.lower().strip()

    final_match = re.search(
        r'final\s*answer[:\s]*\*{0,2}\s*(entailment|neutral|contradiction)', text_lower
    )
    if final_match:
        return LABEL_NAMES.index(final_match.group(1))

    last_pos = -1
    last_label = None
    for i, lbl in enumerate(LABEL_NAMES):
        pos = text_lower.rfind(lbl)
        if pos > last_pos:
            last_pos = pos
            last_label = i

    return last_label if last_label is not None else 1  # fallback: neutral


def llm_classify(premise: str, hypothesis: str) -> tuple[int, str, float]:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(premise=premise, hypothesis=hypothesis)},
    ]

    start = time.perf_counter()
    try:
        response = state.client.chat_completion(messages=messages, max_tokens=512, temperature=0.1)
        text = response.choices[0].message.content
        label = extract_label(text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return label, text, elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning(f"LLM call failed ({elapsed_ms:.0f}ms): {e}")
        raise

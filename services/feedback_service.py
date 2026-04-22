import json
import logging
import os
from core.config import settings

logger = logging.getLogger(__name__)


def load_feedback() -> list[dict]:
    if os.path.isfile(settings.feedback_file):
        try:
            with open(settings.feedback_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_feedback(entries: list[dict]):
    with open(settings.feedback_file, "w") as f:
        json.dump(entries, f, indent=2, default=str)


def append_feedback(entry: dict):
    entries = load_feedback()
    entries.append(entry)
    save_feedback(entries)
    logger.info(f"Feedback saved: id={entry['id']} correct={entry['is_correct']}")


def compute_stats(entries: list[dict]) -> dict:
    if not entries:
        return {
            "total": 0, "correct": 0, "incorrect": 0, "accuracy": None,
            "corrections_by_label": {}, "corrections_by_model": {},
            "avg_confidence_correct": None, "avg_confidence_incorrect": None,
        }

    correct = [e for e in entries if e.get("is_correct")]
    incorrect = [e for e in entries if not e.get("is_correct")]

    corrections_by_label: dict[str, int] = {}
    for e in incorrect:
        cl = e.get("correct_label") or "not_provided"
        corrections_by_label[cl] = corrections_by_label.get(cl, 0) + 1

    corrections_by_model: dict[str, int] = {}
    for e in incorrect:
        m = e.get("model") or "unknown"
        corrections_by_model[m] = corrections_by_model.get(m, 0) + 1

    correct_confs = [e["confidence"] for e in correct if e.get("confidence") is not None]
    incorrect_confs = [e["confidence"] for e in incorrect if e.get("confidence") is not None]

    return {
        "total": len(entries),
        "correct": len(correct),
        "incorrect": len(incorrect),
        "accuracy": round(len(correct) / len(entries), 4),
        "corrections_by_label": corrections_by_label,
        "corrections_by_model": corrections_by_model,
        "avg_confidence_correct": round(sum(correct_confs) / len(correct_confs), 4) if correct_confs else None,
        "avg_confidence_incorrect": round(sum(incorrect_confs) / len(incorrect_confs), 4) if incorrect_confs else None,
    }

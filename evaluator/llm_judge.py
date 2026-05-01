from __future__ import annotations

import logging
import re
from typing import Any, Dict

from agent.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from llm_client import LLMClient

logger = logging.getLogger(__name__)


_JUDGE_NUMERIC_FIELDS = {
    "accuracy": (1, 5),
    "completeness": (1, 5),
    "clarity": (1, 5),
    "usefulness": (1, 5),
    "average_score": (1.0, 5.0),
    "factual_consistency": (0.0, 1.0),
    "score": (0.0, 1.0),
}

_DEFAULT_DISABLED_JUDGE = {
    "enabled": False,
    "score": 0.0,
    "score_scale": "0-1",
    "accuracy": 0,
    "completeness": 0,
    "clarity": 0,
    "usefulness": 0,
    "average_score": 0.0,
    "factual_consistency": 0.0,
    "fully_correct": False,
    "critical_errors": [],
    "unsupported_claims": [],
    "scoring_point_matches": [],
}

# Keep old name as alias for backward compat with other modules
DEFAULT_DISABLED_JUDGE = _DEFAULT_DISABLED_JUDGE


def _regex_fallback_extract(raw_text: str) -> Dict[str, Any]:
    """Extract numeric judge fields from raw LLM text when JSON parsing fails."""
    result: Dict[str, Any] = {}
    for field, (lo, hi) in _JUDGE_NUMERIC_FIELDS.items():
        # Match patterns like "accuracy": 4 or accuracy: 3 or "accuracy": 0.85
        pattern = rf'"{field}"\s*:\s*([\d.]+)'
        match = re.search(pattern, raw_text)
        if not match:
            pattern = rf'{field}\s*[:=]\s*([\d.]+)'
            match = re.search(pattern, raw_text)
        if match:
            try:
                val = float(match.group(1))
                result[field] = val if lo <= val <= hi else lo
            except ValueError:
                result[field] = lo
        else:
            result[field] = lo
    bool_match = re.search(r'"fully_correct"\s*:\s*(true|false)', raw_text, re.IGNORECASE)
    result["fully_correct"] = bool(bool_match and bool_match.group(1).lower() == "true")
    result["critical_errors"] = []
    result["unsupported_claims"] = []
    result["scoring_point_matches"] = []
    return result


class LLMJudge:
    """Unified LLM Judge with qwen_eval-compatible dimensions."""

    def __init__(self, llm: LLMClient, enabled: bool = True) -> None:
        self.llm = llm
        self.enabled = enabled

    def judge(self, question: str, reference: str, prediction: str, scoring_points: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return dict(DEFAULT_DISABLED_JUDGE)
        if not self.llm.available:
            return dict(DEFAULT_DISABLED_JUDGE)

        prompt = build_judge_user_prompt(question, reference, prediction, scoring_points)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        # Single LLM call; try JSON parse → repair → regex fallback on the same response
        raw_response = self.llm.chat(messages, temperature=0.1)
        if raw_response.error or not raw_response.content:
            logger.warning("LLM Judge API call failed for question: %s; error: %s", question[:80], raw_response.error)
            return dict(DEFAULT_DISABLED_JUDGE)

        text = raw_response.content.strip()
        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        start = text.find("{")
        if start >= 0:
            text = text[start:]

        # Try 1: direct JSON parse
        import json
        try:
            end = text.rfind("}") + 1
            if end > start:
                result = json.loads(text[:end])
                if isinstance(result, dict):
                    return normalize_unified_judge(result, enabled=True)
        except json.JSONDecodeError:
            pass

        # Try 2: JSON repair (handles truncated responses)
        from llm_client import _repair_truncated_json
        repaired = _repair_truncated_json(text)
        if repaired is not None:
            logger.info("JSON repair succeeded for question: %s", question[:60])
            return normalize_unified_judge(repaired, enabled=True)

        # Try 3: regex fallback (extracts core numeric fields from raw text)
        extracted = _regex_fallback_extract(raw_response.content)
        if extracted.get("accuracy", 0) > 0 or extracted.get("factual_consistency", 0) > 0:
            logger.info("Regex fallback succeeded for question: %s", question[:60])
            return normalize_unified_judge(extracted, enabled=True)

        logger.warning("All parse attempts failed for question: %s", question[:80])
        return dict(DEFAULT_DISABLED_JUDGE)


def normalize_unified_judge(result: Dict[str, Any], enabled: bool = True) -> Dict[str, Any]:
    payload = dict(DEFAULT_DISABLED_JUDGE)
    payload.update(result or {})
    payload["enabled"] = enabled
    payload["score_scale"] = "0-1"

    for key in ["accuracy", "completeness", "clarity", "usefulness"]:
        payload[key] = _clamp_int(payload.get(key), 1, 5) if enabled else 0

    qwen_scores = [float(payload[key]) for key in ["accuracy", "completeness", "clarity", "usefulness"]]
    average_score = payload.get("average_score")
    payload["average_score"] = _clamp_float(average_score, 1.0, 5.0) if average_score is not None else 0.0
    if payload["average_score"] == 0.0:
        payload["average_score"] = sum(qwen_scores) / len(qwen_scores)

    payload["factual_consistency"] = _clamp_float(payload.get("factual_consistency"), 0.0, 1.0)
    payload["score"] = compute_llm_score(payload)
    payload["fully_correct"] = bool(payload.get("fully_correct")) if enabled else False
    for key in ["critical_errors", "unsupported_claims", "scoring_point_matches"]:
        value = payload.get(key)
        payload[key] = value if isinstance(value, list) else []
    return {key: payload[key] for key in DEFAULT_DISABLED_JUDGE}


def compute_llm_score(judge: Dict[str, Any]) -> float:
    accuracy = _normalize_qwen_score(judge.get("accuracy"))
    completeness = _normalize_qwen_score(judge.get("completeness"))
    usefulness = _normalize_qwen_score(judge.get("usefulness"))
    clarity = _normalize_qwen_score(judge.get("clarity"))
    factual = _clamp_float(judge.get("factual_consistency"), 0.0, 1.0)
    return round(
        0.35 * accuracy
        + 0.25 * completeness
        + 0.20 * factual
        + 0.10 * usefulness
        + 0.10 * clarity,
        6,
    )


def _normalize_qwen_score(value: Any) -> float:
    return (_clamp_float(value, 1.0, 5.0) - 1.0) / 4.0


def _clamp_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        number = int(round(float(value)))
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))


def _clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except Exception:
        number = minimum
    return max(minimum, min(maximum, number))

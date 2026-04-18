from __future__ import annotations

from typing import Any, Dict

from llm_client import LLMClient


DEFAULT_DISABLED_JUDGE = {
    "enabled": False,
    "score": 0.0,
    "score_scale": "0-1",
    "accuracy": 0,
    "completeness": 0,
    "clarity": 0,
    "usefulness": 0,
    "average_score": 0.0,
    "factual_consistency": 0.0,
}


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

        prompt = f"""你是一个严格、公正的中文技术问答评测器。请对比参考答案和预测答案，输出唯一 JSON，不要输出 Markdown。
评分要求：
1. accuracy/completeness/clarity/usefulness 使用 1-5 分，5 分最好，保持与 qwen_eval.py 可比。
2. factual_consistency 使用 0-1 小数。
3. score 使用 0-1 小数，按以下公式给出：
   score = 0.35 * AccuracyNorm + 0.25 * CompletenessNorm + 0.20 * FactualConsistency + 0.10 * UsefulnessNorm + 0.10 * ClarityNorm
   其中 AccuracyNorm/CompletenessNorm/UsefulnessNorm/ClarityNorm = (对应 1-5 分 - 1) / 4。
4. 如果预测答案编造型号、参数、波形、事实来源或关键原理，应显著降低 accuracy、factual_consistency 和 score。
5. 如果预测答案表达不同但技术含义正确，不要因为字面不同扣重分。

必须输出 JSON 字段：
{{
  "score": 0.0,
  "accuracy": 1,
  "completeness": 1,
  "clarity": 1,
  "usefulness": 1,
  "average_score": 1.0,
  "factual_consistency": 0.0
}}

问题：
{question}

参考答案：
{reference}

预测答案：
{prediction}

采分点命中结果：
{scoring_points}
"""
        result, error = self.llm.json_chat(
            [
                {"role": "system", "content": "你是专业、严格、可复现的中文技术答案质量评估工具。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        if error:
            return dict(DEFAULT_DISABLED_JUDGE)
        return normalize_unified_judge(result, enabled=True)


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

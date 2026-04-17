from __future__ import annotations

from typing import Any, Dict

from llm_client import LLMClient


DEFAULT_DISABLED_JUDGE = {
    "enabled": False,
    "score": None,
    "semantic_correctness": None,
    "completeness": None,
    "factual_errors": [],
    "missing_points": [],
    "off_topic": None,
    "clarity": None,
    "comment": "LLM Judge disabled or unavailable.",
}


class LLMJudge:
    def __init__(self, llm: LLMClient, enabled: bool = False) -> None:
        self.llm = llm
        self.enabled = enabled

    def judge(self, question: str, reference: str, prediction: str, scoring_points: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or not self.llm.available:
            return dict(DEFAULT_DISABLED_JUDGE)
        prompt = f"""请作为严格的答案质量评测器，对比参考答案和预测答案。只输出 JSON。
字段：
score: 0到1的小数；
semantic_correctness: 0到1；
completeness: 0到1；
factual_errors: 字符串数组；
missing_points: 字符串数组；
off_topic: true/false；
clarity: 0到1；
comment: 简短中文评语。

问题：{question}
参考答案：{reference}
预测答案：{prediction}
采分点结果：{scoring_points}
"""
        result, error = self.llm.json_chat(
            [
                {"role": "system", "content": "你是可靠的中文技术问答评测器。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        if error:
            payload = dict(DEFAULT_DISABLED_JUDGE)
            payload.update({"enabled": True, "score": 0.0, "comment": error})
            return payload
        result["enabled"] = True
        return result


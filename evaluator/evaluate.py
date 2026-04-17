from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from configs.config import RuntimeConfig
from evaluator.jaccard_eval import bigram_jaccard
from evaluator.llm_judge import LLMJudge
from evaluator.rouge_eval import rouge_l
from evaluator.scoring_points import judge_scoring_points
from evaluator.semantic_similarity import SemanticSimilarity
from llm_client import LLMClient
from schemas import StandardSample


@dataclass
class SampleEvaluation:
    sample_id: str
    semantic_similarity: Dict[str, Any]
    rouge_l: float
    bigram_jaccard: float
    llm_judge: Dict[str, Any]
    scoring_points: Dict[str, Any]
    final_score: float
    error_analysis: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "semantic_similarity": self.semantic_similarity,
            "rouge_l": self.rouge_l,
            "bigram_jaccard": self.bigram_jaccard,
            "llm_judge": self.llm_judge,
            "scoring_points": self.scoring_points,
            "final_score": self.final_score,
            "error_analysis": self.error_analysis,
        }


class Evaluator:
    def __init__(self, config: RuntimeConfig) -> None:
        eval_cfg = config.raw.get("evaluation", {})
        model_cfg = config.raw.get("model", {})
        self.weights = eval_cfg.get("final_weights", {})
        self.semantic = SemanticSimilarity(
            backend=eval_cfg.get("semantic_backend", "auto"),
            bertscore_model=eval_cfg.get("bertscore_model", "bert-base-chinese"),
            bertscore_num_layers=eval_cfg.get("bertscore_num_layers"),
            sentence_model=eval_cfg.get("sentence_transformer_model", "paraphrase-multilingual-MiniLM-L12-v2"),
        )
        judge_llm = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.judge_model,
            temperature=float(model_cfg.get("temperature", 0.2)),
            max_tokens=int(model_cfg.get("max_tokens", 2000)),
        )
        self.judge = LLMJudge(judge_llm, enabled=bool(eval_cfg.get("enable_llm_judge", False)))

    def evaluate(self, sample: StandardSample, prediction: str) -> SampleEvaluation:
        reference = sample.reference_answer
        semantic = self.semantic.score(reference, prediction)
        rouge = rouge_l(reference, prediction)
        jaccard = bigram_jaccard(reference, prediction)
        scoring_points = judge_scoring_points(reference, prediction).to_json()
        judge_result = self.judge.judge(sample.question_text, reference, prediction, scoring_points)
        final = (
            float(self.weights.get("semantic_similarity", 0.60)) * semantic.score
            + float(self.weights.get("rouge_l", 0.25)) * rouge
            + float(self.weights.get("bigram_jaccard", 0.15)) * jaccard
        )
        error_analysis = self._error_analysis(semantic.score, rouge, jaccard, scoring_points)
        return SampleEvaluation(
            sample_id=sample.sample_id,
            semantic_similarity={"score": semantic.score, "backend": semantic.backend, "error": semantic.error},
            rouge_l=rouge,
            bigram_jaccard=jaccard,
            llm_judge=judge_result,
            scoring_points=scoring_points,
            final_score=final,
            error_analysis=error_analysis,
        )

    def _error_analysis(self, semantic: float, rouge: float, jaccard: float, scoring_points: Dict[str, Any]) -> Dict[str, Any]:
        reasons: list[str] = []
        if semantic < 0.55:
            reasons.append("语义相似度偏低")
        if rouge < 0.35:
            reasons.append("参考答案关键表达覆盖不足")
        if jaccard < 0.25:
            reasons.append("关键词、数字或组件名字面命中不足")
        if float(scoring_points.get("coverage", 0.0)) < 0.5:
            reasons.append("采分点覆盖率不足")
        return {"reasons": reasons, "severity": "high" if len(reasons) >= 2 else "medium" if reasons else "low"}

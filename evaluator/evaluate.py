from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from configs.config import RuntimeConfig
from evaluator.jaccard_eval import build_technical_idf, technical_entity_match
from evaluator.llm_judge import DEFAULT_DISABLED_JUDGE, LLMJudge
from evaluator.rouge_eval import claim_rouge_l
from evaluator.scoring_points import judge_scoring_points
from evaluator.semantic_similarity import SemanticSimilarity
from llm_client import LLMClient
from schemas import StandardSample
from tools.dataset_parser import DatasetParser


@dataclass
class SampleEvaluation:
    sample_id: str
    semantic_similarity: Dict[str, Any]
    llm_judge: Dict[str, Any]
    scoring_points: Dict[str, Any]
    final_score: float
    claim_rouge_l: Dict[str, Any]
    technical_entity_match: Dict[str, Any]
    fully_correct: bool
    error_analysis: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "semantic_similarity": self.semantic_similarity,
            "llm_judge": self.llm_judge,
            "scoring_points": self.scoring_points,
            "final_score": self.final_score,
            "claim_rouge_l": self.claim_rouge_l,
            "technical_entity_match": self.technical_entity_match,
            "fully_correct": self.fully_correct,
            "error_analysis": self.error_analysis,
        }


class Evaluator:
    def __init__(self, config: RuntimeConfig) -> None:
        eval_cfg = config.raw.get("evaluation", {})
        model_cfg = config.raw.get("model", {})
        timeout = int(config.raw.get("runtime", {}).get("request_timeout_seconds", 20))
        self.weights = eval_cfg.get("final_weights", {})
        self.technical_idf = self._build_technical_idf(config)
        self.semantic = SemanticSimilarity(
            backend=eval_cfg.get("semantic_backend", "auto"),
            bertscore_model=eval_cfg.get("bertscore_model", "bert-base-chinese"),
            bertscore_num_layers=eval_cfg.get("bertscore_num_layers"),
            sentence_model=eval_cfg.get("sentence_transformer_model", "paraphrase-multilingual-MiniLM-L12-v2"),
            device=eval_cfg.get("device"),
        )
        judge_llm = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.judge_model,
            temperature=float(model_cfg.get("temperature", 0.2)),
            max_tokens=int(eval_cfg.get("judge_max_tokens", model_cfg.get("max_tokens", 4000))),
            timeout=timeout,
        )
        self.judge = LLMJudge(judge_llm, enabled=bool(eval_cfg.get("enable_llm_judge", True)))

    def evaluate(self, sample: StandardSample, prediction: str, use_llm_judge: bool = True) -> SampleEvaluation:
        reference = sample.reference_answer
        semantic = self.semantic.score(reference, prediction)
        claim_rouge = claim_rouge_l(reference, prediction)
        scoring_points = judge_scoring_points(reference, prediction).to_json()
        technical_entities = technical_entity_match(
            reference,
            prediction,
            support_text=sample.question_text,
            idf=self.technical_idf,
            support_aliases=self._scoring_point_aliases(scoring_points),
        )
        judge_result = (
            self.judge.judge(sample.question_text, reference, prediction, scoring_points)
            if use_llm_judge
            else self.judge_disabled()
        )
        final = self._final_score(
            judge_result,
            semantic.score,
            claim_rouge.score,
            technical_entities.score,
            scoring_points,
        )
        fully_correct = self._fully_correct(judge_result, scoring_points)
        error_analysis = self._error_analysis(
            semantic.score,
            claim_rouge.score,
            technical_entities.score,
            scoring_points,
            judge_result,
            final,
            fully_correct,
        )
        return SampleEvaluation(
            sample_id=sample.sample_id,
            semantic_similarity={"score": semantic.score, "backend": semantic.backend, "error": semantic.error},
            llm_judge=judge_result,
            scoring_points=scoring_points,
            final_score=final,
            claim_rouge_l=claim_rouge.to_json(),
            technical_entity_match=technical_entities.to_json(),
            fully_correct=fully_correct,
            error_analysis=error_analysis,
        )

    def _build_technical_idf(self, config: RuntimeConfig) -> Dict[str, float]:
        try:
            samples = DatasetParser(config.dataset_path, config.image_root).load()
        except Exception:  # noqa: BLE001
            return {}
        documents = [f"{sample.question_text}\n{sample.reference_answer}" for sample in samples]
        return build_technical_idf(documents)

    def _scoring_point_aliases(self, scoring_points: Dict[str, Any]) -> list[str]:
        aliases: list[str] = []
        for point in scoring_points.get("structured_points", []) or []:
            aliases.extend(str(alias) for alias in point.get("aliases", []) or [])
            text = point.get("text")
            if text:
                aliases.append(str(text))
        return aliases

    def judge_disabled(self) -> Dict[str, Any]:
        return dict(DEFAULT_DISABLED_JUDGE)

    def _final_score(
        self,
        judge_result: Dict[str, Any],
        semantic: float,
        claim_rouge: float,
        technical_entity: float,
        scoring_points: Dict[str, Any],
    ) -> float:
        weights = {
            "llm_judge": float(self.weights.get("llm_judge", 0.45)),
            "structured_point_coverage": float(self.weights.get("structured_point_coverage", 0.25)),
            "semantic_similarity": float(self.weights.get("semantic_similarity", 0.20)),
            "claim_rouge_l": float(self.weights.get("claim_rouge_l", 0.05)),
            "technical_entity_match": float(self.weights.get("technical_entity_match", 0.05)),
        }
        values = {
            "llm_judge": float(judge_result.get("score", 0.0) or 0.0) if judge_result.get("enabled") else None,
            "structured_point_coverage": scoring_points.get("coverage"),
            "semantic_similarity": semantic,
            "claim_rouge_l": claim_rouge,
            "technical_entity_match": technical_entity,
        }
        weighted_sum = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            value = values.get(key)
            if value is None:
                continue
            weighted_sum += weight * float(value)
            total_weight += weight
        score = weighted_sum / total_weight if total_weight else 0.0
        if self._has_core_contradiction(scoring_points):
            score = min(score, 0.45)
        return round(score, 6)

    def _fully_correct(self, judge_result: Dict[str, Any], scoring_points: Dict[str, Any]) -> bool:
        if self._has_core_contradiction(scoring_points):
            return False
        if scoring_points.get("critical_errors"):
            return False
        coverage = scoring_points.get("coverage")
        required_coverage = scoring_points.get("required_coverage")
        if coverage is not None and float(coverage) < 0.80:
            return False
        if required_coverage is not None and float(required_coverage) < 0.80:
            return False
        if scoring_points.get("core_conclusion_hit") is False and required_coverage is not None:
            return False
        if judge_result.get("enabled"):
            return (
                int(judge_result.get("accuracy") or 0) >= 4
                and int(judge_result.get("completeness") or 0) >= 4
                and float(judge_result.get("factual_consistency") or 0.0) >= 0.85
                and float(judge_result.get("score") or 0.0) >= 0.80
                and not judge_result.get("critical_errors")
            )
        return coverage is not None and float(coverage) >= 0.90

    def _has_core_contradiction(self, scoring_points: Dict[str, Any]) -> bool:
        return any(
            match.get("type") == "core_conclusion" and match.get("status") == "contradicted"
            for match in scoring_points.get("matches", []) or []
        )

    def _error_analysis(
        self,
        semantic: float,
        claim_rouge: float,
        technical_entity: float,
        scoring_points: Dict[str, Any],
        judge_result: Dict[str, Any],
        final_score: float,
        fully_correct: bool,
    ) -> Dict[str, Any]:
        reasons: list[str] = []
        if semantic < 0.55:
            reasons.append("语义相似度偏低")
        if claim_rouge < 0.20:
            reasons.append("参考答案 claim 覆盖不足")
        if technical_entity < 0.20:
            reasons.append("技术实体覆盖不足或偏离参考答案")
        coverage = scoring_points.get("coverage")
        if coverage is None:
            reasons.append("参考答案未抽取到有效结构化采分点")
        elif float(coverage) < 0.5:
            reasons.append("采分点覆盖率不足")
        if self._has_core_contradiction(scoring_points):
            reasons.append("核心结论与参考答案矛盾")
        if scoring_points.get("unsupported_key_tokens"):
            reasons.append("存在参考答案未支持的关键技术实体")
        if judge_result.get("enabled") and float(judge_result.get("score", 0.0) or 0.0) < 0.5:
            reasons.append("LLM Judge 综合质量评分偏低")
        if not fully_correct:
            reasons.append("未达到完全正确判定阈值")
        return {
            "reasons": reasons,
            "severity": "high" if len(reasons) >= 2 or final_score < 0.45 else "medium" if reasons else "low",
        }

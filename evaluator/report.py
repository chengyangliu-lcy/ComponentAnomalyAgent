from __future__ import annotations

from typing import Any, Dict, Iterable, List


DEFAULT_FINAL_WEIGHTS = {
    "llm_judge": 0.45,
    "structured_point_coverage": 0.25,
    "semantic_similarity": 0.20,
    "claim_rouge_l": 0.05,
    "technical_entity_match": 0.05,
}

LLM_JUDGE_SCORE_WEIGHTS = {
    "accuracy_norm": 0.35,
    "completeness_norm": 0.25,
    "factual_consistency": 0.20,
    "usefulness_norm": 0.10,
    "clarity_norm": 0.10,
}


def summarize_scores(
    rows: Iterable[Dict[str, Any]],
    final_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    items = list(rows)
    if not items:
        return {"total": 0}
    final_weights = final_weights or DEFAULT_FINAL_WEIGHTS
    totals = {
        "final_score": _avg(item.get("final_score", 0.0) for item in items),
        "semantic_similarity": _avg(item.get("semantic_similarity", {}).get("score", 0.0) for item in items),
        "claim_rouge_l": _avg(item.get("claim_rouge_l", {}).get("score", 0.0) for item in items),
        "technical_entity_match": _avg(item.get("technical_entity_match", {}).get("score", 0.0) for item in items),
        "technical_entity_precision": _avg(item.get("technical_entity_match", {}).get("precision", 0.0) for item in items),
        "technical_entity_recall": _avg(item.get("technical_entity_match", {}).get("recall", 0.0) for item in items),
        "technical_entity_f_beta": _avg(item.get("technical_entity_match", {}).get("f_beta", 0.0) for item in items),
        "technical_entity_unsupported_weight_rate": _avg(
            item.get("technical_entity_match", {}).get("unsupported_entity_rate", 0.0) for item in items
        ),
        "scoring_point_coverage": _avg(
            item.get("scoring_points", {}).get("coverage")
            for item in items
            if item.get("scoring_points", {}).get("coverage") is not None
        ),
        "required_point_coverage": _avg(
            item.get("scoring_points", {}).get("required_coverage")
            for item in items
            if item.get("scoring_points", {}).get("required_coverage") is not None
        ),
    }
    coverage_null_count = sum(1 for item in items if item.get("scoring_points", {}).get("coverage") is None)
    fully_correct_count = sum(1 for item in items if item.get("fully_correct"))
    critical_error_count = sum(
        1
        for item in items
        if item.get("scoring_points", {}).get("critical_errors")
        or item.get("llm_judge", {}).get("critical_errors")
    )
    core_conclusion_items = [
        item for item in items if item.get("scoring_points", {}).get("required_coverage") is not None
    ]
    core_conclusion_hit_count = sum(1 for item in core_conclusion_items if item.get("scoring_points", {}).get("core_conclusion_hit"))
    unsupported_entity_rate = _avg(
        item.get("technical_entity_match", {}).get("unsupported_entity_rate", 0.0) for item in items
    )
    judge_keys = [
        "score",
        "accuracy",
        "completeness",
        "clarity",
        "usefulness",
        "average_score",
        "factual_consistency",
    ]
    llm_judge = {}
    for key in judge_keys:
        values = [
            item.get("llm_judge", {}).get(key)
            for item in items
            if item.get("llm_judge", {}).get("enabled")
            and isinstance(item.get("llm_judge", {}).get(key), (int, float))
        ]
        if values:
            llm_judge[key] = round(_avg(values), 4)
    summary = {
        "total": len(items),
        "final_score": round(totals["final_score"], 4),
        "scoring_rule": {
            "final_score": {
                "formula": "weighted average of llm_judge, structured_point_coverage, semantic_similarity, claim_rouge_l, and technical_entity_match; disabled judge or null coverage is reweighted",
                "weights": _round_weights(final_weights),
            },
            "llm_judge_score": {
                "formula": "weighted average of normalized accuracy, normalized completeness, factual_consistency, normalized usefulness, and normalized clarity",
                "weights": _round_weights(LLM_JUDGE_SCORE_WEIGHTS),
            },
        },
        "averages": {key: round(value, 4) for key, value in totals.items()},
        "quality_rates": {
            "fully_correct_rate": round(fully_correct_count / len(items), 4),
            "critical_error_rate": round(critical_error_count / len(items), 4),
            "core_conclusion_hit_rate": round(core_conclusion_hit_count / len(core_conclusion_items), 4)
            if core_conclusion_items
            else 0.0,
            "unsupported_entity_rate": round(unsupported_entity_rate, 4),
            "coverage_null_count": coverage_null_count,
        },
    }
    if llm_judge:
        summary["llm_judge"] = llm_judge
    return summary


def _avg(values: Iterable[float]) -> float:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else 0.0


def _round_weights(weights: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in weights.items()}


def build_error_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "summary": summarize_scores(rows),
        "samples": [
            {
                "sample_id": row.get("sample_id"),
                "final_score": row.get("final_score"),
                "llm_judge_score": row.get("llm_judge", {}).get("score"),
                "fully_correct": row.get("fully_correct"),
                "reasons": row.get("error_analysis", {}).get("reasons", []),
                "missed_points": row.get("scoring_points", {}).get("missed_points", [])[:8],
            }
            for row in sorted(rows, key=lambda item: item.get("final_score", 0.0))[:30]
        ],
    }

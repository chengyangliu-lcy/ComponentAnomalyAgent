from __future__ import annotations

from typing import Any, Dict, Iterable, List


DEFAULT_FINAL_WEIGHTS = {
    "llm_judge": 0.50,
    "semantic_similarity": 0.25,
    "scoring_point_coverage": 0.10,
    "rouge_l": 0.10,
    "bigram_jaccard": 0.05,
}

DEFAULT_LEGACY_FINAL_WEIGHTS = {
    "semantic_similarity": 0.60,
    "rouge_l": 0.25,
    "bigram_jaccard": 0.15,
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
    legacy_final_weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    items = list(rows)
    if not items:
        return {"total": 0}
    final_weights = final_weights or DEFAULT_FINAL_WEIGHTS
    legacy_final_weights = legacy_final_weights or DEFAULT_LEGACY_FINAL_WEIGHTS
    totals = {
        "final_score": _avg(item.get("final_score", 0.0) for item in items),
        "legacy_final_score": _avg(item.get("legacy_final_score", item.get("final_score", 0.0)) for item in items),
        "semantic_similarity": _avg(item.get("semantic_similarity", {}).get("score", 0.0) for item in items),
        "rouge_l": _avg(item.get("rouge_l", 0.0) for item in items),
        "bigram_jaccard": _avg(item.get("bigram_jaccard", 0.0) for item in items),
        "scoring_point_coverage": _avg(item.get("scoring_points", {}).get("coverage", 0.0) for item in items),
    }
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
        "legacy_final_score": round(totals["legacy_final_score"], 4),
        "scoring_rule": {
            "final_score": {
                "formula": "weighted average of llm_judge, semantic_similarity, scoring_point_coverage, rouge_l, and bigram_jaccard",
                "weights": _round_weights(final_weights),
            },
            "llm_judge_score": {
                "formula": "weighted average of normalized accuracy, normalized completeness, factual_consistency, normalized usefulness, and normalized clarity",
                "weights": _round_weights(LLM_JUDGE_SCORE_WEIGHTS),
            },
            "legacy_final_score": {
                "formula": "weighted average of semantic_similarity, rouge_l, and bigram_jaccard",
                "weights": _round_weights(legacy_final_weights),
            },
        },
        "averages": {key: round(value, 4) for key, value in totals.items()},
    }
    if llm_judge:
        summary["llm_judge"] = llm_judge
    return summary


def _avg(values: Iterable[float]) -> float:
    nums = [float(value) for value in values]
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
                "legacy_final_score": row.get("legacy_final_score"),
                "llm_judge_score": row.get("llm_judge", {}).get("score"),
                "reasons": row.get("error_analysis", {}).get("reasons", []),
                "missed_points": row.get("scoring_points", {}).get("missed_points", [])[:8],
            }
            for row in sorted(rows, key=lambda item: item.get("final_score", 0.0))[:30]
        ],
    }

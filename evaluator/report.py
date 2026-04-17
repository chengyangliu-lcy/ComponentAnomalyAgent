from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List


def summarize_scores(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    if not items:
        return {"total": 0}
    totals = {
        "final_score": _avg(item.get("final_score", 0.0) for item in items),
        "semantic_similarity": _avg(item.get("semantic_similarity", {}).get("score", 0.0) for item in items),
        "rouge_l": _avg(item.get("rouge_l", 0.0) for item in items),
        "bigram_jaccard": _avg(item.get("bigram_jaccard", 0.0) for item in items),
        "scoring_point_coverage": _avg(item.get("scoring_points", {}).get("coverage", 0.0) for item in items),
    }
    judge_scores = [
        item.get("llm_judge", {}).get("score")
        for item in items
        if isinstance(item.get("llm_judge", {}).get("score"), (int, float))
    ]
    if judge_scores:
        totals["llm_judge"] = _avg(judge_scores)
    severities = Counter(item.get("error_analysis", {}).get("severity", "unknown") for item in items)
    return {
        "total": len(items),
        "averages": {key: round(value, 4) for key, value in totals.items()},
        "error_severity_counts": dict(severities),
        "lowest_samples": sorted(
            [{"sample_id": item.get("sample_id"), "final_score": item.get("final_score", 0.0)} for item in items],
            key=lambda item: item["final_score"],
        )[:10],
    }


def _avg(values: Iterable[float]) -> float:
    nums = [float(value) for value in values]
    return sum(nums) / len(nums) if nums else 0.0


def build_error_analysis(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "summary": summarize_scores(rows),
        "samples": [
            {
                "sample_id": row.get("sample_id"),
                "final_score": row.get("final_score"),
                "reasons": row.get("error_analysis", {}).get("reasons", []),
                "missed_points": row.get("scoring_points", {}).get("missed_points", [])[:8],
            }
            for row in sorted(rows, key=lambda item: item.get("final_score", 0.0))[:30]
        ],
    }


from __future__ import annotations

from typing import Any, Dict, List

from evaluator.report import summarize_scores


def compare_runs(baseline_rows: List[Dict[str, Any]], agent_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    baseline_by_id = {row["sample_id"]: row for row in baseline_rows if row.get("sample_id")}
    agent_by_id = {row["sample_id"]: row for row in agent_rows if row.get("sample_id")}
    shared = sorted(set(baseline_by_id) & set(agent_by_id))
    deltas = []
    for sample_id in shared:
        baseline_score = float(baseline_by_id[sample_id].get("final_score", 0.0))
        agent_score = float(agent_by_id[sample_id].get("final_score", 0.0))
        metric_deltas = _sample_metric_deltas(baseline_by_id[sample_id], agent_by_id[sample_id])
        deltas.append(
            {
                "sample_id": sample_id,
                "baseline": baseline_score,
                "agent": agent_score,
                "delta": agent_score - baseline_score,
                **{f"{key}_delta": value for key, value in metric_deltas.items()},
            }
        )
    metric_names = ["llm_judge", "semantic_similarity", "claim_rouge_l", "technical_entity_match"]
    return {
        "shared_samples": len(shared),
        "baseline_summary": summarize_scores(
            [baseline_by_id[sample_id] for sample_id in shared],
            common_sample_count=len(shared),
        ),
        "agent_summary": summarize_scores(
            [agent_by_id[sample_id] for sample_id in shared],
            common_sample_count=len(shared),
        ),
        "average_delta": round(sum(item["delta"] for item in deltas) / len(deltas), 4) if deltas else 0.0,
        "metric_deltas": {
            metric: round(sum(item.get(f"{metric}_delta", 0.0) for item in deltas) / len(deltas), 4)
            if deltas
            else 0.0
            for metric in metric_names
        },
        "all_sample_deltas": sorted(deltas, key=lambda item: item["sample_id"]),
        "improved_samples": sorted([item for item in deltas if item["delta"] > 0], key=lambda item: item["delta"], reverse=True)[:30],
        "regressed_samples": sorted([item for item in deltas if item["delta"] < 0], key=lambda item: item["delta"])[:30],
    }


def _sample_metric_deltas(baseline: Dict[str, Any], agent: Dict[str, Any]) -> Dict[str, float]:
    return {
        "llm_judge": round(_nested_score(agent, "llm_judge") - _nested_score(baseline, "llm_judge"), 4),
        "semantic_similarity": round(
            _nested_score(agent, "semantic_similarity") - _nested_score(baseline, "semantic_similarity"),
            4,
        ),
        "claim_rouge_l": round(_nested_score(agent, "claim_rouge_l") - _nested_score(baseline, "claim_rouge_l"), 4),
        "technical_entity_match": round(
            _nested_score(agent, "technical_entity_match") - _nested_score(baseline, "technical_entity_match"),
            4,
        ),
    }


def _nested_score(row: Dict[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, dict):
        return float(value.get("score") or 0.0)
    return float(value or 0.0)

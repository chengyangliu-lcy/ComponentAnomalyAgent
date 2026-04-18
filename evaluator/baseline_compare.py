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
        baseline_legacy = float(baseline_by_id[sample_id].get("legacy_final_score", baseline_score))
        agent_legacy = float(agent_by_id[sample_id].get("legacy_final_score", agent_score))
        deltas.append(
            {
                "sample_id": sample_id,
                "baseline": baseline_score,
                "agent": agent_score,
                "delta": agent_score - baseline_score,
                "baseline_legacy": baseline_legacy,
                "agent_legacy": agent_legacy,
                "legacy_delta": agent_legacy - baseline_legacy,
            }
        )
    return {
        "shared_samples": len(shared),
        "baseline_summary": summarize_scores([baseline_by_id[sample_id] for sample_id in shared]),
        "agent_summary": summarize_scores([agent_by_id[sample_id] for sample_id in shared]),
        "average_delta": round(sum(item["delta"] for item in deltas) / len(deltas), 4) if deltas else 0.0,
        "improved_samples": sorted([item for item in deltas if item["delta"] > 0], key=lambda item: item["delta"], reverse=True)[:30],
        "regressed_samples": sorted([item for item in deltas if item["delta"] < 0], key=lambda item: item["delta"])[:30],
    }

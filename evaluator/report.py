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
    predictions: Iterable[Dict[str, Any]] | None = None,
    common_sample_count: int | None = None,
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
    summary["sample_counts"] = _sample_counts(items, common_sample_count=common_sample_count)
    kb_diagnostics = _kb_diagnostics(list(predictions or []))
    if kb_diagnostics:
        summary["kb_diagnostics"] = kb_diagnostics
    return summary


def _avg(values: Iterable[float]) -> float:
    nums = [float(value) for value in values if value is not None]
    return sum(nums) / len(nums) if nums else 0.0


def _round_weights(weights: Dict[str, float]) -> Dict[str, float]:
    return {key: round(float(value), 4) for key, value in weights.items()}


def _sample_counts(items: list[Dict[str, Any]], common_sample_count: int | None = None) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    missing = 0
    for item in items:
        sample_id = str(item.get("sample_id") or "")
        if not sample_id:
            missing += 1
            continue
        counts[sample_id] = counts.get(sample_id, 0) + 1
    duplicate_sample_count = sum(1 for count in counts.values() if count > 1)
    duplicate_row_count = sum(count - 1 for count in counts.values() if count > 1)
    result: Dict[str, Any] = {
        "rows": len(items),
        "unique_samples": len(counts),
        "duplicate_sample_count": duplicate_sample_count,
        "duplicate_row_count": duplicate_row_count,
        "missing_sample_id_rows": missing,
    }
    if common_sample_count is not None:
        result["common_samples"] = int(common_sample_count)
    return result


def _kb_diagnostics(predictions: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not predictions:
        return {}
    calls = 0
    nonempty = 0
    chunks = 0
    kb_candidate_count = 0
    kb_used_count = 0
    kb_discarded_count = 0
    noise_filtered_count = 0
    low_value_source_filtered_count = 0
    low_value_project_filtered_count = 0
    required_terms_filtered_count = 0
    high_relevance_count = 0
    index_path = ""
    index_exists: bool | None = None
    answer_used = 0
    selected_samples = 0
    for pred in predictions:
        selected = [
            action.get("tool_name")
            for action in (pred.get("plan") or {}).get("selected_actions", []) or []
            if isinstance(action, dict)
        ]
        if "local_retrieve" in selected:
            selected_samples += 1
        answer = str(pred.get("answer") or "")
        if "local_kb_chunk" in answer or "本地" in answer and "知识库" in answer:
            answer_used += 1
        for event in pred.get("tool_trace", []) or []:
            if not isinstance(event, dict):
                continue
            if event.get("tool_name") != "local_retrieve" and event.get("action") != "circuit_md_fts_search":
                continue
            calls += 1
            metadata = (event.get("outputs") or {}).get("metadata") or {}
            status = metadata.get("index_status") or {}
            if metadata.get("index_dir"):
                index_path = str(metadata.get("index_dir"))
            if status.get("db_path"):
                index_path = str(status.get("db_path"))
            if isinstance(status.get("exists"), bool):
                index_exists = bool(status.get("exists"))
            event_chunks = int(metadata.get("chunks") or 0)
            evidence_delta = int((event.get("outputs") or {}).get("evidence_delta") or 0)
            count = max(event_chunks, evidence_delta, len((event.get("outputs") or {}).get("sources") or []))
            chunks += count
            kb_candidate_count += int(metadata.get("kb_candidate_count") or count)
            kb_used_count += int(metadata.get("kb_used_count") or count)
            kb_discarded_count += int(metadata.get("kb_discarded_count") or 0)
            noise_filtered_count += int(metadata.get("noise_filtered_count") or 0)
            low_value_source_filtered_count += int(metadata.get("low_value_source_filtered_count") or 0)
            low_value_project_filtered_count += int(metadata.get("low_value_project_filtered_count") or 0)
            required_terms_filtered_count += int(metadata.get("required_terms_filtered_count") or 0)
            high_relevance_count += int(metadata.get("high_relevance_count") or 0)
            if count > 0:
                nonempty += 1
    if calls == 0 and selected_samples == 0:
        return {}
    return {
        "local_retrieve_selected_samples": selected_samples,
        "local_retrieve_calls": calls,
        "local_retrieve_nonempty_rate": round(nonempty / calls, 4) if calls else 0.0,
        "avg_chunks": round(chunks / calls, 4) if calls else 0.0,
        "index_path": index_path,
        "index_exists": index_exists,
        "kb_evidence_used_rate": round(answer_used / len(predictions), 4) if predictions else 0.0,
        "kb_candidate_count": kb_candidate_count,
        "kb_used_count": kb_used_count,
        "kb_discarded_count": kb_discarded_count,
        "noise_filtered_count": noise_filtered_count,
        "low_value_source_filtered_count": low_value_source_filtered_count,
        "low_value_project_filtered_count": low_value_project_filtered_count,
        "required_terms_filtered_count": required_terms_filtered_count,
        "high_relevance_count": high_relevance_count,
        "high_relevance_rate": round(high_relevance_count / kb_used_count, 4) if kb_used_count else 0.0,
    }


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

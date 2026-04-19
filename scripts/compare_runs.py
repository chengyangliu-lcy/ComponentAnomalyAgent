from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluator.baseline_compare import compare_runs
from tools.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare unified agent and baseline evaluation JSONL files.")
    parser.add_argument("--agent", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--agent-predictions", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--duplicates",
        choices=["error", "keep-last"],
        default="error",
        help="How to handle duplicate sample_id rows in eval inputs. Default errors to prevent accidental unfair compare.",
    )
    parser.add_argument(
        "--sample-set",
        choices=["error", "shared"],
        default="error",
        help="How to handle different agent/baseline sample sets. Default errors; use shared only for diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_rows = _read_jsonl(Path(args.agent), duplicates=args.duplicates, label="agent")
    baseline_rows = _read_jsonl(Path(args.baseline), duplicates=args.duplicates, label="baseline")
    _check_sample_sets(agent_rows, baseline_rows, mode=args.sample_set)
    report = compare_runs(baseline_rows, agent_rows)
    report["sample_set"] = _sample_set_report(agent_rows, baseline_rows)
    report["win_rate"] = _win_rate(report)
    prediction_path = Path(args.agent_predictions) if args.agent_predictions else Path(args.agent).with_name("predictions.jsonl")
    if prediction_path.exists():
        predictions = _read_jsonl(prediction_path, duplicates="keep-last", label="agent predictions")
        report["agent_runtime"] = _agent_runtime_stats(predictions)
    write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _read_jsonl(path: Path, duplicates: str = "error", label: str = "jsonl") -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    seen: dict[str, int] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("post_id") or "")
        if not sample_id:
            continue
        seen[sample_id] = seen.get(sample_id, 0) + 1
        if seen[sample_id] == 2:
            duplicate_ids.append(sample_id)
    if duplicate_ids and duplicates == "error":
        examples = ", ".join(duplicate_ids[:10])
        raise SystemExit(
            f"{label} file has duplicate sample_id rows: {len(duplicate_ids)} duplicates. "
            f"Examples: {examples}. Re-run cleanly or pass --duplicates keep-last."
        )
    if duplicate_ids and duplicates == "keep-last":
        latest: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in rows:
            sample_id = str(row.get("sample_id") or row.get("post_id") or "")
            if not sample_id:
                continue
            if sample_id not in latest:
                order.append(sample_id)
            latest[sample_id] = row
        rows = [latest[sample_id] for sample_id in order]
    return rows


def _sample_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("sample_id") or row.get("post_id") or "") for row in rows if row.get("sample_id") or row.get("post_id")}


def _check_sample_sets(agent_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], mode: str) -> None:
    agent_ids = _sample_ids(agent_rows)
    baseline_ids = _sample_ids(baseline_rows)
    missing_in_agent = sorted(baseline_ids - agent_ids)
    missing_in_baseline = sorted(agent_ids - baseline_ids)
    if mode == "error" and (missing_in_agent or missing_in_baseline):
        raise SystemExit(
            "agent and baseline eval files do not contain the same sample_id set. "
            f"missing_in_agent={len(missing_in_agent)} examples={missing_in_agent[:10]}; "
            f"missing_in_baseline={len(missing_in_baseline)} examples={missing_in_baseline[:10]}. "
            "Retry failed generation/evaluation first, or pass --sample-set shared only for diagnostics."
        )


def _sample_set_report(agent_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]]) -> dict[str, Any]:
    agent_ids = _sample_ids(agent_rows)
    baseline_ids = _sample_ids(baseline_rows)
    return {
        "agent_samples": len(agent_ids),
        "baseline_samples": len(baseline_ids),
        "shared_samples": len(agent_ids & baseline_ids),
        "missing_in_agent": sorted(baseline_ids - agent_ids)[:30],
        "missing_in_baseline": sorted(agent_ids - baseline_ids)[:30],
    }


def _win_rate(report: dict[str, Any]) -> float:
    improved = len(report.get("improved_samples", []))
    regressed = len(report.get("regressed_samples", []))
    tied = max(0, int(report.get("shared_samples", 0)) - improved - regressed)
    total = improved + regressed + tied
    return round(improved / total, 4) if total else 0.0


def _agent_runtime_stats(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = 0
    final_timeouts = 0
    action_counts: dict[str, int] = {}
    tool_sequences: dict[str, int] = {}
    for row in predictions:
        stop = str((row.get("plan") or {}).get("final_stop_reason") or "")
        if "fallback finish" in stop:
            fallback += 1
        sequence = []
        for action in (row.get("plan") or {}).get("selected_actions", []) or []:
            tool_name = str(action.get("tool_name") or "")
            if tool_name:
                action_counts[tool_name] = action_counts.get(tool_name, 0) + 1
                sequence.append(tool_name)
        if sequence:
            key = ">".join(sequence)
            tool_sequences[key] = tool_sequences.get(key, 0) + 1
        errors = " ".join(str(error) for error in row.get("errors", []) or [])
        trace_errors = " ".join(str(event.get("error") or "") for event in row.get("tool_trace", []) or [])
        if "timed out" in errors.lower() or "timed out" in trace_errors.lower() or "request timed out" in trace_errors.lower():
            final_timeouts += int(any((event.get("tool_name") == "finish_answer" and event.get("error")) for event in row.get("tool_trace", []) or []))
    total = len(predictions)
    return {
        "samples": total,
        "fallback_finish_count": fallback,
        "fallback_finish_rate": round(fallback / total, 4) if total else 0.0,
        "final_answer_timeout_count": final_timeouts,
        "action_counts": action_counts,
        "top_tool_sequences": sorted(
            [{"sequence": key, "count": value} for key, value in tool_sequences.items()],
            key=lambda item: item["count"],
            reverse=True,
        )[:10],
    }


if __name__ == "__main__":
    main()

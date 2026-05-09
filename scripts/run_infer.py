from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.pipeline import AgentPipeline, result_path
from configs.config import load_config
from tools.dense_retriever import DenseRetriever
from tools.dataset_parser import DatasetParser
from tools.sample_ids import filter_items_by_sample_ids, read_sample_ids_file
from tools.utils import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ComponentAnomalyAgent inference.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="agent_run")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--sample-ids-file", default=None, help="Text file with one sample_id/post_id per line.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--enable-web", action="store_true")
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--enable-local-retrieval", action="store_true")
    parser.add_argument("--disable-local-retrieval", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent samples to infer. Defaults to runtime.max_workers.")
    return parser.parse_args()


def _build_shared_dense_retriever(config: Any) -> Any:
    """Build and load a DenseRetriever once to share across all workers.

    Loads both the embedding model and index so each worker thread reuses
    the same GPU memory instead of loading N copies (~4.6GB each).
    """
    retrieval_cfg = dict(config.raw.get("retrieval", {}))
    embedding_model = retrieval_cfg.get("embedding_model", "")
    if not embedding_model:
        return None
    from pathlib import Path
    raw_path = config.raw.get("paths", {}).get("local_kb_index", "")
    kb_dir = Path(raw_path) if raw_path else Path("knowledge_base/circuit_diagnosis_fts")
    if kb_dir.is_absolute():
        index_dir = kb_dir
    else:
        index_dir = Path(__file__).resolve().parents[1] / kb_dir
    retriever = DenseRetriever(
        model_name=embedding_model,
        index_dir=index_dir,
        device=retrieval_cfg.get("device", "cuda"),
        batch_size=int(retrieval_cfg.get("embedding_batch_size", 64)),
    )
    dense_status = retriever.status()
    if not dense_status.usable:
        print(f"[infer] shared DenseRetriever not usable: {dense_status.error or 'index missing'}")
        return None
    retriever.load_index()
    retriever.load_model()
    print(f"[infer] shared DenseRetriever loaded: model={embedding_model} chunks={dense_status.chunk_count} dim={dense_status.embedding_dim}")
    return retriever


def main() -> None:
    args = parse_args()
    overrides: dict[str, Any] = {}
    if args.enable_web or args.disable_web:
        overrides.setdefault("agent", {})["enable_web_search"] = bool(args.enable_web and not args.disable_web)
    if args.enable_local_retrieval or args.disable_local_retrieval:
        overrides.setdefault("agent", {})["enable_local_retrieval"] = bool(
            args.enable_local_retrieval and not args.disable_local_retrieval
        )
    config = load_config(args.config, overrides=overrides or None)
    config.ensure_dirs()
    parser = DatasetParser(config.dataset_path, config.image_root)
    samples = parser.load()
    if args.sample_id:
        samples = [sample for sample in samples if sample.sample_id == args.sample_id or sample.post_id == args.sample_id]
        if not samples:
            raise SystemExit(f"sample not found: {args.sample_id}")
    sample_ids = read_sample_ids_file(args.sample_ids_file)
    if sample_ids:
        before = len(samples)
        samples = filter_items_by_sample_ids(samples, sample_ids, lambda sample: sample.sample_id)
        if not samples:
            raise SystemExit(f"no samples matched --sample-ids-file: {args.sample_ids_file}")
        missing = len(sample_ids) - len(samples)
        print(f"[infer] sample_ids_file={args.sample_ids_file} matched={len(samples)}/{before} missing={missing}")
    if args.limit is not None:
        samples = samples[: args.limit]
    output_path = Path(args.output) if args.output else result_path(config.outputs_dir, args.experiment, "predictions.jsonl")
    summary_path = output_path.with_suffix(".summary.json")
    trace_dir = output_path.parent / "traces"
    seen = set()
    if output_path.exists() and args.no_resume:
        output_path.unlink()
    if output_path.exists() and not args.no_resume:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["sample_id"])
    pending_samples = [sample for sample in samples if sample.sample_id not in seen]
    skipped = len(samples) - len(pending_samples)
    max_workers = max(1, int(args.max_workers or config.raw.get("runtime", {}).get("max_workers", 1)))
    shared_retriever = _build_shared_dense_retriever(config) if max_workers > 1 else None
    print(
        f"[infer] experiment={args.experiment} requested={len(samples)} "
        f"pending={len(pending_samples)} skipped={skipped} max_workers={max_workers} "
        f"shared_retriever={'yes' if shared_retriever else 'no'} output={output_path}"
    )
    completed = 0
    hard_failed = 0
    warning_samples = 0
    start = time.perf_counter()
    progress = tqdm(pending_samples, desc="agent生成", unit="sample")
    if max_workers == 1:
        pipeline = AgentPipeline(config)
        for sample in progress:
            progress.set_postfix_str(f"id={sample.sample_id}")
            started = time.perf_counter()
            try:
                row = pipeline.run_sample(sample).to_json()
            except Exception as exc:  # noqa: BLE001
                row = _failed_result(sample, exc, time.perf_counter() - started)
            append_jsonl(output_path, row)
            _write_trace(trace_dir, row)
            if _is_hard_failed(row):
                hard_failed += 1
            elif row.get("errors"):
                warning_samples += 1
            completed += 1
            _update_progress(progress, row)
    else:
        worker_state = threading.local()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_sample_with_worker_pipeline, config, worker_state, sample, shared_retriever): sample
                for sample in pending_samples
            }
            for future in as_completed(futures):
                sample = futures[future]
                progress.set_postfix_str(f"id={sample.sample_id}")
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = _failed_result(sample, exc, 0.0)
                append_jsonl(output_path, row)
                _write_trace(trace_dir, row)
                if _is_hard_failed(row):
                    hard_failed += 1
                elif row.get("errors"):
                    warning_samples += 1
                completed += 1
                _update_progress(progress, row)
                progress.update(1)
        progress.close()
    elapsed = time.perf_counter() - start
    write_json(
        summary_path,
        {
            "experiment": args.experiment,
            "requested_samples": len(samples),
            "skipped_by_resume": skipped,
            "new_completed": completed,
            "failed": hard_failed,
            "hard_failed": hard_failed,
            "warning_samples": warning_samples,
            "output": str(output_path),
            "trace_dir": str(trace_dir),
            "model": config.agent_model,
            "web_enabled": bool(config.raw.get("agent", {}).get("enable_web_search")),
            "max_workers": max_workers,
            "elapsed_seconds": round(elapsed, 4),
        },
    )
    print(
        f"[infer] finished completed={completed} failed={hard_failed} warnings={warning_samples} skipped={skipped} "
        f"elapsed={elapsed:.2f}s summary={summary_path}"
    )


def _run_sample_with_worker_pipeline(config: Any, worker_state: threading.local, sample: Any, shared_retriever: Any = None) -> dict[str, Any]:
    pipeline = getattr(worker_state, "pipeline", None)
    if pipeline is None:
        pipeline = AgentPipeline(config, shared_dense_retriever=shared_retriever)
        worker_state.pipeline = pipeline
    started = time.perf_counter()
    try:
        return pipeline.run_sample(sample).to_json()
    except Exception as exc:  # noqa: BLE001
        return _failed_result(sample, exc, time.perf_counter() - started)


def _failed_result(sample: Any, exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "question": sample.question_text,
        "answer": "",
        "tools_used": [],
        "web_searched": False,
        "tool_trace": [],
        "reasoning_summary": "inference failed before final answer generation.",
        "elapsed_seconds": round(elapsed, 4),
        "token_usage": {},
        "errors": [str(exc)],
        "plan": None,
    }


def _is_hard_failed(row: dict[str, Any]) -> bool:
    if not row.get("errors"):
        return False
    if not str(row.get("answer") or "").strip():
        return True
    if str(row.get("reasoning_summary") or "").startswith("inference failed before final answer generation"):
        return True
    return False


def _write_trace(trace_dir: Path, row: dict[str, Any]) -> None:
    sample_id = str(row.get("sample_id") or "unknown")
    answer = row.get("answer", "")
    steps = _compact_tool_trace(row.get("tool_trace", []), answer=answer)
    # Insert question as first user message
    steps.insert(0, {"step": 0, "role": "user", "content": row.get("question", "")})
    trace_payload = {
        "sample_id": sample_id,
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "errors": row.get("errors", []),
        "steps": steps,
    }
    write_json(trace_dir / f"{sample_id}.trace.json", trace_payload)


def _compact_tool_trace(events: list[dict[str, Any]], answer: str = "") -> list[dict[str, Any]]:
    """Reconstruct agent conversation as system/user/assistant/tool steps."""
    steps = []
    step_num = 0

    for evt in events:
        tool = evt.get("tool_name", "")
        action = evt.get("action", "")
        outputs = evt.get("outputs", {})

        # Skip bootstrap events
        if tool == "agent_planner" and action == "initialize_loop":
            continue
        if tool == "openhands_sdk":
            continue

        # Planner select_action → assistant message
        if tool == "agent_planner" and action == "select_action":
            step_num += 1
            effective = outputs.get("effective_action") or outputs.get("validated_action") or {}
            chosen = effective.get("tool_name", "")
            args = effective.get("args", {})
            reason = effective.get("reason", "")
            # Strip _meta from args
            clean_args = {k: v for k, v in args.items() if k != "_meta" and v is not None}
            assistant_msg = {"tool": chosen}
            if clean_args:
                assistant_msg["args"] = clean_args
            if reason:
                assistant_msg["reason"] = reason
            steps.append({"step": step_num, "role": "assistant", "content": assistant_msg})
            continue

        # Planner rejected action → assistant error
        if tool == "agent_planner" and action != "select_action":
            step_num += 1
            steps.append({"step": step_num, "role": "assistant", "content": {"error": evt.get("summary", "")}})
            continue

        # Tool execution → tool message
        obs = outputs.get("observation", {})
        evidences = obs.get("evidence", [])
        tool_content: dict[str, Any] = {"tool": tool, "success": evt.get("success", False)}

        # Extract key info from inputs
        inputs = evt.get("inputs", {})
        if inputs.get("query"):
            tool_content["query"] = inputs["query"]

        # finish_answer: attach the final answer text
        if tool == "finish_answer" and answer:
            tool_content["answer"] = answer
        elif evidences:
            # Evidence: only keep source + content (skip title which is often redundant)
            tool_content["evidence"] = [
                {"source": e.get("source"), "content": e.get("content")}
                for e in evidences
            ]

        # Error
        if evt.get("error"):
            tool_content["error"] = evt["error"]

        steps.append({"step": step_num, "role": "tool", "content": tool_content})

    return steps


def _trace_stats(row: dict[str, Any]) -> dict[str, Any]:
    events = row.get("tool_trace", []) or []
    planner_events = [event for event in events if event.get("tool_name") == "agent_planner"]
    search_events = [event for event in events if event.get("tool_name") == "web_search"]
    return {
        "event_count": len(events),
        "successful_events": sum(1 for event in events if event.get("success")),
        "failed_events": sum(1 for event in events if not event.get("success")),
        "tool_sequence": [event.get("tool_name") for event in events],
        "action_sequence": [
            event.get("outputs", {}).get("action") or event.get("action")
            for event in events
        ],
        "recoverable_error_count": sum(
            1 for event in events if event.get("outputs", {}).get("recoverable_error")
        ),
        "planner_events": [
            {
                "action": event.get("action"),
                "summary": event.get("summary"),
                "outputs": event.get("outputs", {}),
            }
            for event in planner_events
        ],
        "search_event_count": len(search_events),
        "search_queries": [
            event.get("inputs", {}).get("query")
            for event in search_events
            if event.get("inputs", {}).get("query")
        ],
    }


def _update_progress(progress: tqdm, row: dict[str, Any]) -> None:
    sample_id = row.get("sample_id", "")
    tools = row.get("tools_used", [])
    elapsed = float(row.get("elapsed_seconds", 0.0) or 0.0)
    errors = row.get("errors", [])
    progress.set_postfix(
        {
            "id": sample_id,
            "tools": len(tools),
            "sec": f"{elapsed:.1f}",
            "err": len(errors),
        }
    )
    tqdm.write(
        f"[infer] done sample_id={sample_id} tools={tools} "
        f"web={row.get('web_searched')} elapsed={elapsed:.2f}s errors={len(errors)}"
    )


if __name__ == "__main__":
    main()

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
from tools.dataset_parser import DatasetParser
from tools.utils import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ComponentAnomalyAgent inference.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="agent_run")
    parser.add_argument("--sample-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--enable-web", action="store_true")
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent samples to infer. Defaults to runtime.max_workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = None
    if args.enable_web or args.disable_web:
        overrides = {"agent": {"enable_web_search": bool(args.enable_web and not args.disable_web)}}
    config = load_config(args.config, overrides=overrides)
    config.ensure_dirs()
    parser = DatasetParser(config.dataset_path, config.image_root)
    samples = parser.load()
    if args.sample_id:
        samples = [sample for sample in samples if sample.sample_id == args.sample_id or sample.post_id == args.sample_id]
        if not samples:
            raise SystemExit(f"sample not found: {args.sample_id}")
    if args.limit is not None:
        samples = samples[: args.limit]
    output_path = Path(args.output) if args.output else result_path(config.outputs_dir, args.experiment, "predictions.jsonl")
    summary_path = output_path.with_suffix(".summary.json")
    seen = set()
    if output_path.exists() and not args.no_resume:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["sample_id"])
    pending_samples = [sample for sample in samples if sample.sample_id not in seen]
    skipped = len(samples) - len(pending_samples)
    max_workers = max(1, int(args.max_workers or config.raw.get("runtime", {}).get("max_workers", 1)))
    print(
        f"[infer] experiment={args.experiment} requested={len(samples)} "
        f"pending={len(pending_samples)} skipped={skipped} max_workers={max_workers} output={output_path}"
    )
    completed = 0
    failed = 0
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
            if row.get("errors"):
                failed += 1
            completed += 1
            _update_progress(progress, row)
    else:
        worker_state = threading.local()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_sample_with_worker_pipeline, config, worker_state, sample): sample
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
                if row.get("errors"):
                    failed += 1
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
            "failed": failed,
            "output": str(output_path),
            "model": config.agent_model,
            "web_enabled": bool(config.raw.get("agent", {}).get("enable_web_search")),
            "max_workers": max_workers,
            "elapsed_seconds": round(elapsed, 4),
        },
    )
    print(
        f"[infer] finished completed={completed} failed={failed} skipped={skipped} "
        f"elapsed={elapsed:.2f}s summary={summary_path}"
    )


def _run_sample_with_worker_pipeline(config: Any, worker_state: threading.local, sample: Any) -> dict[str, Any]:
    pipeline = getattr(worker_state, "pipeline", None)
    if pipeline is None:
        pipeline = AgentPipeline(config)
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

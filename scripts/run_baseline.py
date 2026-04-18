from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from evaluator.evaluate import Evaluator
from evaluator.report import summarize_scores
from tools.dataset_parser import DatasetParser
from tools.utils import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adapt qwen_eval.py baseline results to unified scoring.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--baseline-results", default=None, help="Existing qwen_eval.py JSONL output.")
    parser.add_argument("--run-qwen-eval", action="store_true", help="Run qwen_eval.py before adapting.")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent samples to evaluate. Defaults to runtime.max_workers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    max_workers = max(1, int(args.max_workers or config.raw.get("runtime", {}).get("max_workers", 1)))
    if args.run_qwen_eval:
        subprocess.run(
            [sys.executable, str(ROOT / "qwen_eval.py"), "--max-workers", str(max_workers)],
            cwd=str(ROOT),
            check=True,
        )
    baseline_path = Path(args.baseline_results or ROOT / "evaluation_results.jsonl")
    if not baseline_path.exists():
        raise SystemExit(
            f"baseline result file not found: {baseline_path}. Run qwen_eval.py or pass --baseline-results."
        )
    dataset = {sample.sample_id: sample for sample in DatasetParser(config.dataset_path, config.image_root).load()}
    raw_rows = []
    with baseline_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_rows.append(json.loads(line))
    if args.limit is not None:
        raw_rows = raw_rows[: args.limit]
    output_path = Path(args.output) if args.output else config.outputs_dir / args.experiment / "baseline_eval_results.jsonl"
    if output_path.exists():
        output_path.unlink()
    print(
        f"[baseline] experiment={args.experiment} raw_rows={len(raw_rows)} "
        f"baseline_results={baseline_path} output={output_path}"
    )

    eval_items = []
    skipped = 0
    for row in raw_rows:
        sample_id = str(row.get("sample_id") or row.get("post_id"))
        sample = dataset.get(sample_id)
        if not sample:
            skipped += 1
            tqdm.write(f"[baseline] skip missing sample: {sample_id}")
            continue
        prediction = row.get("answer") or row.get("generated_answer") or ""
        eval_items.append((sample, prediction))

    eval_rows = []
    print(f"[baseline] pending={len(eval_items)} skipped={skipped} max_workers={max_workers}")
    start = time.perf_counter()
    if max_workers == 1:
        print("[baseline] initializing unified evaluator...")
        evaluator = Evaluator(config)
        print("[baseline] evaluator ready")
        progress = tqdm(eval_items, desc="baseline unified eval", unit="sample")
        for sample, prediction in progress:
            progress.set_postfix_str(f"id={sample.sample_id}")
            eval_row = _evaluate_with_evaluator(evaluator, sample, prediction)
            eval_rows.append(eval_row)
            append_jsonl(output_path, eval_row)
            _update_progress(progress, eval_row)
    else:
        worker_state = threading.local()
        progress = tqdm(total=len(eval_items), desc="baseline unified eval", unit="sample")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_with_worker_evaluator, config, worker_state, sample, prediction): sample
                for sample, prediction in eval_items
            }
            for future in as_completed(futures):
                sample = futures[future]
                progress.set_postfix_str(f"id={sample.sample_id}")
                try:
                    eval_row = future.result()
                except Exception as exc:  # noqa: BLE001
                    eval_row = _failed_eval_result(sample.sample_id, exc, 0.0)
                eval_rows.append(eval_row)
                append_jsonl(output_path, eval_row)
                _update_progress(progress, eval_row)
                progress.update(1)
        progress.close()

    eval_cfg = config.raw.get("evaluation", {})
    write_json(
        config.outputs_dir / args.experiment / "baseline_score.json",
        summarize_scores(
            eval_rows,
            final_weights=eval_cfg.get("final_weights"),
            legacy_final_weights=eval_cfg.get("legacy_final_weights"),
        ),
    )
    elapsed = time.perf_counter() - start
    print(
        f"[baseline] finished evaluated={len(eval_rows)} skipped={skipped} "
        f"max_workers={max_workers} elapsed={elapsed:.2f}s wrote={output_path}"
    )


def _evaluate_with_worker_evaluator(
    config: Any,
    worker_state: threading.local,
    sample: Any,
    prediction: str,
) -> dict[str, Any]:
    evaluator = getattr(worker_state, "evaluator", None)
    if evaluator is None:
        evaluator = Evaluator(config)
        worker_state.evaluator = evaluator
    return _evaluate_with_evaluator(evaluator, sample, prediction)


def _evaluate_with_evaluator(evaluator: Evaluator, sample: Any, prediction: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        row = evaluator.evaluate(sample, prediction).to_json()
    except Exception as exc:  # noqa: BLE001
        return _failed_eval_result(sample.sample_id, exc, time.perf_counter() - started)
    row["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return row


def _failed_eval_result(sample_id: str, exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "semantic_similarity": {"score": 0.0, "backend": "", "error": str(exc)},
        "rouge_l": 0.0,
        "bigram_jaccard": 0.0,
        "llm_judge": {"enabled": False, "score": 0.0},
        "scoring_points": {"coverage": 0.0, "matched_points": [], "missed_points": []},
        "final_score": 0.0,
        "legacy_final_score": 0.0,
        "error_analysis": {"reasons": [str(exc)], "severity": "high"},
        "elapsed_seconds": round(elapsed, 4),
        "errors": [str(exc)],
    }


def _update_progress(progress: tqdm, row: dict[str, Any]) -> None:
    sample_id = row.get("sample_id", "")
    elapsed = float(row.get("elapsed_seconds", 0.0) or 0.0)
    errors = row.get("errors", [])
    progress.set_postfix(
        {
            "id": sample_id,
            "score": f"{float(row['final_score']):.4f}",
            "llm": row.get("llm_judge", {}).get("enabled"),
            "sec": f"{elapsed:.1f}",
            "err": len(errors),
        }
    )
    tqdm.write(
        f"[baseline] done sample_id={sample_id} final_score={float(row['final_score']):.4f} "
        f"legacy={float(row.get('legacy_final_score', 0.0)):.4f} "
        f"llm={float(row.get('llm_judge', {}).get('score', 0.0)):.4f} "
        f"elapsed={elapsed:.2f}s errors={len(errors)}"
    )


if __name__ == "__main__":
    main()

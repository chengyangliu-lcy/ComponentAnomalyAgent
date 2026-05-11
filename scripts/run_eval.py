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

from configs.config import load_config
from evaluator.evaluate import Evaluator
from evaluator.report import build_error_analysis, summarize_scores
from tools.dataset_parser import DatasetParser
from tools.utils import append_jsonl, read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--experiment", default="agent_run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None, help="Concurrent samples to evaluate. Defaults to runtime.max_workers.")
    parser.add_argument(
        "--mode",
        choices=["composite", "local-only"],
        default="composite",
        help="composite: unified LLM Judge plus local metrics; local-only: local metrics without LLM calls.",
    )
    parser.add_argument("--no-resume", action="store_true", help="Delete existing eval results and re-evaluate all samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    dataset = {sample.sample_id: sample for sample in DatasetParser(config.dataset_path, config.image_root).load()}
    predictions = []
    with Path(args.predictions).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    if args.limit is not None:
        predictions = predictions[: args.limit]
    default_name = "eval_results.jsonl"
    output_path = Path(args.output) if args.output else config.outputs_dir / args.experiment / default_name
    seen = set()
    if output_path.exists() and args.no_resume:
        output_path.unlink()
    if output_path.exists() and not args.no_resume:
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["sample_id"])
    print(
        f"[eval] experiment={args.experiment} mode={args.mode} "
        f"predictions={len(predictions)} output={output_path}"
    )
    eval_items = []
    skipped = 0
    seen_count = 0
    for pred in predictions:
        sample_id = str(pred.get("sample_id"))
        if sample_id in seen:
            seen_count += 1
            continue
        sample = dataset.get(sample_id)
        if not sample:
            skipped += 1
            tqdm.write(f"[eval] skip missing sample: {sample_id}")
            continue
        prediction = pred.get("answer") or pred.get("generated_answer") or ""
        eval_items.append((sample, prediction))

    rows = []
    max_workers = max(1, int(args.max_workers or config.raw.get("runtime", {}).get("max_workers", 1)))
    print(f"[eval] pending={len(eval_items)} skipped={skipped} resumed={seen_count} max_workers={max_workers}")
    start = time.perf_counter()
    use_llm_judge = args.mode != "local-only"
    if max_workers == 1:
        print("[eval] initializing evaluator and metric backends...")
        evaluator = Evaluator(config)
        print("[eval] evaluator ready")
        progress = tqdm(eval_items, desc="unified eval", unit="sample")
        for sample, prediction in progress:
            progress.set_postfix_str(f"id={sample.sample_id}")
            row = _evaluate_with_evaluator(evaluator, sample, prediction, use_llm_judge)
            rows.append(row)
            append_jsonl(output_path, row)
            _update_progress(progress, row)
    else:
        worker_state = threading.local()
        progress = tqdm(total=len(eval_items), desc="unified eval", unit="sample")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_with_worker_evaluator, config, worker_state, sample, prediction, use_llm_judge): sample
                for sample, prediction in eval_items
            }
            for future in as_completed(futures):
                sample = futures[future]
                progress.set_postfix_str(f"id={sample.sample_id}")
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = _failed_eval_result(sample.sample_id, exc, 0.0)
                rows.append(row)
                append_jsonl(output_path, row)
                _update_progress(progress, row)
                progress.update(1)
        progress.close()

    eval_cfg = config.raw.get("evaluation", {})
    _remove_stale_summary(config.outputs_dir / args.experiment / "agent_score.json")
    all_rows = list(read_jsonl(output_path))
    write_json(
        config.outputs_dir / args.experiment / "evaluation_summary.json",
        summarize_scores(
            all_rows,
            final_weights=eval_cfg.get("final_weights"),
            predictions=predictions,
        ),
    )
    write_json(config.outputs_dir / args.experiment / "error_analysis.json", build_error_analysis(all_rows))
    elapsed = time.perf_counter() - start
    print(
        f"[eval] finished evaluated={len(all_rows)} new={len(rows)} skipped={skipped} resumed={seen_count} "
        f"max_workers={max_workers} elapsed={elapsed:.2f}s wrote={output_path}"
    )


def _remove_stale_summary(path: Path) -> None:
    if path.exists():
        path.unlink()


def _evaluate_with_worker_evaluator(
    config: Any,
    worker_state: threading.local,
    sample: Any,
    prediction: str,
    use_llm_judge: bool,
) -> dict[str, Any]:
    evaluator = getattr(worker_state, "evaluator", None)
    if evaluator is None:
        evaluator = Evaluator(config)
        worker_state.evaluator = evaluator
    return _evaluate_with_evaluator(evaluator, sample, prediction, use_llm_judge)


def _evaluate_with_evaluator(evaluator: Evaluator, sample: Any, prediction: str, use_llm_judge: bool) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        row = evaluator.evaluate(sample, prediction, use_llm_judge=use_llm_judge).to_json()
    except Exception as exc:  # noqa: BLE001
        return _failed_eval_result(sample.sample_id, exc, time.perf_counter() - started)
    row["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return row


def _failed_eval_result(sample_id: str, exc: Exception, elapsed: float) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "semantic_similarity": {"score": 0.0, "backend": "", "error": str(exc)},
        "llm_judge": {"enabled": False, "score": 0.0},
        "scoring_points": {"coverage": None, "matched_points": [], "missed_points": [], "critical_errors": []},
        "final_score": 0.0,
        "claim_rouge_l": {"score": 0.0, "claims": [], "claim_scores": []},
        "technical_entity_match": {
            "score": 0.0,
            "reference_entities": [],
            "prediction_entities": [],
            "support_entities": [],
            "matched_entities": [],
            "missed_entities": [],
            "unsupported_entities": [],
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f_beta": 0.0,
            "unsupported_entity_weight": 0.0,
            "unsupported_entity_rate": 0.0,
            "entity_counts": {
                "reference": {"structured": 0, "domain_abbrev": 0, "keyphrase": 0, "total": 0},
                "prediction": {"structured": 0, "domain_abbrev": 0, "keyphrase": 0, "total": 0},
                "support": {"structured": 0, "domain_abbrev": 0, "keyphrase": 0, "total": 0},
                "unsupported": {"structured": 0, "domain_abbrev": 0, "keyphrase": 0, "total": 0},
            },
            "matched_by_type": {"structured": [], "domain_abbrev": [], "keyphrase": []},
            "missed_by_type": {"structured": [], "domain_abbrev": [], "keyphrase": []},
        },
        "fully_correct": False,
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
        f"[eval] done sample_id={sample_id} final_score={float(row['final_score']):.4f} "
        f"llm={float(row.get('llm_judge', {}).get('score', 0.0)):.4f} "
        f"elapsed={elapsed:.2f}s errors={len(errors)}"
    )


if __name__ == "__main__":
    main()

"""Re-evaluate only samples where the LLM Judge previously failed.

Usage:
    python scripts/repair_eval.py \
        --config configs/kb_diagnosis.yaml \
        --experiment agent_full_v4_rerun \
        --max-workers 4

Reads eval_results.jsonl, identifies entries where llm_judge.enabled is False,
re-runs the full evaluation for those samples only, and overwrites those entries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config
from evaluator.evaluate import Evaluator
from tools.dataset_parser import DatasetParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair failed LLM Judge entries in eval results.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_dirs()

    output_dir = config.outputs_dir / args.experiment
    eval_path = output_dir / "eval_results.jsonl"
    predictions_path = output_dir.parent / args.experiment.replace("_rerun", "") / "predictions.jsonl"

    # Try to find predictions in current experiment dir first, then fall back
    if not (output_dir / "predictions.jsonl").exists() and not predictions_path.exists():
        # Search for predictions in related directories
        for candidate in output_dir.parent.iterdir():
            pred = candidate / "predictions.jsonl"
            if pred.exists():
                predictions_path = pred
                break

    if (output_dir / "predictions.jsonl").exists():
        predictions_path = output_dir / "predictions.jsonl"

    # Load dataset and predictions
    dataset = {s.sample_id: s for s in DatasetParser(config.dataset_path, config.image_root).load()}

    predictions = {}
    with Path(predictions_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pred = json.loads(line)
                sample_id = str(pred.get("sample_id"))
                answer = pred.get("answer") or pred.get("generated_answer") or ""
                predictions[sample_id] = answer

    # Load existing eval results
    rows = []
    failed_ids = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
                if not row.get("llm_judge", {}).get("enabled"):
                    failed_ids.append(row.get("sample_id"))

    print(f"[repair] experiment={args.experiment} total_rows={len(rows)} failed_llm_judge={len(failed_ids)}")

    if not failed_ids:
        print("[repair] No failed entries found. Nothing to do.")
        return

    evaluator = Evaluator(config)
    repaired = 0
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id not in failed_ids:
            continue
        prediction = predictions.get(sample_id, "")
        sample = dataset.get(sample_id)
        if not sample or not prediction:
            print(f"[repair] skip {sample_id}: missing sample or prediction")
            continue

        try:
            new_eval = evaluator.evaluate(sample, prediction, use_llm_judge=True).to_json()
            new_eval["elapsed_seconds"] = row.get("elapsed_seconds", 0)
            # Update the row in-place
            idx = rows.index(row)
            rows[idx] = new_eval
            if new_eval.get("llm_judge", {}).get("enabled"):
                repaired += 1
                print(f"[repair] fixed {sample_id}: llm_score={new_eval['llm_judge']['score']:.4f} final={new_eval['final_score']:.4f}")
            else:
                print(f"[repair] still failed {sample_id}: llm_judge not enabled")
        except Exception as exc:
            print(f"[repair] error for {sample_id}: {exc}")

    # Rewrite the entire file
    eval_path.unlink()
    with eval_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[repair] done: repaired={repaired}/{len(failed_ids)} wrote={eval_path}")


if __name__ == "__main__":
    main()
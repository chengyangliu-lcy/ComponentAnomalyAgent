from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config.ensure_dirs()
    if args.run_qwen_eval:
        subprocess.run([sys.executable, str(ROOT / "qwen_eval.py")], cwd=str(ROOT), check=True)
    baseline_path = Path(args.baseline_results or ROOT / "evaluation_results.jsonl")
    if not baseline_path.exists():
        raise SystemExit(
            f"baseline result file not found: {baseline_path}. Run qwen_eval.py or pass --baseline-results."
        )
    dataset = {sample.sample_id: sample for sample in DatasetParser(config.dataset_path, config.image_root).load()}
    rows = []
    with baseline_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[: args.limit]
    output_path = Path(args.output) if args.output else config.outputs_dir / args.experiment / "baseline_eval_results.jsonl"
    if output_path.exists():
        output_path.unlink()
    evaluator = Evaluator(config)
    eval_rows = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("post_id"))
        sample = dataset.get(sample_id)
        if not sample:
            continue
        prediction = row.get("answer") or row.get("generated_answer") or ""
        eval_row = evaluator.evaluate(sample, prediction).to_json()
        eval_rows.append(eval_row)
        append_jsonl(output_path, eval_row)
    write_json(config.outputs_dir / args.experiment / "baseline_score.json", summarize_scores(eval_rows))
    print(f"[baseline] wrote {output_path}")


if __name__ == "__main__":
    main()


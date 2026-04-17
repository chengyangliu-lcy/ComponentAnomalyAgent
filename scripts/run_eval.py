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
from evaluator.report import build_error_analysis, summarize_scores
from tools.dataset_parser import DatasetParser
from tools.utils import append_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--experiment", default="agent_run")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
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
    output_path = Path(args.output) if args.output else config.outputs_dir / args.experiment / "eval_results.jsonl"
    if output_path.exists():
        output_path.unlink()
    evaluator = Evaluator(config)
    rows = []
    for pred in predictions:
        sample_id = str(pred.get("sample_id"))
        sample = dataset.get(sample_id)
        if not sample:
            print(f"[eval] skip missing sample: {sample_id}")
            continue
        row = evaluator.evaluate(sample, pred.get("answer") or pred.get("generated_answer") or "").to_json()
        rows.append(row)
        append_jsonl(output_path, row)
        print(f"[eval] {sample_id} final_score={row['final_score']:.4f}")
    write_json(config.outputs_dir / args.experiment / "agent_score.json", summarize_scores(rows))
    write_json(config.outputs_dir / args.experiment / "error_analysis.json", build_error_analysis(rows))
    print(f"[eval] wrote {output_path}")


if __name__ == "__main__":
    main()


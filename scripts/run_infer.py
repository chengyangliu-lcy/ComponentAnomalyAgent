from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
    pipeline = AgentPipeline(config)
    completed = 0
    for sample in samples:
        if sample.sample_id in seen:
            continue
        result = pipeline.run_sample(sample)
        append_jsonl(output_path, result.to_json())
        completed += 1
        print(f"[infer] {sample.sample_id} done, tools={result.tools_used}, score_input_saved={output_path}")
    write_json(
        summary_path,
        {
            "experiment": args.experiment,
            "requested_samples": len(samples),
            "new_completed": completed,
            "output": str(output_path),
            "model": config.agent_model,
            "web_enabled": bool(config.raw.get("agent", {}).get("enable_web_search")),
        },
    )


if __name__ == "__main__":
    main()

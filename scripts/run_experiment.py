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
from evaluator.baseline_compare import compare_runs
from tools.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference, evaluation, and optional baseline comparison.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--experiment", default="agent_experiment")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--eval-max-workers", type=int, default=None)
    parser.add_argument("--enable-web", action="store_true")
    parser.add_argument("--disable-web", action="store_true")
    parser.add_argument("--baseline-eval", default=None)
    parser.add_argument("--no-resume", action="store_true", help="Disable resume; re-run all stages from scratch.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[experiment]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    exp_dir = config.outputs_dir / args.experiment
    predictions = exp_dir / "predictions.jsonl"
    eval_results = exp_dir / "eval_results.jsonl"
    infer_cmd = [sys.executable, "scripts/run_infer.py", "--experiment", args.experiment]
    if args.config:
        infer_cmd.extend(["--config", args.config])
    if args.limit is not None:
        infer_cmd.extend(["--limit", str(args.limit)])
    if args.max_workers is not None:
        infer_cmd.extend(["--max-workers", str(args.max_workers)])
    if args.enable_web:
        infer_cmd.append("--enable-web")
    if args.disable_web:
        infer_cmd.append("--disable-web")
    if args.no_resume:
        infer_cmd.append("--no-resume")
    pred_count = _count_jsonl_rows(predictions)
    if not args.no_resume and pred_count > 0:
        print(f"[experiment] predictions.jsonl already has {pred_count} rows, skipping inference (use --no-resume to force)")
    else:
        _run(infer_cmd)
    eval_cmd = [sys.executable, "scripts/run_eval.py", "--experiment", args.experiment, "--predictions", str(predictions)]
    if args.config:
        eval_cmd.extend(["--config", args.config])
    eval_workers = args.eval_max_workers or args.max_workers
    if eval_workers is not None:
        eval_cmd.extend(["--max-workers", str(eval_workers)])
    if args.no_resume:
        eval_cmd.append("--no-resume")
    eval_count = _count_jsonl_rows(eval_results)
    if not args.no_resume and eval_count > 0:
        print(f"[experiment] eval_results.jsonl already has {eval_count} rows, skipping evaluation (use --no-resume to force)")
    else:
        _run(eval_cmd)
    if args.baseline_eval:
        baseline_rows = _read_jsonl(Path(args.baseline_eval))
        agent_rows = _read_jsonl(eval_results)
        write_json(exp_dir / "compare_report.json", compare_runs(baseline_rows, agent_rows))


if __name__ == "__main__":
    main()

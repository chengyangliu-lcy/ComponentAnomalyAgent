from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select reproducible random sample IDs from a dataset JSONL.")
    parser.add_argument("--dataset", default="2025_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    with Path(args.dataset).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    sample_ids = [str(row["post_id"]) for row in rows if row.get("post_id")]
    rng = random.Random(args.seed)
    rng.shuffle(sample_ids)
    selected = sample_ids[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    print(
        f"[select_samples] dataset={args.dataset} total={len(sample_ids)} "
        f"selected={len(selected)} seed={args.seed} output={output}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dedupe JSONL rows by sample_id/post_id.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep", choices=["last", "first"], default="last")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _read_jsonl(Path(args.input))
    deduped = dedupe_rows(rows, keep=args.keep)
    _write_jsonl(Path(args.output), deduped)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "rows": len(rows),
                "unique_rows": len(deduped),
                "duplicates_removed": len(rows) - len(deduped),
                "keep": args.keep,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def dedupe_rows(rows: list[dict[str, Any]], keep: str = "last") -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("post_id") or "")
        if not sample_id:
            continue
        if sample_id not in latest:
            order.append(sample_id)
            latest[sample_id] = row
        elif keep == "last":
            latest[sample_id] = row
    return [latest[sample_id] for sample_id in order]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from tools.circuit_kb import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CIRCUIT_MD_ROOT,
    DEFAULT_MAX_DOCS,
    DEFAULT_MIN_CHUNK_CHARS,
    build_circuit_md_kb,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local Circuit Markdown SQLite FTS knowledge base.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-docs", type=int, default=DEFAULT_MAX_DOCS)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--min-chunk-chars", type=int, default=DEFAULT_MIN_CHUNK_CHARS)
    args = parser.parse_args()

    config = load_config(args.config)
    paths_cfg = config.raw.get("paths", {})
    source_dir = Path(args.source_dir or paths_cfg.get("circuit_md_root") or DEFAULT_CIRCUIT_MD_ROOT)
    output_dir = Path(args.output_dir) if args.output_dir else config.local_kb_index
    if not source_dir.is_absolute():
        source_dir = PROJECT_ROOT / source_dir
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if not source_dir.exists():
        raise SystemExit(f"Circuit Markdown source dir not found: {source_dir}")

    meta = build_circuit_md_kb(
        source_dir=source_dir,
        output_dir=output_dir,
        max_docs=args.max_docs,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
    )
    print(f"Built Circuit Markdown KB at {output_dir}")
    print(f"documents={meta['documents']} chunks={meta['chunks']}")
    print(f"stats={meta['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

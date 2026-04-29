from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from tools.hackster_kb import DEFAULT_EMBEDDING_MODEL, build_hackster_kb


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Hackster Common Crawl local knowledge base.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--csv", dest="csv_path", default=None)
    parser.add_argument("--warc-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-chars", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    config = load_config(args.config)
    csv_path = Path(args.csv_path) if args.csv_path else config.knowledge_csv
    warc_root = Path(args.warc_root) if args.warc_root else config.warc_root
    output_dir = Path(args.output_dir) if args.output_dir else config.local_kb_index
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not warc_root.is_absolute():
        warc_root = Path.cwd() / warc_root
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not warc_root.exists():
        raise SystemExit(f"WARC root not found: {warc_root}")

    meta = build_hackster_kb(
        csv_path=csv_path,
        warc_root=warc_root,
        output_dir=output_dir,
        limit=args.limit,
        max_docs=args.max_docs,
        embedding_model=args.embedding_model,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Built Hackster KB at {output_dir}")
    print(f"documents={meta['documents']} chunks={meta['chunks']} embedding={meta['embedding'].get('status')}")
    print(f"stats={meta['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.circuit_kb import (
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_DOCS,
    DEFAULT_MAX_SOURCE_FILE_BYTES,
    DEFAULT_MIN_CHUNK_CHARS,
    build_diagnosis_kb,
)

DEFAULT_WARC_ROOT = Path("/media/work/1ECC291B3E106A4A/xinyang/circuit/warc_output")
DEFAULT_SOURCE_IDS = (
    "c9feda27-eb86-43aa-bc04-d39d69344a8d",  # elecfans.com (Chinese tech articles - primary)
    "98f81c1f-cfda-46a6-91c0-921a7894d20b",  # electronicsforu.com (English articles - secondary)
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a high-precision circuit diagnosis KB from WARC sources.")
    parser.add_argument("--warc-root", default=str(DEFAULT_WARC_ROOT))
    parser.add_argument("--source-id", action="append", default=None, help="UUID subdir under --warc-root. Can be repeated.")
    parser.add_argument("--source-dir", action="append", default=None, help="Explicit source directory. Can be repeated.")
    parser.add_argument("--output-dir", default="knowledge_base/circuit_diagnosis_fts")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_SOURCE_FILE_BYTES)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--min-chunk-chars", type=int, default=DEFAULT_MIN_CHUNK_CHARS)
    args = parser.parse_args()

    warc_root = Path(args.warc_root)
    explicit_source_dirs = [Path(path) for path in args.source_dir or []]
    source_dirs = list(explicit_source_dirs)
    source_ids = args.source_id or ([] if explicit_source_dirs else list(DEFAULT_SOURCE_IDS))
    source_dirs.extend(warc_root / source_id for source_id in source_ids)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    meta = build_diagnosis_kb(
        source_dirs,
        output_dir,
        max_docs=args.max_docs,
        max_file_bytes=args.max_file_bytes,
        progress_every=args.progress_every,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
    )
    print(f"Built diagnosis KB at {output_dir}")
    print(f"documents={meta['documents']} chunks={meta['chunks']}")
    print(f"stats={meta['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
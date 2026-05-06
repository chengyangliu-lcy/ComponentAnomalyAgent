"""Build improved KB with enhanced multi-language noise cleaning.

Routes:
    A) elecfans.com (c9feda27) + electronicsforu.com (98f81c1f)
    B) All 7 original WARC sources plus radiokot and eet-china

Usage:
    # Quick build (top-2 best sources, ~5000 docs)
    uv run python scripts/build_improved_kb.py --preset quick --max-docs 5000

    # Full build (all electronics sources, ~8000 docs)
    uv run python scripts/build_improved_kb.py --preset full --max-docs 8000

    # Custom
    uv run python scripts/build_improved_kb.py --source-id <uuid> --source-id <uuid> --output-dir knowledge_base/my_kb
"""

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
    DEFAULT_MIN_CHUNK_CHARS,
    DEFAULT_MAX_SOURCE_FILE_BYTES,
    build_diagnosis_kb,
)

DEFAULT_WARC_ROOT = Path("/media/work/1ECC291B3E106A4A/xinyang/circuit/warc_output")

PRESETS = {
    "quick": (
        "c9feda27-eb86-43aa-bc04-d39d69344a8d",  # elecfans.com (Chinese primary)
        "98f81c1f-cfda-46a6-91c0-921a7894d20b",  # electronicsforu.com (English secondary)
    ),
    "full": (
        "c9feda27-eb86-43aa-bc04-d39d69344a8d",  # elecfans.com
        "98f81c1f-cfda-46a6-91c0-921a7894d20b",  # electronicsforu.com
        "df5771a0-8fad-4d88-8a57-248fd4ef5f6e",  # radiokot.ru
        "cf80eebf-5d42-4556-b2c3-a4de6966df52",  # mbb.eet-china.com
        "b8ee701e-9911-43a2-9cc0-3a1a1bda8782",  # bbs.eeworld.com.cn
        "c9df5712-eb33-40f1-9d8b-dc5baf39f21d",  # oshwlab.com
        "efedf473-6a15-42fd-ae8c-321460f852f1",  # eet-china.com
    ),
    # Full without bbs.eeworld.com.cn (data leak avoidance) and without
    # oshwlab.com (user profiles / project pages, no diagnostic value) and
    # radiokot.ru (severe Cyrillic encoding damage, test set is Chinese-only)
    "full_no_leak": (
        "c9feda27-eb86-43aa-bc04-d39d69344a8d",  # elecfans.com
        "98f81c1f-cfda-46a6-91c0-921a7894d20b",  # electronicsforu.com
        "cf80eebf-5d42-4556-b2c3-a4de6966df52",  # mbb.eet-china.com
        "efedf473-6a15-42fd-ae8c-321460f852f1",  # eet-china.com
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build improved KB with enhanced noise cleaning")
    parser.add_argument("--warc-root", default=str(DEFAULT_WARC_ROOT))
    parser.add_argument("--source-id", action="append", default=None, help="UUID subdir under --warc-root")
    parser.add_argument("--preset", choices=list(PRESETS), default="quick")
    parser.add_argument("--output-dir", default="knowledge_base/circuit_diagnosis_fts_hq_v3")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_SOURCE_FILE_BYTES)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--chunk-chars", type=int, default=DEFAULT_CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--min-chunk-chars", type=int, default=DEFAULT_MIN_CHUNK_CHARS)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    warc_root = Path(args.warc_root)
    source_ids = args.source_id or list(PRESETS[args.preset])
    source_dirs = [warc_root / sid for sid in source_ids]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    print(f"Building improved KB -> {output_dir}")
    print(f"Sources ({len(source_ids)}): {source_ids}")
    print(f"max_docs={args.max_docs}, chunk_chars={args.chunk_chars}, min_chunk_chars={args.min_chunk_chars}, workers={args.workers}")

    meta = build_diagnosis_kb(
        source_dirs,
        output_dir,
        max_docs=args.max_docs,
        max_file_bytes=args.max_file_bytes,
        progress_every=args.progress_every,
        chunk_chars=args.chunk_chars,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
        workers=args.workers,
    )

    print(f"\nBuild complete: {output_dir}")
    print(f"  documents = {meta['documents']}")
    print(f"  chunks    = {meta['chunks']}")
    print(f"  stats     = {meta['stats']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

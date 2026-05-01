from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import load_config
from tools.dense_retriever import DenseRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dense embedding index for a Circuit Markdown KB.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--kb-dir", default=None, help="KB directory containing circuit_md.sqlite")
    parser.add_argument("--model", default=None, help="Sentence-transformers model name (e.g. BAAI/bge-m3)")
    parser.add_argument("--device", default=None, help="Device for encoding (cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    retrieval_cfg = dict(config.raw.get("retrieval", {}))
    kb_dir = Path(args.kb_dir) if args.kb_dir else config.local_kb_index
    if not kb_dir.is_absolute():
        kb_dir = PROJECT_ROOT / kb_dir
    model_name = args.model or retrieval_cfg.get("embedding_model", "BAAI/bge-m3")
    device = args.device or retrieval_cfg.get("device", "cuda")
    batch_size = args.batch_size or int(retrieval_cfg.get("embedding_batch_size", 64))

    db_path = kb_dir / "circuit_md.sqlite"
    if not db_path.exists():
        print(f"KB database not found at {db_path}")
        return 1

    retriever = DenseRetriever(
        model_name=model_name,
        index_dir=kb_dir,
        device=device,
        batch_size=batch_size,
    )
    print(f"Loading model {model_name} on {device}...")
    retriever.load_model()
    print(f"Building dense index from {db_path}...")
    meta = retriever.build_index(db_path, progress_every=args.progress_every)
    if "error" in meta:
        print(f"Error: {meta['error']}")
        return 1
    print(f"Built dense index at {kb_dir}")
    print(f"  model: {meta.get('model_name')}")
    print(f"  chunks: {meta.get('chunk_count')}")
    print(f"  dim: {meta.get('embedding_dim')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
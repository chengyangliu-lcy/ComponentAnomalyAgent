from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download evaluator models into the local models directory.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--bertscore-model", default="bert-base-chinese")
    parser.add_argument("--sentence-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = (ROOT / args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    bert_dir = models_dir / args.bertscore_model.replace("/", "__")
    sentence_dir = models_dir / args.sentence_model.replace("/", "__")

    print(f"[models] downloading BERTScore model to {bert_dir}")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.bertscore_model)
    model = AutoModel.from_pretrained(args.bertscore_model)
    tokenizer.save_pretrained(bert_dir)
    model.save_pretrained(bert_dir)

    print(f"[models] downloading sentence-transformers model to {sentence_dir}")
    from sentence_transformers import SentenceTransformer

    sentence_model = SentenceTransformer(args.sentence_model)
    sentence_model.save(str(sentence_dir))

    print("[models] done")
    print(f"BERTSCORE_MODEL_PATH={bert_dir}")
    print(f"SENTENCE_MODEL_PATH={sentence_dir}")


if __name__ == "__main__":
    main()


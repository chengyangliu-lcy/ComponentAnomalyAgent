from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


EMBEDDINGS_FILENAME = "embeddings.npy"
CHUNK_IDS_FILENAME = "chunk_ids.json"
DENSE_INDEX_META_FILENAME = "dense_index_meta.json"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NORMALIZE = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class DenseIndexStatus:
    index_dir: str
    has_embeddings: bool
    has_chunk_ids: bool
    chunk_count: int
    embedding_dim: int
    model_name: str
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.has_embeddings and self.has_chunk_ids and self.chunk_count > 0 and not self.error

    def to_json(self) -> dict[str, Any]:
        return {
            "index_dir": self.index_dir,
            "has_embeddings": self.has_embeddings,
            "has_chunk_ids": self.has_chunk_ids,
            "chunk_count": self.chunk_count,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "usable": self.usable,
            "error": self.error,
        }


def _find_local_model_dir(model_name: str) -> Path | None:
    """Find the local model directory for a given model name.

    Checks both flat (owner__model) and hierarchical (owner/model) formats
    under the project models/ directory.
    """
    flat = MODELS_DIR / model_name.replace("/", "__")
    if flat.exists() and (flat / "config.json").exists():
        return flat
    hierarchical = MODELS_DIR / model_name
    if hierarchical.exists() and (hierarchical / "config.json").exists():
        return hierarchical
    # Also check modelscope cache pattern
    for subdir in MODELS_DIR.iterdir():
        if not subdir.is_dir():
            continue
        # modelscope stores as: models/owner/model-name/hash/files
        if model_name.split("/")[-1].lower() in subdir.name.lower():
            for inner in subdir.iterdir():
                if inner.is_dir() and (inner / "config.json").exists():
                    return inner
    return None


class DenseRetriever:
    """Dense vector retriever using sentence-transformers embedding.

    Builds and queries a dense index alongside the existing SQLite FTS5
    sparse index, enabling hybrid (dense + sparse) retrieval.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-4B",
        index_dir: Path | str = "",
        device: str = "cuda",
        batch_size: int = DEFAULT_BATCH_SIZE,
        normalize: bool = DEFAULT_NORMALIZE,
    ) -> None:
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = None
        self._embeddings: np.ndarray | None = None
        self._chunk_ids: list[int] | None = None
        self._encode_lock = threading.Lock()

    def status(self) -> DenseIndexStatus:
        embeddings_path = self.index_dir / EMBEDDINGS_FILENAME
        chunk_ids_path = self.index_dir / CHUNK_IDS_FILENAME
        meta_path = self.index_dir / DENSE_INDEX_META_FILENAME
        has_embeddings = embeddings_path.exists()
        has_chunk_ids = chunk_ids_path.exists()
        chunk_count = 0
        embedding_dim = 0
        model_name = self.model_name
        error = None
        if has_embeddings:
            try:
                arr = np.load(str(embeddings_path))
                chunk_count = arr.shape[0]
                embedding_dim = arr.shape[1]
            except Exception as exc:
                error = f"embeddings load error: {exc}"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                model_name = meta.get("model_name", model_name)
            except Exception:
                pass
        if has_chunk_ids:
            try:
                ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))
                if not has_embeddings:
                    chunk_count = len(ids)
            except Exception as exc:
                error = f"chunk_ids load error: {exc}"
        return DenseIndexStatus(
            index_dir=str(self.index_dir),
            has_embeddings=has_embeddings,
            has_chunk_ids=has_chunk_ids,
            chunk_count=chunk_count,
            embedding_dim=embedding_dim,
            model_name=model_name,
            error=error,
        )

    MAX_ENCODE_LENGTH = 2048  # truncate inputs to avoid OOM on large models

    def load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        local_dir = _find_local_model_dir(self.model_name)
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        kwargs["model_kwargs"] = {"torch_dtype": "float16"}
        if local_dir is not None:
            kwargs["local_files_only"] = True
            self._model = SentenceTransformer(str(local_dir), **kwargs)
        else:
            self._model = SentenceTransformer(self.model_name, **kwargs)
        self._model.max_seq_length = self.MAX_ENCODE_LENGTH

    def build_index(self, db_path: Path | str, *, progress_every: int = 0) -> dict[str, Any]:
        """Build dense embeddings from all chunks in SQLite KB."""
        if self._model is None:
            self.load_model()
        db = Path(db_path)
        if not db.exists():
            return {"error": f"missing {db}", "chunk_count": 0}
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT chunk_id, text FROM chunks ORDER BY chunk_id").fetchall())
        conn.close()
        if not rows:
            return {"error": "no chunks in KB", "chunk_count": 0}
        chunk_ids = [int(row["chunk_id"]) for row in rows]
        texts = [str(row["text"] or "") for row in rows]
        # For models like Qwen3-Embedding, document encoding uses no prompt
        embeddings = self._encode_batch(texts, prompt_name="document", progress_every=progress_every)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self.index_dir / EMBEDDINGS_FILENAME), embeddings)
        (self.index_dir / CHUNK_IDS_FILENAME).write_text(json.dumps(chunk_ids), encoding="utf-8")
        meta = {
            "model_name": self.model_name,
            "embedding_dim": embeddings.shape[1],
            "chunk_count": len(chunk_ids),
            "normalize": self.normalize,
            "device": self.device,
            "batch_size": self.batch_size,
        }
        (self.index_dir / DENSE_INDEX_META_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._embeddings = embeddings
        self._chunk_ids = chunk_ids
        return meta

    def load_index(self) -> None:
        """Load pre-built embeddings index."""
        embeddings_path = self.index_dir / EMBEDDINGS_FILENAME
        chunk_ids_path = self.index_dir / CHUNK_IDS_FILENAME
        if not embeddings_path.exists() or not chunk_ids_path.exists():
            return
        self._embeddings = np.load(str(embeddings_path))
        self._chunk_ids = json.loads(chunk_ids_path.read_text(encoding="utf-8"))

    def search(self, query: str, limit: int = 4) -> list[tuple[float, int]]:
        """Return (cosine_similarity, chunk_id) pairs for query.

        Returns empty list if index is not loaded or model is not available.
        Thread-safe: encode is serialized via _encode_lock so shared
        retrievers can be used across multiple worker threads.
        """
        if self._embeddings is None or self._chunk_ids is None:
            return []
        if self._model is None:
            try:
                self.load_model()
            except OSError:
                return []
        # For models like Qwen3-Embedding, query encoding uses prompt_name="query"
        # which prepends the instruction prefix from config_sentence_transformers.json
        with self._encode_lock:
            query_embedding = self._model.encode(
                [query], prompt_name="query", normalize_embeddings=self.normalize,
            )
        query_vec = query_embedding[0]
        scores = np.dot(self._embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:limit]
        return [(float(scores[idx]), self._chunk_ids[idx]) for idx in top_indices]

    def _encode_batch(self, texts: list[str], *, prompt_name: str | None = None, progress_every: int = 0) -> np.ndarray:
        """Encode texts in batches, optionally with progress output."""
        total = len(texts)
        if progress_every > 0:
            print(f"encoding {total} chunks (batch_size={self.batch_size})...", flush=True)
        embeddings = self._model.encode(
            texts,
            prompt_name=prompt_name,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(embeddings)
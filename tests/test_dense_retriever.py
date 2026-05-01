from __future__ import annotations

import json
import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from tools.dense_retriever import DenseRetriever, DenseIndexStatus, EMBEDDINGS_FILENAME, CHUNK_IDS_FILENAME


def _make_test_kb(dir_path: Path, n_chunks: int = 3) -> Path:
    """Create a minimal SQLite KB with n_chunks for testing."""
    db_path = dir_path / "circuit_md.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (chunk_id INTEGER PRIMARY KEY, page_id INTEGER, title TEXT, url TEXT, text TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS pages (page_id INTEGER PRIMARY KEY, path TEXT, title TEXT, url TEXT, published_at TEXT, text_hash TEXT)")
    for i in range(n_chunks):
        conn.execute("INSERT INTO pages (page_id, path, title, url, published_at, text_hash) VALUES (?, ?, ?, ?, ?, ?)",
                      (i, f"page_{i}.md", f"Title {i}", f"https://example.com/{i}", "2024", f"hash{i}"))
        conn.execute("INSERT INTO chunks (chunk_id, page_id, title, url, text) VALUES (?, ?, ?, ?, ?)",
                      (i, i, f"Title {i}", f"https://example.com/{i}", f"Test chunk text about BMS battery management system number {i}."))
    conn.commit()
    conn.close()
    return db_path


class DenseRetrieverTests(unittest.TestCase):

    def test_status_returns_empty_when_no_index(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        retriever = DenseRetriever(model_name="test-model", index_dir=tmp)
        status = retriever.status()

        self.assertFalse(status.has_embeddings)
        self.assertFalse(status.has_chunk_ids)
        self.assertEqual(status.chunk_count, 0)
        self.assertFalse(status.usable)

    def test_status_returns_usable_when_index_exists(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        embeddings = np.random.randn(5, 8).astype(np.float32)
        np.save(str(tmp / EMBEDDINGS_FILENAME), embeddings)
        (tmp / CHUNK_IDS_FILENAME).write_text(json.dumps([1, 2, 3, 4, 5]), encoding="utf-8")

        retriever = DenseRetriever(model_name="test-model", index_dir=tmp)
        status = retriever.status()

        self.assertTrue(status.has_embeddings)
        self.assertTrue(status.has_chunk_ids)
        self.assertEqual(status.chunk_count, 5)
        self.assertEqual(status.embedding_dim, 8)
        self.assertTrue(status.usable)

    def test_load_index_reads_embeddings_and_chunk_ids(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        embeddings = np.random.randn(3, 4).astype(np.float32)
        np.save(str(tmp / EMBEDDINGS_FILENAME), embeddings)
        (tmp / CHUNK_IDS_FILENAME).write_text(json.dumps([10, 20, 30]), encoding="utf-8")

        retriever = DenseRetriever(model_name="test-model", index_dir=tmp)
        retriever.load_index()

        np.testing.assert_array_equal(retriever._embeddings, embeddings)
        self.assertEqual(retriever._chunk_ids, [10, 20, 30])

    def test_load_index_is_noop_when_files_missing(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        retriever = DenseRetriever(model_name="test-model", index_dir=tmp)
        retriever.load_index()

        self.assertIsNone(retriever._embeddings)
        self.assertIsNone(retriever._chunk_ids)

    def test_search_returns_empty_when_index_not_loaded(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        retriever = DenseRetriever(model_name="test-model", index_dir=tmp)

        results = retriever.search("BMS battery", limit=3)

        self.assertEqual(results, [])

    def test_search_returns_empty_when_no_model_available(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        embeddings = np.random.randn(3, 4).astype(np.float32)
        np.save(str(tmp / EMBEDDINGS_FILENAME), embeddings)
        (tmp / CHUNK_IDS_FILENAME).write_text(json.dumps([1, 2, 3]), encoding="utf-8")

        retriever = DenseRetriever(model_name="mock-model", index_dir=tmp)
        retriever.load_index()
        # Simulate a model that raises on load (e.g. missing HF repo)
        retriever.load_model = lambda: (_ for _ in ()).throw(OSError("no model"))
        results = retriever.search("query", limit=2)

        self.assertEqual(results, [])

    def test_build_index_saves_files_and_returns_meta(self) -> None:
        """Build index without actual model by mocking _encode_batch."""
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)
        db_path = _make_test_kb(tmp, n_chunks=3)

        retriever = DenseRetriever(model_name="mock-model", index_dir=tmp)
        # Mock the encoding step
        fake_embeddings = np.random.randn(3, 8).astype(np.float32)
        retriever._encode_batch = lambda texts, **kw: fake_embeddings
        retriever._model = True  # bypass load_model check

        meta = retriever.build_index(db_path)

        self.assertEqual(meta["model_name"], "mock-model")
        self.assertEqual(meta["chunk_count"], 3)
        self.assertEqual(meta["embedding_dim"], 8)
        self.assertTrue((tmp / EMBEDDINGS_FILENAME).exists())
        self.assertTrue((tmp / CHUNK_IDS_FILENAME).exists())
        saved_ids = json.loads((tmp / CHUNK_IDS_FILENAME).read_text(encoding="utf-8"))
        self.assertEqual(saved_ids, [0, 1, 2])

    def test_build_index_returns_error_for_missing_db(self) -> None:
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)

        retriever = DenseRetriever(model_name="mock-model", index_dir=tmp)
        retriever._model = True

        meta = retriever.build_index(tmp / "nonexistent.sqlite")

        self.assertIn("error", meta)

    def test_search_returns_top_k_results_with_mock_data(self) -> None:
        """Test search with pre-built mock embeddings and a mock model."""
        tmp = Path("outputs") / "test_dense_retriever" / f"{uuid4().hex}"
        tmp.mkdir(parents=True, exist_ok=True)

        # Create embeddings where chunk 1 is most similar to query "BMS"
        # Use 4-dim vectors for simplicity
        embeddings = np.array([
            [0.1, 0.1, 0.1, 0.1],  # chunk 0: generic
            [0.9, 0.8, 0.7, 0.6],  # chunk 1: "BMS" similar
            [0.2, 0.2, 0.2, 0.2],  # chunk 2: generic
        ], dtype=np.float32)
        np.save(str(tmp / EMBEDDINGS_FILENAME), embeddings)
        (tmp / CHUNK_IDS_FILENAME).write_text(json.dumps([0, 1, 2]), encoding="utf-8")

        retriever = DenseRetriever(model_name="mock-model", index_dir=tmp)
        retriever.load_index()

        # Mock the model to return a "BMS-like" query vector
        class MockModel:
            def encode(self, texts, prompt_name=None, normalize_embeddings=False, show_progress_bar=False, **kwargs):
                return np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)

        retriever._model = MockModel()

        results = retriever.search("BMS battery", limit=2)

        self.assertEqual(len(results), 2)
        # Chunk 1 should be the top result (closest to query vector)
        self.assertEqual(results[0][1], 1)
        self.assertGreater(results[0][0], results[1][0])


class DenseIndexStatusTests(unittest.TestCase):

    def test_usable_requires_all_conditions(self) -> None:
        status = DenseIndexStatus(
            index_dir="/tmp",
            has_embeddings=True,
            has_chunk_ids=True,
            chunk_count=5,
            embedding_dim=8,
            model_name="test",
            error=None,
        )
        self.assertTrue(status.usable)

    def test_not_usable_when_error(self) -> None:
        status = DenseIndexStatus(
            index_dir="/tmp",
            has_embeddings=True,
            has_chunk_ids=True,
            chunk_count=5,
            embedding_dim=8,
            model_name="test",
            error="load failed",
        )
        self.assertFalse(status.usable)

    def test_not_usable_when_missing_embeddings(self) -> None:
        status = DenseIndexStatus(
            index_dir="/tmp",
            has_embeddings=False,
            has_chunk_ids=True,
            chunk_count=5,
            embedding_dim=0,
            model_name="test",
            error=None,
        )
        self.assertFalse(status.usable)


if __name__ == "__main__":
    unittest.main()
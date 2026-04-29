from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from uuid import uuid4
import unittest
from unittest.mock import patch

from tools.evidence_tools import LocalRetrieveExecutor
from tools.hackster_kb import (
    HacksterHybridRetriever,
    build_hackster_kb,
    canonical_url,
    chunk_text,
    extract_warc_payload,
    html_to_text,
    warc_local_path,
)


class HacksterKbTests(unittest.TestCase):
    def test_html_to_text_removes_noise_and_keeps_project_content(self) -> None:
        title, text = html_to_text(
            b"""
            <html><head><title>Arduino Motor Driver</title><script>bad()</script></head>
            <body><nav>menu</nav><main><h1>Motor Driver</h1>
            <p>This project explains current sense filtering and MOSFET gate drive.</p>
            <pre>analogWrite(pin, value);</pre></main></body></html>
            """
        )

        self.assertEqual(title, "Arduino Motor Driver")
        self.assertIn("current sense filtering", text)
        self.assertIn("analogWrite", text)
        self.assertNotIn("bad()", text)
        self.assertNotIn("menu", text)

    def test_extract_warc_payload_from_compressed_slice(self) -> None:
        with _workspace_tempdir() as root:
            warc = Path(root) / "sample.warc.gz"
            html = b"<html><body><p>Hackster payload text</p></body></html>"
            record = (
                b"WARC/1.0\r\nWARC-Type: response\r\n\r\n"
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"
                + html
            )
            compressed = gzip.compress(record)
            warc.write_bytes(compressed)

            payload = extract_warc_payload(warc, 0, len(compressed))

        self.assertEqual(payload, html)

    def test_build_hackster_kb_writes_documents_chunks_and_meta(self) -> None:
        with _workspace_tempdir() as root:
            root_path = Path(root)
            warc_root = root_path / "warc"
            warc_file = warc_root / "crawl-data" / "sample.warc.gz"
            warc_file.parent.mkdir(parents=True)
            body = (
                "<html><head><title>Current Sense Project</title></head><body><main>"
                "<h1>Current Sense Project</h1>"
                "<p>This Hackster project covers current sense filtering, MOSFET drive, "
                "Arduino firmware, and troubleshooting noisy measurements.</p>"
                "</main></body></html>"
            ).encode("utf-8")
            record = b"WARC/1.0\r\n\r\nHTTP/1.1 200 OK\r\n\r\n" + body
            compressed = gzip.compress(record)
            warc_file.write_bytes(compressed)
            csv_path = root_path / "index.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "url",
                        "fetch_status",
                        "content_mime_type",
                        "content_digest",
                        "fetch_time",
                        "url_path",
                        "url_query",
                        "warc_filename",
                        "warc_record_offset",
                        "warc_record_length",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "url": "https://www.hackster.io/user/current-sense/",
                        "fetch_status": "200",
                        "content_mime_type": "text/html",
                        "content_digest": "digest-a",
                        "fetch_time": "2016-08-30 00:00:00.000",
                        "url_path": "/user/current-sense/",
                        "url_query": "",
                        "warc_filename": "crawl-data/sample.warc.gz",
                        "warc_record_offset": "0",
                        "warc_record_length": str(len(compressed)),
                    }
                )
                writer.writerow(
                    {
                        "url": "https://www.hackster.io/user/ignored",
                        "fetch_status": "404",
                        "content_mime_type": "text/html",
                        "content_digest": "digest-b",
                        "fetch_time": "2016-08-30 00:00:00.000",
                        "url_path": "/user/ignored",
                        "url_query": "",
                        "warc_filename": "crawl-data/sample.warc.gz",
                        "warc_record_offset": "0",
                        "warc_record_length": str(len(compressed)),
                    }
                )

            with patch("tools.hackster_kb.write_embeddings", return_value={"status": "mocked", "count": 1}):
                meta = build_hackster_kb(csv_path, warc_root, root_path / "kb")

            chunks = _read_jsonl(root_path / "kb" / "chunks.jsonl")
            docs = _read_jsonl(root_path / "kb" / "documents.jsonl")
            stored_meta = json.loads((root_path / "kb" / "index_meta.json").read_text(encoding="utf-8"))

        self.assertEqual(meta["documents"], 1)
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(stored_meta["embedding"]["status"], "mocked")
        self.assertEqual(chunks[0]["metadata"]["warc_filename"], "crawl-data/sample.warc.gz")

    def test_hybrid_retriever_keyword_fallback_and_executor(self) -> None:
        with _workspace_tempdir() as root:
            kb = Path(root) / "kb"
            kb.mkdir()
            rows = [
                {
                    "chunk_id": "doc1:0000",
                    "doc_id": "doc1",
                    "url": "https://www.hackster.io/project/current-sense",
                    "title": "Current Sense Filter",
                    "text": "Current sense filtering reduces leading edge noise in MOSFET motor drivers.",
                    "metadata": {"fetch_time": "2016", "content_digest": "a", "warc_filename": "x"},
                },
                {
                    "chunk_id": "doc2:0000",
                    "doc_id": "doc2",
                    "url": "https://www.hackster.io/project/led",
                    "title": "LED Blink",
                    "text": "Blink an LED with Arduino delay.",
                    "metadata": {"fetch_time": "2016", "content_digest": "b", "warc_filename": "y"},
                },
            ]
            with (kb / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            retriever = HacksterHybridRetriever(kb, dense_weight=0.0, keyword_weight=1.0)
            executor = LocalRetrieveExecutor(retriever)
            run = executor.run("MOSFET current sense noise", limit=1)

        self.assertTrue(run.success)
        self.assertEqual(run.evidence[0].metadata["kind"], "local_kb_chunk")
        self.assertEqual(run.evidence[0].metadata["chunk_id"], "doc1:0000")
        self.assertIn("current sense", run.evidence[0].content.lower())

    def test_small_helpers(self) -> None:
        self.assertEqual(canonical_url("HTTPS://WWW.HACKSTER.IO/a/b/"), "https://www.hackster.io/a/b")
        self.assertTrue(str(warc_local_path(Path("root"), "a/b/c.warc.gz")).endswith(str(Path("a") / "b" / "c.warc.gz")))
        with _workspace_tempdir() as root:
            single_warc = Path(root) / "combined.warc"
            single_warc.write_bytes(b"")
            self.assertEqual(warc_local_path(single_warc, "a/b/c.warc.gz"), single_warc)
        self.assertGreaterEqual(len(chunk_text("a" * 2200, chunk_chars=1000, overlap=100)), 3)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class _workspace_tempdir:
    def __enter__(self) -> str:
        base = Path(__file__).resolve().parents[1] / "outputs" / "test_hackster_kb"
        base.mkdir(parents=True, exist_ok=True)
        self.path = base / uuid4().hex
        self.path.mkdir(parents=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit, urlunsplit

from bs4 import BeautifulSoup

from schemas import Evidence
from tools.retriever import tokenize
from tools.utils import compact_text


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHUNK_CHARS = 1000
DEFAULT_CHUNK_OVERLAP = 150


def _progress(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        return iterable
    return tqdm(iterable, **kwargs)


@dataclass(frozen=True)
class HacksterDocument:
    doc_id: str
    url: str
    title: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class HacksterChunk:
    chunk_id: str
    doc_id: str
    url: str
    title: str
    text: str
    metadata: dict[str, Any]


def canonical_url(url: str) -> str:
    parsed = urlsplit((url or "").strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


def warc_local_path(warc_root: Path, warc_filename: str) -> Path:
    if warc_root.is_file():
        return warc_root
    return warc_root / Path(str(warc_filename).replace("/", os.sep))


def iter_candidate_rows(csv_path: Path, limit: int | None = None) -> Iterator[dict[str, str]]:
    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("fetch_status") != "200":
                continue
            if (row.get("content_mime_type") or "").lower() != "text/html":
                continue
            yield row
            count += 1
            if limit is not None and count >= limit:
                break


def dedupe_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for row in rows:
        key = row.get("content_digest") or canonical_url(row.get("url") or "")
        if not key:
            continue
        current = best.get(key)
        if current is None or _row_rank(row) > _row_rank(current):
            best[key] = row
    return list(best.values())


def _row_rank(row: dict[str, str]) -> tuple[str, int, int]:
    fetch_time = row.get("fetch_time") or ""
    url = row.get("url") or ""
    https_bonus = 1 if url.startswith("https://") else 0
    query_penalty = 0 if not row.get("url_query") else -1
    return fetch_time, https_bonus, query_penalty


def extract_warc_payload(warc_path: Path, offset: int, length: int) -> bytes:
    with warc_path.open("rb") as f:
        f.seek(offset)
        compressed = f.read(length)
    try:
        record = gzip.decompress(compressed)
    except OSError:
        record = compressed
    return _extract_http_body(record)


def _extract_warc_payload_from_stream(stream: gzip.GzipFile, offset: int, length: int) -> bytes:
    if stream.tell() > offset:
        stream.seek(offset)
    elif stream.tell() < offset:
        stream.read(offset - stream.tell())
    return _extract_http_body(stream.read(length))


def _extract_http_body(record: bytes) -> bytes:
    separator = b"\r\n\r\n"
    payload = record
    if payload.startswith(b"WARC/"):
        head_end = payload.find(separator)
        if head_end < 0:
            return payload
        payload = payload[head_end + len(separator) :]
    if payload.startswith(b"HTTP/"):
        http_end = payload.find(separator)
        if http_end < 0:
            return payload
        headers = payload[:http_end].decode("iso-8859-1", errors="replace")
        body = payload[http_end + len(separator) :]
        header_map = _parse_http_headers(headers)
        encoding = header_map.get("content-encoding", "").lower()
        if "gzip" in encoding:
            try:
                body = gzip.decompress(body)
            except OSError:
                pass
        payload = body
    return _clean_payload_boundaries(payload)


def _clean_payload_boundaries(payload: bytes) -> bytes:
    next_warc = payload.find(b"\r\nWARC/1.0")
    if next_warc >= 0:
        payload = payload[:next_warc]
    leading_http = payload.find(b"HTTP/1.1 ")
    if 0 <= leading_http < 4096:
        http_end = payload.find(b"\r\n\r\n", leading_http)
        if http_end >= 0:
            payload = payload[http_end + 4 :]
    return payload.strip()


def _parse_http_headers(headers: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for line in headers.splitlines()[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _is_gzip_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _offset_points_to_gzip_member(path: Path, offset: int) -> bool:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            return f.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def html_to_text(html: bytes | str, max_chars: int = 20000) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "form"]):
        tag.decompose()
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())
    main = soup.find("main") or soup.find("article") or soup.body or soup
    parts: list[str] = []
    for element in main.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"], recursive=True):
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) >= 3:
            parts.append(text)
    if not parts:
        parts = [" ".join(main.get_text(" ", strip=True).split())]
    text = "\n".join(_dedupe_nearby(parts))
    return title, compact_text(text, max_chars)


def _dedupe_nearby(parts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        kept.append(part)
    return kept


def chunk_text(text: str, chunk_chars: int = DEFAULT_CHUNK_CHARS, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    normalized = "\n".join(line.strip() for line in (text or "").splitlines() if line.strip())
    if not normalized:
        return []
    if len(normalized) <= chunk_chars:
        return [normalized]
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + chunk_chars)
        if end < len(normalized):
            boundary = max(normalized.rfind("\n", start, end), normalized.rfind(". ", start, end), normalized.rfind("。", start, end))
            if boundary > start + chunk_chars // 2:
                end = boundary + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_hackster_kb(
    csv_path: Path,
    warc_root: Path,
    output_dir: Path,
    *,
    limit: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_docs: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = dedupe_rows(iter_candidate_rows(csv_path, limit=limit))
    if max_docs is not None:
        rows = rows[:max_docs]
    use_decompressed_stream = False
    if warc_root.is_file() and rows:
        try:
            first_offset = int(rows[0].get("warc_record_offset") or "0")
        except ValueError:
            first_offset = 0
        use_decompressed_stream = _is_gzip_file(warc_root) and not _offset_points_to_gzip_member(warc_root, first_offset)
    if use_decompressed_stream:
        rows = sorted(rows, key=lambda item: int(item.get("warc_record_offset") or "0"))

    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    stream: gzip.GzipFile | None = gzip.open(warc_root, "rb") if use_decompressed_stream else None
    progress = _progress(rows, desc="extract Hackster WARC", unit="record")
    for row in progress:
        stats["candidate_rows"] += 1
        try:
            offset = int(row.get("warc_record_offset") or "0")
            length = int(row.get("warc_record_length") or "0")
        except ValueError:
            stats["bad_warc_offsets"] += 1
            continue
        warc_path = warc_local_path(warc_root, row.get("warc_filename") or "")
        if not warc_path.exists():
            stats["missing_warc_files"] += 1
            continue
        try:
            if stream is not None and warc_path == warc_root:
                payload = _extract_warc_payload_from_stream(stream, offset, length)
            else:
                payload = extract_warc_payload(warc_path, offset, length)
            title, text = html_to_text(payload)
        except Exception:
            stats["extract_errors"] += 1
            continue
        if len(text) < 120:
            stats["short_documents"] += 1
            continue
        url = canonical_url(row.get("url") or "")
        doc_id = hashlib.sha1((row.get("content_digest") or url).encode("utf-8")).hexdigest()[:16]
        metadata = {
            "url_path": row.get("url_path") or "",
            "fetch_time": row.get("fetch_time") or "",
            "content_digest": row.get("content_digest") or "",
            "warc_filename": row.get("warc_filename") or "",
            "warc_record_offset": offset,
            "warc_record_length": length,
        }
        doc = {"doc_id": doc_id, "url": url, "title": title or url, "text": text, "metadata": metadata}
        doc_chunks = [chunk for chunk in chunk_text(text, chunk_chars=chunk_chars, overlap=chunk_overlap) if _is_useful_chunk(chunk)]
        if not doc_chunks:
            stats["no_useful_chunks"] += 1
            continue
        documents.append(doc)
        for index, chunk in enumerate(doc_chunks):
            chunk_id = f"{doc_id}:{index:04d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "url": url,
                    "title": title or url,
                    "text": chunk,
                    "metadata": {**metadata, "chunk_index": index},
                }
            )
        stats["documents"] += 1
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                documents=len(documents),
                chunks=len(chunks),
                short=stats.get("short_documents", 0),
                skipped=stats.get("no_useful_chunks", 0),
            )
    if stream is not None:
        stream.close()

    _write_jsonl(output_dir / "documents.jsonl", documents)
    _write_jsonl(output_dir / "chunks.jsonl", chunks)
    embedding_info = write_embeddings(output_dir / "embeddings.npy", [item["text"] for item in chunks], embedding_model)
    meta = {
        "source_csv": str(csv_path),
        "warc_root": str(warc_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filters": {"fetch_status": "200", "content_mime_type": "text/html"},
        "embedding_model": embedding_model,
        "chunk_chars": chunk_chars,
        "chunk_overlap": chunk_overlap,
        "documents": len(documents),
        "chunks": len(chunks),
        "stats": dict(stats),
        "embedding": embedding_info,
    }
    (output_dir / "index_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def write_embeddings(path: Path, texts: list[str], model_name: str) -> dict[str, Any]:
    if not texts:
        return {"status": "empty", "count": 0}
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "count": 0, "error": str(exc)}
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    array = np.asarray(embeddings, dtype="float32")
    np.save(path, array)
    return {"status": "written", "count": int(array.shape[0]), "dimensions": int(array.shape[1])}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_useful_chunk(text: str) -> bool:
    normalized = " ".join((text or "").split())
    if len(normalized) < 120:
        return False
    lowered = normalized.lower()
    noisy_markers = [
        "warc/1.0",
        "http/1.1",
        "set-cookie:",
        "content-type:",
        "redirect_to=",
        "users/auth/",
        "recaptchasitekey",
        "reactportal",
        "hckui__buttons__",
    ]
    if any(marker in lowered for marker in noisy_markers):
        return False
    alnum = sum(char.isalnum() for char in normalized)
    if alnum / max(len(normalized), 1) < 0.45:
        return False
    return len(normalized.split()) >= 10


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class HacksterHybridRetriever:
    def __init__(
        self,
        index_dir: Path | str,
        *,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        dense_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.dense_weight = dense_weight
        self.keyword_weight = keyword_weight
        self._chunks: list[dict[str, Any]] | None = None
        self._tokenized: list[list[str]] | None = None
        self._idf: dict[str, float] | None = None
        self._embeddings: Any | None = None
        self._model: Any | None = None

    @property
    def chunks(self) -> list[dict[str, Any]]:
        if self._chunks is None:
            self._chunks = _read_jsonl(self.index_dir / "chunks.jsonl")
        return self._chunks

    def search(self, query: str, limit: int = 4) -> list[Evidence]:
        query = (query or "").strip()
        if not query or not self.chunks:
            return []
        keyword_scores = self._keyword_scores(query)
        dense_scores = self._dense_scores(query)
        combined: list[tuple[float, dict[str, Any]]] = []
        for idx, chunk in enumerate(self.chunks):
            score = self.keyword_weight * keyword_scores[idx] + self.dense_weight * dense_scores[idx]
            title_url_text = f"{chunk.get('title','')} {chunk.get('url','')}".lower()
            for token in tokenize(query):
                if token in title_url_text:
                    score += 0.15
            if score > 0:
                combined.append((score, chunk))
        combined.sort(key=lambda item: item[0], reverse=True)
        return [self._to_evidence(chunk, score) for score, chunk in combined[:limit]]

    def _keyword_scores(self, query: str) -> list[float]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return [0.0] * len(self.chunks)
        tokenized = self._tokenized_chunks()
        idf = self._idf_scores(tokenized)
        raw_scores: list[float] = []
        query_set = set(query_tokens)
        for tokens in tokenized:
            counts = Counter(tokens)
            length_norm = math.sqrt(max(len(tokens), 1))
            score = sum((counts[token] / length_norm) * idf.get(token, 0.0) for token in query_set)
            raw_scores.append(score)
        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score <= 0:
            return [0.0] * len(raw_scores)
        return [score / max_score for score in raw_scores]

    def _dense_scores(self, query: str) -> list[float]:
        embeddings = self._load_embeddings()
        if embeddings is None:
            return [0.0] * len(self.chunks)
        model = self._load_model()
        if model is None:
            return [0.0] * len(self.chunks)
        try:
            import numpy as np

            query_embedding = model.encode([query], normalize_embeddings=True)
            scores = np.asarray(embeddings @ np.asarray(query_embedding[0], dtype="float32"), dtype="float32")
            scores = (scores + 1.0) / 2.0
            return [float(score) for score in scores]
        except Exception:
            return [0.0] * len(self.chunks)

    def _tokenized_chunks(self) -> list[list[str]]:
        if self._tokenized is None:
            self._tokenized = [tokenize(f"{chunk.get('title','')} {chunk.get('text','')} {chunk.get('url','')}") for chunk in self.chunks]
        return self._tokenized

    def _idf_scores(self, tokenized: list[list[str]]) -> dict[str, float]:
        if self._idf is not None:
            return self._idf
        doc_count = len(tokenized)
        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))
        self._idf = {
            token: math.log((doc_count + 1) / (frequency + 0.5)) + 1.0
            for token, frequency in document_frequency.items()
        }
        return self._idf

    def _load_embeddings(self) -> Any | None:
        if self._embeddings is not None:
            return self._embeddings
        path = self.index_dir / "embeddings.npy"
        if not path.exists():
            return None
        try:
            import numpy as np

            embeddings = np.load(path)
            if len(embeddings) != len(self.chunks):
                return None
            self._embeddings = embeddings
            return self._embeddings
        except Exception:
            return None

    def _load_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model)
            return self._model
        except Exception:
            return None

    def _to_evidence(self, chunk: dict[str, Any], score: float) -> Evidence:
        metadata = dict(chunk.get("metadata") or {})
        metadata.update({"kind": "local_kb_chunk", "chunk_id": chunk.get("chunk_id"), "doc_id": chunk.get("doc_id")})
        return Evidence(
            source=str(chunk.get("url") or ""),
            title=str(chunk.get("title") or chunk.get("url") or "Hackster KB"),
            content=compact_text(str(chunk.get("text") or ""), 4000),
            score=round(float(score), 4),
            metadata=metadata,
        )

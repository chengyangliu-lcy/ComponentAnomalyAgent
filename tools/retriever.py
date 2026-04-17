from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from schemas import Evidence
from tools.utils import compact_text


TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_.+-]+")


def tokenize(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "") if token.strip()]


@dataclass
class LocalDocument:
    path: Path
    title: str
    text: str
    tokens: List[str]


class LocalRetriever:
    def __init__(self, corpus_root: Path, max_doc_chars: int = 6000) -> None:
        self.corpus_root = corpus_root
        self.max_doc_chars = max_doc_chars
        self._docs: List[LocalDocument] | None = None

    def _iter_paths(self) -> Iterable[Path]:
        if not self.corpus_root.exists():
            return []
        return self.corpus_root.glob("*/*/*.md")

    def _load_docs(self) -> List[LocalDocument]:
        docs: List[LocalDocument] = []
        for path in self._iter_paths():
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            title = path.stem
            docs.append(LocalDocument(path=path, title=title, text=compact_text(text, self.max_doc_chars), tokens=tokenize(text[:12000])))
        return docs

    @property
    def docs(self) -> List[LocalDocument]:
        if self._docs is None:
            self._docs = self._load_docs()
        return self._docs

    def search(self, query: str, limit: int = 4, post_id: str | None = None) -> List[Evidence]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_set = set(query_tokens)
        scored: List[tuple[float, LocalDocument]] = []
        for doc in self.docs:
            token_set = set(doc.tokens)
            overlap = len(query_set & token_set)
            if post_id and post_id in str(doc.path):
                overlap += 8
            if overlap <= 0:
                continue
            coverage = overlap / max(len(query_set), 1)
            density = overlap / math.sqrt(max(len(token_set), 1))
            scored.append((coverage + density, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Evidence(
                source=str(doc.path),
                title=doc.title,
                content=doc.text,
                score=round(score, 4),
                metadata={"kind": "local_markdown"},
            )
            for score, doc in scored[:limit]
        ]


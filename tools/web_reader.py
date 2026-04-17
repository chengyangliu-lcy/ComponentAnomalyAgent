from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

from schemas import Evidence
from tools.utils import compact_text


@dataclass
class WebReadResult:
    evidence: Optional[Evidence]
    error: Optional[str] = None


class WebReader:
    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def read(self, url: str, max_chars: int = 5000) -> WebReadResult:
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 ComponentAnomalyAgent/1.0"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else url
            text = soup.get_text(" ", strip=True)
            return WebReadResult(
                evidence=Evidence(
                    source=url,
                    title=title,
                    content=compact_text(text, max_chars=max_chars),
                    metadata={"status_code": response.status_code, "kind": "web_page"},
                )
            )
        except Exception as exc:  # noqa: BLE001
            return WebReadResult(evidence=None, error=str(exc))


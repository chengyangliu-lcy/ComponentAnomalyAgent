from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

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
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            }
        )

    def read(self, url: str, max_chars: int = 5000) -> WebReadResult:
        try:
            if not self._is_public_http_url(url):
                return WebReadResult(evidence=None, error=f"blocked invalid or non-public URL: {url}")
            response = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and "text/plain" not in content_type and content_type:
                return WebReadResult(evidence=None, error=f"unsupported content-type: {content_type}")
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                tag.decompose()
            title = soup.title.get_text(" ", strip=True) if soup.title else response.url
            main = soup.find("article") or soup.find("main") or soup.body or soup
            text = main.get_text(" ", strip=True)
            return WebReadResult(
                evidence=Evidence(
                    source=response.url,
                    title=title,
                    content=compact_text(text, max_chars=max_chars),
                    metadata={"status_code": response.status_code, "kind": "web_page"},
                )
            )
        except Exception as exc:  # noqa: BLE001
            return WebReadResult(evidence=None, error=str(exc))

    def _is_public_http_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        host = parsed.hostname or ""
        blocked = ("localhost", "127.", "10.", "192.168.", "172.16.", "0.0.0.0")
        return not any(host.startswith(prefix) for prefix in blocked)

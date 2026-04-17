from __future__ import annotations

from urllib.parse import quote_plus, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

from schemas import Evidence
from tools.utils import compact_text


class WebSearch:
    """No-key HTML search helper. It records empty/blocked results instead of failing."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def search(self, query: str, limit: int = 5) -> tuple[list[Evidence], str | None]:
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
        try:
            response = requests.get(
                search_url,
                timeout=self.timeout,
                headers={"User-Agent": "Mozilla/5.0 ComponentAnomalyAgent/1.0"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            results: list[Evidence] = []
            for node in soup.select("li.b_algo"):
                link = node.find("a")
                if not link or not link.get("href"):
                    continue
                url = self._clean_url(link["href"])
                title = link.get_text(" ", strip=True)
                snippet_node = node.find("p")
                snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
                results.append(
                    Evidence(
                        source=url,
                        title=title,
                        content=compact_text(snippet, max_chars=1000),
                        metadata={"kind": "web_search_result", "query": query},
                    )
                )
                if len(results) >= limit:
                    break
            return results, None if results else "search returned no parseable results"
        except Exception as exc:  # noqa: BLE001
            return [], str(exc)

    def _clean_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("bing.com") and parsed.path == "/ck/a":
            values = parse_qs(parsed.query).get("u")
            if values:
                return unquote(values[0])
        return url


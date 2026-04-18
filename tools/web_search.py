from __future__ import annotations

import base64
import re
import time
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from schemas import Evidence
from tools.utils import compact_text


class WebSearch:
    """No-key HTML search helper returning structured search results.

    This mirrors the OpenClaw split between search and fetch: search only
    returns normalized candidate URLs and snippets; WebReader fetches pages.
    """

    def __init__(self, timeout: int = 20, provider: str = "duckduckgo", cache_ttl_seconds: int = 900) -> None:
        self.timeout = timeout
        self.provider = provider
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[float, list[Evidence], str | None]] = {}
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            }
        )

    def search(self, query: str, limit: int = 5) -> tuple[list[Evidence], str | None]:
        cache_key = f"{self.provider}:{query}:{limit}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self.cache_ttl_seconds:
            return cached[1], cached[2]
        provider_order = [self.provider, "bing", "yahoo"]
        provider_order = list(dict.fromkeys(provider_order))
        errors: list[str] = []
        results: list[Evidence] = []
        for provider in provider_order:
            provider_results, provider_error = self._try_provider(provider, query, limit)
            if provider_error:
                errors.append(provider_error)
            filtered_results = self._rank_and_filter(query, provider_results, limit) if provider_results else []
            if filtered_results:
                results = filtered_results
                break
            if provider_results:
                errors.append(f"{provider}: no relevant results after filtering")
        try:
            error = None if results else f"search returned no parseable relevant results; provider_errors={'; '.join(errors)}"
            self._cache[cache_key] = (time.time(), results, error)
            return results, error
        except Exception as exc:  # noqa: BLE001
            return [], str(exc)

    def _try_provider(self, provider: str, query: str, limit: int) -> tuple[list[Evidence], str | None]:
        try:
            if provider == "duckduckgo":
                results = self._search_duckduckgo(query, limit)
            elif provider == "bing":
                results = self._search_bing(query, limit)
            elif provider == "yahoo":
                results = self._search_yahoo(query, limit)
            else:
                return [], f"{provider}: unsupported search provider"
            return results, None if results else f"{provider}: no parseable results"
        except Exception as exc:  # noqa: BLE001
            return [], f"{provider}: {exc}"

    def _search_duckduckgo(self, query: str, limit: int) -> list[Evidence]:
        search_url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.session.get(search_url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[Evidence] = []
        for node in soup.select(".result"):
            link = node.select_one(".result__a")
            if not link:
                continue
            url = self._clean_url(link.get("href", ""))
            if not self._is_public_http_url(url):
                continue
            snippet_node = node.select_one(".result__snippet")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            results.append(
                Evidence(
                    source=url,
                    title=link.get_text(" ", strip=True),
                    content=compact_text(snippet, max_chars=1000),
                    metadata={"kind": "web_search_result", "query": query, "provider": "duckduckgo"},
                )
            )
            if len(results) >= limit * 3:
                break
        return results

    def _search_bing(self, query: str, limit: int) -> list[Evidence]:
        search_url = f"https://www.bing.com/search?q={quote_plus(query)}"
        response = self.session.get(search_url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[Evidence] = []
        for node in soup.select("li.b_algo"):
            link = node.find("a")
            if not link:
                continue
            url = self._clean_url(link.get("href", ""))
            if not self._is_public_http_url(url):
                continue
            snippet_node = node.find("p")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            results.append(
                Evidence(
                    source=url,
                    title=link.get_text(" ", strip=True),
                    content=compact_text(snippet, max_chars=1000),
                    metadata={"kind": "web_search_result", "query": query, "provider": "bing"},
                )
            )
            if len(results) >= limit * 3:
                break
        return results

    def _search_yahoo(self, query: str, limit: int) -> list[Evidence]:
        search_url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
        response = self.session.get(search_url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[Evidence] = []
        for node in soup.select("div.dd.algo, div.algo"):
            link = node.select_one("h3 a") or node.find("a")
            if not link:
                continue
            url = self._clean_url(link.get("href", ""))
            if not self._is_public_http_url(url):
                continue
            snippet_node = node.select_one(".compText") or node.select_one("p")
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            title = link.get_text(" ", strip=True)
            results.append(
                Evidence(
                    source=url,
                    title=title,
                    content=compact_text(snippet, max_chars=1000),
                    metadata={"kind": "web_search_result", "query": query, "provider": "yahoo"},
                )
            )
            if len(results) >= limit * 3:
                break
        return results

    def _clean_url(self, url: str) -> str:
        if not url:
            return ""
        if re.fullmatch(r"a1[A-Za-z0-9_-]+", url):
            return self._decode_bing_u(url)
        url = urljoin("https://duckduckgo.com", url)
        parsed = urlparse(url)
        if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
            values = parse_qs(parsed.query).get("uddg")
            if values:
                return unquote(values[0])
        if parsed.netloc.endswith("bing.com") and parsed.path.startswith("/ck/"):
            values = parse_qs(parsed.query).get("u")
            if values:
                return self._decode_bing_u(values[0])
        if parsed.netloc.endswith("search.yahoo.com") and "/RU=" in parsed.path:
            match = re.search(r"/RU=([^/]+)", parsed.path)
            if match:
                return unquote(match.group(1))
        return url

    def _decode_bing_u(self, value: str) -> str:
        value = unquote(value)
        if value.startswith(("http://", "https://")):
            return value
        # Bing commonly emits "a1" + urlsafe-base64(url without padding).
        if re.fullmatch(r"a1[A-Za-z0-9_-]+", value):
            payload = value[2:]
            padding = "=" * (-len(payload) % 4)
            try:
                decoded = base64.urlsafe_b64decode(payload + padding).decode("utf-8", errors="ignore")
                if decoded.startswith(("http://", "https://")):
                    return decoded
            except Exception:
                pass
        return value

    def _rank_and_filter(self, query: str, results: list[Evidence], limit: int) -> list[Evidence]:
        seen: set[str] = set()
        scored: list[tuple[float, Evidence]] = []
        for result in results:
            normalized_url = result.source.rstrip("/")
            if normalized_url in seen:
                continue
            seen.add(normalized_url)
            score = self._relevance_score(query, f"{result.title} {result.content} {result.source}")
            if score <= 0:
                continue
            result.score = round(score, 4)
            scored.append((score, result))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    def _relevance_score(self, query: str, text: str) -> float:
        lowered = text.lower()
        query_terms = {term.lower() for term in re.findall(r"[A-Za-z0-9_.+-]+|[\u4e00-\u9fff]{2,}", query)}
        electronics_terms = {
            "tl431",
            "光耦",
            "纹波",
            "电源",
            "电路",
            "反馈",
            "补偿",
            "反激",
            "限流",
            "电流检测",
            "电阻",
            "电容",
            "开关电源",
            "谐振",
            "ripple",
            "supply",
            "feedback",
            "compensation",
            "flyback",
            "current sense",
            "current sensing",
            "resistor",
            "capacitor",
            "optocoupler",
            "rc filter",
        }
        bad_terms = {
            "limited liability",
            "registered agent",
            "llc formation",
            "business entity",
            "irs",
            "investopedia",
            "forbes",
            "legalzoom",
            "company",
            "powerball",
            "lottery",
        }
        if "llc" in query.lower() and not any(self._term_in_text(term, lowered) for term in electronics_terms):
            return 0.0
        score = sum(1.0 for term in electronics_terms if self._term_in_text(term, lowered))
        if score <= 0:
            return 0.0
        ambiguous = {"llc", "原因", "处理", "电子电路"}
        score += 0.2 * sum(
            1 for term in query_terms if term and term not in ambiguous and self._term_in_text(term, lowered)
        )
        score -= 2.0 * sum(1 for term in bad_terms if self._term_in_text(term, lowered))
        return score

    def _term_in_text(self, term: str, lowered_text: str) -> bool:
        lowered_term = term.lower()
        if re.fullmatch(r"[a-z0-9_.+-]+(?:\s+[a-z0-9_.+-]+)*", lowered_term):
            pattern = r"(?<![a-z0-9_.+-])" + re.escape(lowered_term) + r"(?![a-z0-9_.+-])"
            return re.search(pattern, lowered_text) is not None
        return lowered_term in lowered_text

    def _is_public_http_url(self, url: str) -> bool:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False
        host = parsed.hostname or ""
        blocked = ("localhost", "127.", "10.", "192.168.", "172.16.", "0.0.0.0")
        return not any(host.startswith(prefix) for prefix in blocked)

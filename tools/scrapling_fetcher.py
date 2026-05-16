from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from schemas import Evidence
from tools.content_extractor import extract_llm_markdown

logger = logging.getLogger(__name__)

_PRIVATE_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass(frozen=True)
class ScraplingConfig:
    timeout_seconds: float = 15.0
    max_chars: int = 8000
    mode: str = "fetcher"  # "fetcher", "playwright"/"dynamic", or "stealthy"
    content_source: str = "html"  # "html" or "text"
    stealthy_headers: bool = True
    headless: bool = True
    network_idle: bool = True
    wait_ms: int = 0
    wait_selector: str = ""
    disable_resources: bool = False
    block_images: bool = True
    google_search: bool = True
    real_chrome: bool = False
    auto_match: bool = True
    keep_comments: bool = False
    keep_cdata: bool = False
    huge_tree: bool = True
    css_selector: str = ""
    target_elements: tuple[str, ...] = ()
    candidate_selectors: tuple[str, ...] = (
        "article",
        "main",
        "[role='main']",
        ".article",
        ".article-content",
        ".post",
        ".post-content",
        ".entry-content",
        ".content",
        "#content",
        ".main-content",
        "[class*='content']",
        "[id*='content']",
        "[class*='article']",
        "[id*='article']",
        "[class*='post']",
        "[id*='post']",
        ".detail",
        ".detail-content",
        "[class*='detail']",
        "[id*='detail']",
        "section",
    )
    excluded_tags: tuple[str, ...] = ("nav", "footer", "header", "aside", "script", "style", "noscript", "form")
    excluded_selector: str = (
        "header, footer, nav, aside, form, "
        ".sidebar, .side-bar, .rightbar, .right-bar, .leftbar, .left-bar, "
        ".recommend, .recommended, .related, .related-posts, .hot, .rank, .ranking, "
        ".advertisement, .advertisements, .ads, .ad, .banner, .cookie, .modal, "
        ".comment, .comments, .reply, .share, .social, .breadcrumb, .pagination, "
        ".toolbar, .copyright, .download, .login, .register"
    )


@dataclass
class ScraplingFetchResult:
    evidence: Evidence | None = None
    error: str | None = None
    quality_score: float = 0.0
    quality_reason: str = ""


class ScraplingFetcher:
    def __init__(self, config: ScraplingConfig | None = None) -> None:
        self.config = config or ScraplingConfig()

    def fetch(
        self,
        url: str,
        max_chars: int | None = None,
        *,
        title: str = "",
        snippet: str = "",
    ) -> ScraplingFetchResult:
        max_chars = max_chars or self.config.max_chars
        if not _is_public_http_url(url):
            return ScraplingFetchResult(error=f"blocked non-public URL: {url}")
        try:
            page = self._fetch_page(url)
            status = int(getattr(page, "status", 0) or 0)
            if status and status >= 400:
                return ScraplingFetchResult(error=f"scrapling: HTTP {status}")

            raw_html = self._response_body(page)
            if not raw_html.strip():
                return ScraplingFetchResult(error="scrapling: empty content")

            candidates = self._content_candidates(page, raw_html)
            extracted_candidates = []
            for candidate_name, selected, source_format in candidates:
                extracted = extract_llm_markdown(
                    selected,
                    title=title,
                    snippet=snippet,
                    max_chars=max_chars,
                    source_format=source_format,
                )
                if extracted.content.strip():
                    extracted_candidates.append((candidate_name, selected, extracted))
            if not extracted_candidates:
                return ScraplingFetchResult(error="scrapling: content empty after extraction")
            selected_candidate, selected, extracted = max(
                extracted_candidates,
                key=lambda item: (item[2].quality_score, min(item[2].clean_chars, 20000)),
            )
            candidate_scores = [
                {
                    "candidate": candidate_name,
                    "score": round(candidate.quality_score, 4),
                    "clean_chars": candidate.clean_chars,
                    "reason": candidate.quality_reason,
                }
                for candidate_name, _selected_html, candidate in extracted_candidates
            ]

            evidence = Evidence(
                source=url,
                title=url,
                content=extracted.content,
                metadata={
                    "kind": "web_page",
                    "scrapling_status_code": status or None,
                    "scrapling_mode": self.config.mode,
                    "scrapling_content_source": self.config.content_source,
                    "scrapling_raw_chars": len(raw_html),
                    "scrapling_selected_chars": len(selected),
                    "scrapling_selected_candidate": selected_candidate,
                    "scrapling_candidate_scores": candidate_scores,
                    "extractor_metadata": extracted.metadata,
                },
            )
            return ScraplingFetchResult(
                evidence=evidence,
                quality_score=extracted.quality_score,
                quality_reason=extracted.quality_reason,
            )
        except ImportError as exc:
            return ScraplingFetchResult(error=f"scrapling: dependency missing: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.debug("scrapling fetch failed for %s: %s", url, exc)
            return ScraplingFetchResult(error=f"scrapling: {exc}")

    def _fetch_page(self, url: str) -> Any:
        mode = self.config.mode.lower().strip()
        timeout_ms = max(1000, int(self.config.timeout_seconds * 1000))
        selector_config = self._selector_config()
        if mode == "dynamic":
            from scrapling.fetchers import PlayWrightFetcher

            return PlayWrightFetcher.fetch(
                url,
                headless=self.config.headless,
                disable_resources=self.config.disable_resources,
                network_idle=self.config.network_idle,
                timeout=timeout_ms,
                wait=self.config.wait_ms,
                wait_selector=self.config.wait_selector or None,
                google_search=self.config.google_search,
                real_chrome=self.config.real_chrome,
                custom_config=selector_config,
            )
        if mode == "playwright":
            from scrapling.fetchers import PlayWrightFetcher

            return PlayWrightFetcher.fetch(
                url,
                headless=self.config.headless,
                disable_resources=self.config.disable_resources,
                network_idle=self.config.network_idle,
                timeout=timeout_ms,
                wait=self.config.wait_ms,
                wait_selector=self.config.wait_selector or None,
                google_search=self.config.google_search,
                real_chrome=self.config.real_chrome,
                custom_config=selector_config,
            )
        if mode == "stealthy":
            from scrapling.fetchers import StealthyFetcher

            return StealthyFetcher.fetch(
                url,
                headless=self.config.headless,
                block_images=self.config.block_images,
                disable_resources=self.config.disable_resources,
                network_idle=self.config.network_idle,
                timeout=timeout_ms,
                wait=self.config.wait_ms,
                wait_selector=self.config.wait_selector or None,
                google_search=self.config.google_search,
                custom_config=selector_config,
            )
        from scrapling.fetchers import Fetcher

        return Fetcher.get(
            url,
            timeout=self.config.timeout_seconds,
            stealthy_headers=self.config.stealthy_headers,
            custom_config=selector_config,
        )

    def _selector_config(self) -> dict[str, Any]:
        return {
            "auto_match": self.config.auto_match,
            "keep_comments": self.config.keep_comments,
            "keep_cdata": self.config.keep_cdata,
            "huge_tree": self.config.huge_tree,
        }

    def _response_body(self, page: Any) -> str:
        body = getattr(page, "body", b"")
        if isinstance(body, bytes):
            encoding = getattr(page, "encoding", None) or "utf-8"
            return body.decode(encoding, errors="replace")
        return str(body or page)

    def _content_candidates(self, page: Any, html: str) -> list[tuple[str, str, str]]:
        source_format = "html" if self.config.content_source.lower().strip() == "html" else "text"
        candidates: list[tuple[str, str, str]] = []
        parser = self._parser(page, html)

        strict_selectors = [self.config.css_selector, *self.config.target_elements]
        for selector in [item for item in strict_selectors if item]:
            selected = self._scrapling_select(parser, selector)
            if selected:
                candidates.append((f"selector:{selector}", self._clean_html(selected), source_format))

        if not strict_selectors:
            for selector in self.config.candidate_selectors:
                selected = self._scrapling_select(parser, selector)
                if selected:
                    candidates.append((f"candidate:{selector}", self._clean_html(selected), source_format))

        candidates.append(("full_cleaned_html", self._clean_html(html), source_format))
        deduped: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for name, content, fmt in candidates:
            key = content[:1000]
            if content.strip() and key not in seen:
                seen.add(key)
                deduped.append((name, content, fmt))
        return deduped

    def _parser(self, page: Any, html: str) -> Any:
        if hasattr(page, "css") and hasattr(page, "xpath"):
            return page
        from scrapling.parser import Adaptor

        return Adaptor(text=html, url=getattr(page, "url", None), body=getattr(page, "body", b""), **self._selector_config())

    def _scrapling_select(self, parser: Any, selector: str) -> str:
        try:
            selected = parser.css(selector)
            if not selected:
                return ""
            items = selected.get_all() if hasattr(selected, "get_all") else list(selected)
            return "\n".join(str(item) for item in items)
        except Exception:
            return ""

    def _clean_html(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag_name in self.config.excluded_tags:
            for tag in soup.find_all(tag_name):
                tag.decompose()
        if self.config.excluded_selector:
            for tag in soup.select(self.config.excluded_selector):
                tag.decompose()

        if self.config.content_source.lower().strip() == "text":
            return soup.get_text("\n", strip=True)
        return str(soup)


def _is_public_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    hostname = (parsed.hostname or "").lower()
    if not hostname or hostname in _PRIVATE_HOSTS:
        return False
    parts = hostname.split(".")
    if len(parts) == 4:
        try:
            first = int(parts[0])
            if first == 10 or first == 127:
                return False
            if first == 172 and 16 <= int(parts[1]) <= 31:
                return False
            if first == 192 and int(parts[1]) == 168:
                return False
            if first == 169 and int(parts[1]) == 254:
                return False
        except ValueError:
            pass
    return True

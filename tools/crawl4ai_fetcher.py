from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from schemas import Evidence
from tools.content_extractor import extract_llm_markdown, ExtractedContent

logger = logging.getLogger(__name__)

_PRIVATE_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


@dataclass(frozen=True)
class Crawl4AIConfig:
    timeout_seconds: float = 15.0
    max_chars: int = 6000
    word_count_threshold: int = 10
    use_content_filter: bool = True
    content_filter_type: str = "pruning"  # "pruning" or "bm25"
    markdown_mode: str = "best"  # "best", "fit", or "raw"
    content_source: str = "markdown"  # "best", "markdown", or "cleaned_html"
    browser_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
    )
    viewport_width: int = 1365
    viewport_height: int = 900
    enable_stealth: bool = True
    remove_overlay_elements: bool = True
    remove_consent_popups: bool = True
    scan_full_page: bool = True
    scroll_delay: float = 0.2
    max_scroll_steps: int = 5
    wait_until: str = "domcontentloaded"
    delay_before_return_html: float = 0.5
    excluded_tags: tuple[str, ...] = ("nav", "footer", "aside", "script", "style", "noscript", "form")
    excluded_selector: str = (
        "header, footer, nav, aside, form, "
        ".sidebar, .side-bar, .rightbar, .right-bar, .leftbar, .left-bar, "
        ".recommend, .recommended, .related, .related-posts, .hot, .rank, .ranking, "
        ".advertisement, .advertisements, .ads, .ad, .banner, .cookie, .modal, "
        ".comment, .comments, .reply, .share, .social, .breadcrumb, .pagination, "
        ".toolbar, .copyright, .download, .login, .register"
    )
    css_selector: str = ""
    target_elements: tuple[str, ...] = ()
    ignore_links: bool = False


@dataclass
class Crawl4AIFetchResult:
    evidence: Evidence | None = None
    error: str | None = None
    quality_score: float = 0.0
    quality_reason: str = ""


class Crawl4AIFetcher:
    def __init__(self, config: Crawl4AIConfig | None = None) -> None:
        self.config = config or Crawl4AIConfig()

    def fetch(
        self,
        url: str,
        max_chars: int | None = None,
        *,
        title: str = "",
        snippet: str = "",
    ) -> Crawl4AIFetchResult:
        max_chars = max_chars or self.config.max_chars
        if not _is_public_http_url(url):
            return Crawl4AIFetchResult(error=f"blocked non-public URL: {url}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return self._fetch_in_thread(url, max_chars, title=title, snippet=snippet)
        try:
            return asyncio.run(self._async_fetch(url, max_chars, title=title, snippet=snippet))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self._async_fetch(url, max_chars, title=title, snippet=snippet))
            finally:
                loop.close()
        except Exception as exc:
            logger.debug("crawl4ai fetch failed for %s: %s", url, exc)
            return Crawl4AIFetchResult(error=f"crawl4ai: {exc}")

    def _fetch_in_thread(self, url: str, max_chars: int, *, title: str, snippet: str) -> Crawl4AIFetchResult:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.fetch, url, max_chars, title=title, snippet=snippet)
            return future.result()

    async def _async_fetch(self, url: str, max_chars: int, *, title: str, snippet: str) -> Crawl4AIFetchResult:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
        from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter

        md_generator = None
        if self.config.use_content_filter:
            user_query = compact_context(title, snippet)
            if self.config.content_filter_type == "bm25":
                md_generator = DefaultMarkdownGenerator(
                    content_filter=BM25ContentFilter(user_query=user_query or None),
                    options={"ignore_links": self.config.ignore_links},
                )
            else:
                md_generator = DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(user_query=user_query or None),
                    options={"ignore_links": self.config.ignore_links},
                )
        else:
            md_generator = DefaultMarkdownGenerator(options={"ignore_links": self.config.ignore_links})

        run_config_kwargs: dict[str, Any] = {
            "cache_mode": CacheMode.ENABLED,
            "word_count_threshold": self.config.word_count_threshold,
            "page_timeout": max(1000, int(self.config.timeout_seconds * 1000)),
            "wait_until": self.config.wait_until,
            "delay_before_return_html": self.config.delay_before_return_html,
            "excluded_tags": list(self.config.excluded_tags),
            "excluded_selector": self.config.excluded_selector or None,
            "css_selector": self.config.css_selector or None,
            "target_elements": list(self.config.target_elements) if self.config.target_elements else None,
            "remove_overlay_elements": self.config.remove_overlay_elements,
            "remove_consent_popups": self.config.remove_consent_popups,
            "scan_full_page": self.config.scan_full_page,
            "scroll_delay": self.config.scroll_delay,
            "max_scroll_steps": self.config.max_scroll_steps,
            "simulate_user": True,
            "override_navigator": True,
            "magic": True,
            "verbose": False,
            "log_console": False,
        }
        if md_generator:
            run_config_kwargs["markdown_generator"] = md_generator
        run_config = CrawlerRunConfig(**run_config_kwargs)

        browser_config = BrowserConfig(
            headless=True,
            user_agent=self.config.browser_user_agent,
            viewport_width=self.config.viewport_width,
            viewport_height=self.config.viewport_height,
            enable_stealth=self.config.enable_stealth,
            verbose=False,
        )
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await crawler.arun(url=url, config=run_config)

        if not result.success:
            return Crawl4AIFetchResult(error=f"crawl4ai: {result.error_message or 'crawl failed'}")

        extraction_candidates: list[tuple[str, str, str]] = []
        content_source = self.config.content_source.lower().strip()
        use_markdown = content_source in {"best", "markdown"}
        use_cleaned_html = content_source in {"best", "cleaned_html", "html"}
        if use_markdown and result.markdown:
            selected_markdown = self._select_markdown(
                fit_markdown=result.markdown.fit_markdown or "",
                raw_markdown=result.markdown.raw_markdown or "",
                title=title,
                snippet=snippet,
                max_chars=max_chars,
            )
            if selected_markdown:
                extraction_candidates.append(("crawl4ai_markdown", selected_markdown, "text"))
        if use_cleaned_html and result.cleaned_html:
            extraction_candidates.append(("crawl4ai_cleaned_html", result.cleaned_html, "html"))
        if not extraction_candidates:
            return Crawl4AIFetchResult(error="crawl4ai: empty content")

        extracted_candidates: list[tuple[str, ExtractedContent]] = []
        for candidate_name, candidate_text, source_format in extraction_candidates:
            extracted_candidate = extract_llm_markdown(
                candidate_text,
                title=title,
                snippet=snippet,
                max_chars=max_chars,
                source_format=source_format,
            )
            if extracted_candidate.content.strip():
                extracted_candidates.append((candidate_name, extracted_candidate))
        if not extracted_candidates:
            return Crawl4AIFetchResult(error="crawl4ai: content empty after extraction")
        selected_candidate, extracted = max(
            extracted_candidates,
            key=lambda item: (item[1].quality_score, min(item[1].clean_chars, 20000)),
        )
        if not extracted.content.strip():
            return Crawl4AIFetchResult(error="crawl4ai: content empty after extraction")

        evidence = Evidence(
            source=url,
            title=url,
            content=extracted.content,
            metadata={
                "kind": "web_page",
                "crawl4ai_raw_chars": max(len(text) for _name, text, _fmt in extraction_candidates),
                "crawl4ai_status_code": result.status_code,
                "crawl4ai_markdown_mode": self.config.markdown_mode,
                "crawl4ai_content_source": self.config.content_source,
                "crawl4ai_selected_candidate": selected_candidate,
                "extractor_metadata": extracted.metadata,
            },
        )
        return Crawl4AIFetchResult(
            evidence=evidence,
            quality_score=extracted.quality_score,
            quality_reason=extracted.quality_reason,
        )

    def _select_markdown(
        self,
        *,
        fit_markdown: str,
        raw_markdown: str,
        title: str,
        snippet: str,
        max_chars: int,
    ) -> str:
        mode = self.config.markdown_mode.lower().strip()
        if mode == "fit":
            return fit_markdown or raw_markdown
        if mode == "raw":
            return raw_markdown or fit_markdown

        candidates: list[ExtractedContent] = []
        by_content: dict[str, str] = {}
        for name, content in (("fit", fit_markdown), ("raw", raw_markdown)):
            if not content:
                continue
            extracted = extract_llm_markdown(
                content,
                title=title,
                snippet=snippet,
                max_chars=max_chars,
                source_format="text",
            )
            candidates.append(extracted)
            by_content[extracted.extractor + str(extracted.raw_chars)] = content
        if not candidates:
            return fit_markdown or raw_markdown
        best = max(candidates, key=lambda item: (item.quality_score, item.clean_chars))
        return by_content.get(best.extractor + str(best.raw_chars), raw_markdown or fit_markdown)


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
        except ValueError:
            pass
    return True


def compact_context(title: str, snippet: str) -> str:
    context = " ".join(part.strip() for part in [title or "", snippet or ""] if part and part.strip())
    return " ".join(context.split())[:500]

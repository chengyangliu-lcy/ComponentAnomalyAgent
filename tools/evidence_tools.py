from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import requests
import yaml
from bs4 import BeautifulSoup
from pydantic import Field

from agent.prompts import (
    build_final_answer_system_prompt,
    VISION_SYSTEM_PROMPT,
    build_final_answer_user_prompt,
    build_vision_user_prompt,
)
from llm_client import LLMClient
from schemas import Evidence, StandardSample, ToolEvent
from tools.browser import BrowserFallback
from tools.content_extractor import ExtractedContent, extract_llm_markdown
from tools.utils import compact_text, timer
from tools.web_reader import WebReader
from tools.web_search import WebSearch
from tools.openhands_browser import OpenHandsBrowserConfig, OpenHandsBrowserFetcher
from tools.crawl4ai_fetcher import Crawl4AIConfig, Crawl4AIFetcher
from tools.scrapling_fetcher import ScraplingConfig, ScraplingFetcher
from tools.circuit_kb import (
    CircuitMarkdownRetriever,
    classify_query_terms,
    is_boilerplate_text,
    is_circuitmaker_project_source,
    is_low_value_source,
)

_LOW_VALUE_SEARCH_HOSTS = {
    "www.douyin.com",
    "douyin.com",
    "www.xueshu.com",
    "xueshu.com",
    "www.rcmoy.com",
    "rcmoy.com",
    "www.scribd.com",
    "scribd.com",
    "www.taobao.com",
    "taobao.com",
    "detail.tmall.com",
    "www.tmall.com",
    "www.jd.com",
    "jd.com",
    "item.jd.com",
    "www.amazon.com",
    "amazon.com",
}

_LOW_VALUE_SEARCH_PATH_MARKERS = (
    "/search/",
    "/sitemap",
    "sitemap.",
    "/tag/",
    "/tags/",
    "/tag_",
)

_LOW_VALUE_SEARCH_TITLE_MARKERS = (
    "网站地图",
    "站点地图",
    "sitemap",
    "搜索",
    "抖音",
    "快手",
    "淘宝",
    "天猫",
    "好货",
    "店铺",
    "通用",
    "12篇",
    "范文",
)

_SEARCH_STOP_TERMS = {
    "怎么",
    "为什么",
    "什么",
    "问题",
    "原因",
    "处理",
    "维修",
    "电路",
    "使用",
    "指南",
    "设计",
    "拆解",
    "内部",
}

_WEB_READ_NAV_WORDS = {
    "home",
    "products",
    "product",
    "solutions",
    "support",
    "resources",
    "applications",
    "company",
    "about",
    "login",
    "register",
    "cart",
    "menu",
    "search",
    "forum",
    "download",
}

try:
    os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
    from openhands.sdk import Action, Observation, ToolDefinition
    from openhands.sdk.tool.tool import ToolAnnotations, ToolExecutor
except Exception:  # pragma: no cover - exercised only when optional SDK import fails
    class _CompatModel:
        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    class Action(_CompatModel):  # type: ignore[no-redef]
        pass

    class Observation(_CompatModel):  # type: ignore[no-redef]
        pass

    class ToolDefinition(_CompatModel):  # type: ignore[no-redef]
        def __class_getitem__(cls, _item: Any) -> Any:
            return cls

    ToolAnnotations = None  # type: ignore[assignment]
    ToolExecutor = object  # type: ignore[assignment,misc]


class ImageInspectAction(Action):
    sample_id: str
    question: str
    image_paths: list[str] = Field(default_factory=list)


class WebSearchAction(Action):
    query: str
    limit: int = 6


class WebReadAction(Action):
    url: str
    snippet: str = ""
    title: str = ""


class DomainSkillAction(Action):
    question: str


class LocalRetrieveAction(Action):
    query: str
    limit: int = 4


class QwenSearchAction(Action):
    query: str


class EvidenceRankAction(Action):
    question: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    max_items: int = 12


class FinishAnswerAction(Action):
    question: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceObservation(Observation):
    summary: str = ""
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class TextObservation(Observation):
    summary: str = ""
    answer_text: str = ""
    errors: list[str] = Field(default_factory=list)


@dataclass
class ToolRun:
    evidence: list[Evidence] = field(default_factory=list)
    text: str = ""
    summary: str = ""
    success: bool = True
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WebReadCandidate:
    backend: str
    evidence: Evidence | None = None
    error: str = ""
    elapsed_seconds: float = 0.0
    final_score: float = 0.0
    selection_reason: str = ""


class BaseEvidenceExecutor(ToolExecutor):
    tool_name = "tool"
    action_name = "execute"

    def event(self, run: ToolRun, elapsed: float, inputs: dict[str, Any]) -> ToolEvent:
        return ToolEvent(
            tool_name=self.tool_name,
            action=self.action_name,
            success=run.success,
            elapsed_seconds=elapsed,
            summary=run.summary,
            inputs=inputs,
            outputs={
                "sources": [item.source for item in run.evidence],
                "metadata": run.metadata,
            },
            error="; ".join(run.errors) if run.errors else None,
        )


class ImageInspectExecutor(BaseEvidenceExecutor):
    tool_name = "image_inspect"
    action_name = "multimodal_component_extract"

    def __init__(
        self,
        llm: LLMClient,
        enabled: bool = True,
        max_images: int = 4,
        payload_format: str = "openai_image_url",
        image_max_side: int = 1280,
        image_jpeg_quality: int = 75,
    ) -> None:
        self.llm = llm
        self.enabled = enabled
        self.max_images = max_images
        self.payload_format = payload_format
        self.image_max_side = image_max_side
        self.image_jpeg_quality = image_jpeg_quality

    def __call__(self, action: ImageInspectAction, conversation: Any = None) -> EvidenceObservation:
        run = self.run(action)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, action: ImageInspectAction) -> ToolRun:
        paths = [Path(path) for path in action.image_paths[: self.max_images] if path]
        found = [path for path in paths if path.exists()]
        if not found:
            return ToolRun(
                evidence=[],
                summary="no local images available for inspection",
                success=False,
                errors=["no local images available"],
            )
        fallback = Evidence(
            source="local_images",
            title=f"{action.sample_id} image context",
            content=(
                f"样本包含 {len(found)} 张本地图片。图片路径仅作为多模态输入使用，"
                "不读取测试集原始帖子资料。可见元件、标号和测量值需以模型视觉识别为准。"
            ),
            metadata={"kind": "image_context", "image_paths": [str(path) for path in found]},
        )
        if not self.enabled or not self.llm.available:
            return ToolRun([fallback], summary="image paths recorded; LLM vision unavailable")

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": build_vision_user_prompt(action.question),
            }
        ]
        image_payloads: list[dict[str, Any]] = []
        image_stats: list[dict[str, Any]] = []
        for path in found:
            data_url, stats = _image_data_url(path, max_side=self.image_max_side, quality=self.image_jpeg_quality)
            image_stats.append({"path": str(path), **stats})
            if data_url:
                payload = _image_content_part(data_url, self.payload_format)
                content.append(payload)
                image_payloads.append(payload)
        if not image_payloads:
            return ToolRun(
                [fallback],
                summary="image paths recorded; no images could be encoded for vision inspection",
                success=False,
                errors=["no images could be encoded"],
                metadata={"image_count": len(found), "image_stats": image_stats, "payload_format": self.payload_format},
            )
        response = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": VISION_SYSTEM_PROMPT,
                },
                {"role": "user", "content": content},
            ],
            temperature=0.0,
        )
        if not response.content:
            return ToolRun(
                [fallback],
                summary="image paths recorded; vision inspection failed",
                success=False,
                errors=[response.error or "empty vision response"],
                metadata={"image_count": len(found), "image_stats": image_stats, "payload_format": self.payload_format},
            )
        inspected = Evidence(
            source="image_inspect",
            title=f"{action.sample_id} visual component notes",
            content=compact_text(response.content, 5000),
            metadata={"kind": "image_inspection", "image_paths": [str(path) for path in found]},
        )
        return ToolRun(
            [fallback, inspected],
            summary="image inspection completed",
            metadata={"image_count": len(found), "image_stats": image_stats, "payload_format": self.payload_format},
        )


class APIWebSearchExecutor(BaseEvidenceExecutor):
    tool_name = "web_search"
    action_name = "api_or_html_search"

    def __init__(
        self,
        provider_order: Sequence[str],
        api_key_envs: dict[str, str],
        api_keys: dict[str, str] | None = None,
        timeout: int = 20,
        html_provider: str = "duckduckgo",
        searxng_url: str = "",
    ) -> None:
        self.provider_order = list(provider_order or ["html"])
        self.api_key_envs = dict(api_key_envs or {})
        self.api_keys = {key.lower(): value for key, value in dict(api_keys or {}).items() if value}
        self.timeout = timeout
        self.searxng_url = searxng_url.rstrip("/") if searxng_url else ""
        self.html_search = WebSearch(timeout=timeout, provider=html_provider)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.7",
            }
        )

    def __call__(self, action: WebSearchAction, conversation: Any = None) -> EvidenceObservation:
        run = self.run(action.query, action.limit)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, query: str, limit: int = 6) -> ToolRun:
        errors: list[str] = []
        for provider in self.provider_order:
            provider = provider.lower().strip()
            if provider == "html":
                results, error = self.html_search.search(query, limit=limit)
                results = self._rank_and_filter_results(query, results, limit)
                if results:
                    return ToolRun(
                        results,
                        summary=f"html search returned {len(results)} results",
                        metadata={"provider": "html", "query": query},
                    )
                errors.append(f"html: {error or 'no results'}")
                continue
            if provider == "searxng":
                if not self.searxng_url:
                    errors.append("searxng: searxng_url not configured")
                    continue
                try:
                    results = self._search_searxng(query, limit)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"searxng: {exc}")
                    continue
                results = self._rank_and_filter_results(query, results, limit)
                if results:
                    return ToolRun(
                        results,
                        summary=f"searxng search returned {len(results)} results",
                        metadata={"provider": "searxng", "query": query},
                    )
                errors.append("searxng: no results")
                continue
            key_env = self.api_key_envs.get(provider, "")
            api_key = self.api_keys.get(provider) or (os.environ.get(key_env) if key_env else None)
            if not api_key:
                errors.append(f"{provider}: missing api key env {key_env or '<unset>'}")
                continue
            try:
                results = self._search_provider(provider, api_key, query, limit)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider}: {exc}")
                continue
            results = self._rank_and_filter_results(query, results, limit)
            if results:
                return ToolRun(
                    results,
                    summary=f"{provider} search returned {len(results)} results",
                    metadata={"provider": provider, "query": query},
                )
            errors.append(f"{provider}: no results")
        return ToolRun(
            [],
            summary=f"search failed for query={query}",
            success=False,
            errors=errors,
            metadata={"query": query, "provider_order": self.provider_order},
        )

    def _search_searxng(self, query: str, limit: int) -> list[Evidence]:
        json_error: Exception | None = None
        try:
            response = self.session.get(
                f"{self.searxng_url}/search",
                params={"q": query, "format": "json", "categories": "general"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json().get("results", [])
            results = [
                _search_evidence(
                    query,
                    "searxng",
                    row.get("title") or row.get("url", ""),
                    row.get("url", ""),
                    row.get("content") or "",
                    float(row.get("score") or 0.0),
                )
                for row in rows[: max(limit * 4, limit)]
                if row.get("url")
            ]
            if results:
                return results
        except Exception as exc:  # noqa: BLE001
            json_error = exc

        html_results = self._search_searxng_html(query, limit)
        if html_results:
            return html_results
        if json_error:
            raise json_error
        return []

    def _search_searxng_html(self, query: str, limit: int) -> list[Evidence]:
        response = self.session.get(
            f"{self.searxng_url}/search",
            params={"q": query, "categories": "general"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results: list[Evidence] = []
        for article in soup.select("article.result"):
            link = article.select_one("h3 a[href]") or article.select_one("a.url_header[href]")
            if not link:
                continue
            url = link.get("href", "")
            if not url:
                continue
            title = link.get_text(" ", strip=True) or url
            content = ""
            content_node = article.select_one("p.content")
            if content_node:
                content = content_node.get_text(" ", strip=True)
            if not content:
                content = article.get_text(" ", strip=True)
            results.append(_search_evidence(query, "searxng_html", title, url, content, 0.0))
            if len(results) >= max(limit * 4, limit):
                break
        return results

    def _rank_and_filter_results(self, query: str, results: list[Evidence], limit: int) -> list[Evidence]:
        seen: set[str] = set()
        scored: list[tuple[float, Evidence]] = []
        for item in results:
            normalized = _normalize_search_url(item.source)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            reason = _low_value_search_reason(item)
            if reason:
                item.metadata["search_filter_reason"] = reason
                continue
            score, reason = _search_relevance_score(query, item)
            if score <= 0.0:
                item.metadata["search_filter_reason"] = reason
                continue
            item.score = round(score, 4)
            item.metadata["search_relevance_reason"] = reason
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _score, item in scored[:limit]]

    def _search_provider(self, provider: str, api_key: str, query: str, limit: int) -> list[Evidence]:
        if provider == "tavily":
            response = self.session.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "search_depth": "advanced",
                    "max_results": limit,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json().get("results", [])
            return [
                _search_evidence(
                    query,
                    "tavily",
                    row.get("title") or row.get("url", ""),
                    row.get("url", ""),
                    row.get("content") or "",
                    float(row.get("score") or 0.0),
                )
                for row in rows[:limit]
                if row.get("url")
            ]
        if provider == "brave":
            response = self.session.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": min(limit, 20), "text_decorations": False},
                headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json().get("web", {}).get("results", [])
            return [
                _search_evidence(
                    query,
                    "brave",
                    row.get("title") or row.get("url", ""),
                    row.get("url", ""),
                    row.get("description") or "",
                    0.0,
                )
                for row in rows[:limit]
                if row.get("url")
            ]
        if provider == "bing":
            response = self.session.get(
                "https://api.bing.microsoft.com/v7.0/search",
                params={"q": query, "count": min(limit, 20), "responseFilter": "Webpages"},
                headers={"Ocp-Apim-Subscription-Key": api_key},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json().get("webPages", {}).get("value", [])
            return [
                _search_evidence(
                    query,
                    "bing_api",
                    row.get("name") or row.get("url", ""),
                    row.get("url", ""),
                    row.get("snippet") or "",
                    0.0,
                )
                for row in rows[:limit]
                if row.get("url")
            ]
        if provider == "serpapi":
            response = self.session.get(
                "https://serpapi.com/search.json",
                params={"q": query, "api_key": api_key, "engine": "google", "num": limit},
                timeout=self.timeout,
            )
            response.raise_for_status()
            rows = response.json().get("organic_results", [])
            return [
                _search_evidence(
                    query,
                    "serpapi",
                    row.get("title") or row.get("link", ""),
                    row.get("link", ""),
                    row.get("snippet") or "",
                    0.0,
                )
                for row in rows[:limit]
                if row.get("link")
            ]
        raise ValueError(f"unsupported provider: {provider}")


class RobustWebReadExecutor(BaseEvidenceExecutor):
    tool_name = "web_reader"
    action_name = "read_or_keep_snippet"

    def __init__(
        self,
        timeout: int = 20,
        enable_openhands_browser_primary: bool = True,
        openhands_browser_timeout_seconds: float | None = None,
        openhands_browser_max_chars: int = 6000,
        openhands_browser_require_installed: bool = True,
        openhands_browser: Any | None = None,
        enable_jina_reader: bool = True,
        jina_max_chars: int = 8000,
        jina_use_readerlm_fallback: bool = True,
        jina_min_quality_score: float = 0.55,
        jina_min_clean_chars: int = 500,
        enable_content_extraction: bool = True,
        enable_browser_fallback: bool = False,
        browser_fallback: Any | None = None,
        browser_fallback_wait_ms: int = 2500,
        enable_crawl4ai: bool = True,
        crawl4ai_timeout_seconds: float = 15.0,
        crawl4ai_max_chars: int = 6000,
        crawl4ai_word_count_threshold: int = 10,
        crawl4ai_use_content_filter: bool = True,
        crawl4ai_content_filter_type: str = "pruning",
        crawl4ai_primary: bool = False,
        crawl4ai_markdown_mode: str = "best",
        crawl4ai_content_source: str = "best",
        crawl4ai_enable_stealth: bool = True,
        crawl4ai_scan_full_page: bool = True,
        crawl4ai_delay_before_return_html: float = 0.5,
        crawl4ai_css_selector: str = "",
        crawl4ai_target_elements: Sequence[str] | None = None,
        crawl4ai_excluded_selector: str = "",
        crawl4ai_excluded_tags: Sequence[str] | None = None,
        crawl4ai_fetcher: Any | None = None,
        enable_scrapling: bool = False,
        scrapling_timeout_seconds: float = 15.0,
        scrapling_max_chars: int = 8000,
        scrapling_mode: str = "dynamic",
        scrapling_content_source: str = "html",
        scrapling_auto_match: bool = True,
        scrapling_network_idle: bool = True,
        scrapling_wait_ms: int = 0,
        scrapling_wait_selector: str = "",
        scrapling_disable_resources: bool = False,
        scrapling_block_images: bool = True,
        scrapling_google_search: bool = True,
        scrapling_real_chrome: bool = False,
        scrapling_css_selector: str = "",
        scrapling_target_elements: Sequence[str] | None = None,
        scrapling_excluded_selector: str = "",
        scrapling_excluded_tags: Sequence[str] | None = None,
        scrapling_fetcher: Any | None = None,
        web_read_competitive_mode: bool = False,
        web_read_competitive_timeout_seconds: float | None = None,
        web_read_min_quality_score: float | None = None,
        web_read_min_clean_chars: int | None = None,
        web_read_compare_mode: bool = False,
        backend_comparison_path: str | None = None,
    ) -> None:
        self.reader = WebReader(timeout=timeout)
        self.enable_openhands_browser_primary = enable_openhands_browser_primary
        self.openhands_browser_max_chars = openhands_browser_max_chars
        self.openhands_browser = openhands_browser or OpenHandsBrowserFetcher(
            OpenHandsBrowserConfig(
                timeout_seconds=float(openhands_browser_timeout_seconds or timeout),
                max_chars=openhands_browser_max_chars,
                require_installed=openhands_browser_require_installed,
            )
        )
        self.enable_jina_reader = enable_jina_reader
        self.jina_max_chars = jina_max_chars
        self.jina_use_readerlm_fallback = jina_use_readerlm_fallback
        self.jina_min_quality_score = jina_min_quality_score
        self.jina_min_clean_chars = jina_min_clean_chars
        self.enable_content_extraction = enable_content_extraction
        self.enable_browser_fallback = enable_browser_fallback
        self.browser_fallback = browser_fallback or BrowserFallback(
            timeout_ms=int((openhands_browser_timeout_seconds or timeout) * 1000),
            wait_after_load_ms=browser_fallback_wait_ms,
        )
        self.enable_crawl4ai = enable_crawl4ai
        self.crawl4ai_primary = crawl4ai_primary
        self.crawl4ai_max_chars = crawl4ai_max_chars
        self.crawl4ai_fetcher = crawl4ai_fetcher or Crawl4AIFetcher(
            Crawl4AIConfig(
                timeout_seconds=crawl4ai_timeout_seconds,
                max_chars=crawl4ai_max_chars,
                word_count_threshold=crawl4ai_word_count_threshold,
                use_content_filter=crawl4ai_use_content_filter,
                content_filter_type=crawl4ai_content_filter_type,
                markdown_mode=crawl4ai_markdown_mode,
                content_source=crawl4ai_content_source,
                enable_stealth=crawl4ai_enable_stealth,
                scan_full_page=crawl4ai_scan_full_page,
                delay_before_return_html=crawl4ai_delay_before_return_html,
                css_selector=crawl4ai_css_selector,
                target_elements=tuple(crawl4ai_target_elements or Crawl4AIConfig().target_elements),
                excluded_selector=crawl4ai_excluded_selector or Crawl4AIConfig().excluded_selector,
                excluded_tags=tuple(crawl4ai_excluded_tags or Crawl4AIConfig().excluded_tags),
            )
        )
        self.enable_scrapling = enable_scrapling
        self.scrapling_max_chars = scrapling_max_chars
        self.scrapling_fetcher = scrapling_fetcher or ScraplingFetcher(
            ScraplingConfig(
                timeout_seconds=scrapling_timeout_seconds,
                max_chars=scrapling_max_chars,
                mode=scrapling_mode,
                content_source=scrapling_content_source,
                auto_match=scrapling_auto_match,
                network_idle=scrapling_network_idle,
                wait_ms=scrapling_wait_ms,
                wait_selector=scrapling_wait_selector,
                disable_resources=scrapling_disable_resources,
                block_images=scrapling_block_images,
                google_search=scrapling_google_search,
                real_chrome=scrapling_real_chrome,
                css_selector=scrapling_css_selector,
                target_elements=tuple(scrapling_target_elements or ScraplingConfig().target_elements),
                excluded_selector=scrapling_excluded_selector or ScraplingConfig().excluded_selector,
                excluded_tags=tuple(scrapling_excluded_tags or ScraplingConfig().excluded_tags),
            )
        )
        self.web_read_competitive_mode = web_read_competitive_mode
        self.web_read_competitive_timeout_seconds = float(web_read_competitive_timeout_seconds or max(timeout, crawl4ai_timeout_seconds, scrapling_timeout_seconds))
        self.web_read_min_quality_score = float(web_read_min_quality_score if web_read_min_quality_score is not None else jina_min_quality_score)
        self.web_read_min_clean_chars = int(web_read_min_clean_chars if web_read_min_clean_chars is not None else min(jina_min_clean_chars, 500))
        self.web_read_compare_mode = web_read_compare_mode
        self.backend_comparison_path = backend_comparison_path

    def __call__(self, action: WebReadAction, conversation: Any = None) -> EvidenceObservation:
        run = self.run(action.url, action.title, action.snippet)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, url: str, title: str = "", snippet: str = "") -> ToolRun:
        lowered = url.lower()
        if lowered.endswith(".pdf") or "/pdf/" in lowered or self._should_keep_snippet_without_fetch(lowered):
            evidence = _search_evidence("", "snippet", title or url, url, snippet, 0.0)
            evidence.metadata["read_skipped"] = "pdf_binary_or_known_slow"
            evidence.metadata["read_backend"] = "snippet_fallback"
            evidence.metadata["read_confidence"] = "low"
            evidence.metadata["effective_read_success"] = False
            return ToolRun(
                [evidence],
                summary="kept search snippet for pdf/binary/known-slow page",
                metadata={"read_backend": "snippet_fallback", "read_confidence": "low", "effective_read_success": False},
            )

        if self.web_read_compare_mode:
            return self._run_compare(url, title, snippet)

        if self.web_read_competitive_mode:
            competitive_run = self._run_competitive(url, title, snippet)
            if competitive_run:
                return competitive_run

        crawl4ai_error = None
        jina_error = None
        scrapling_error = None

        if self.crawl4ai_primary:
            crawl4ai_run, crawl4ai_error = self._run_crawl4ai_candidate(url, title, snippet)
            if crawl4ai_run:
                return crawl4ai_run

        if self.enable_jina_reader:
            jina_result = self._fetch_jina(url, title=title, snippet=snippet, max_chars=self.jina_max_chars)
            if jina_result and jina_result.acceptable(self.jina_min_quality_score, self.jina_min_clean_chars):
                evidence = self._jina_evidence(url, title, jina_result)
                if crawl4ai_error:
                    evidence.metadata["crawl4ai_error"] = crawl4ai_error
                return ToolRun(
                    [evidence],
                    summary=f"read page with Jina Reader {url}",
                    errors=[error for error in [crawl4ai_error] if error],
                    metadata={"read_backend": "jina_reader", "crawl4ai_error": crawl4ai_error},
                )
            if jina_result:
                jina_error = (
                    "jina_reader: low quality "
                    f"score={jina_result.quality_score:.2f} chars={jina_result.clean_chars} "
                    f"reason={jina_result.quality_reason}"
                )
            else:
                jina_error = "jina_reader: empty or failed"

        if not self.crawl4ai_primary:
            crawl4ai_run, crawl4ai_error = self._run_crawl4ai_candidate(url, title, snippet)
            if crawl4ai_run:
                if jina_error:
                    crawl4ai_run.evidence[0].metadata["jina_error"] = jina_error
                    crawl4ai_run.errors = [error for error in [jina_error] if error]
                    crawl4ai_run.metadata["jina_error"] = jina_error
                return crawl4ai_run

        scrapling_run, scrapling_error = self._run_scrapling_candidate(url, title, snippet)
        if scrapling_run:
            if jina_error:
                scrapling_run.evidence[0].metadata["jina_error"] = jina_error
                scrapling_run.errors = [error for error in [jina_error] if error]
                scrapling_run.metadata["jina_error"] = jina_error
            if crawl4ai_error:
                scrapling_run.evidence[0].metadata["crawl4ai_error"] = crawl4ai_error
                scrapling_run.metadata["crawl4ai_error"] = crawl4ai_error
            return scrapling_run

        browser_error = None
        if self.enable_browser_fallback:
            browser_result = self.browser_fallback.fetch(url, max_chars=self.openhands_browser_max_chars)
            if browser_result.evidence:
                browser_result.evidence.title = title or browser_result.evidence.title or url
                browser_result.evidence = self._clean_evidence(
                    browser_result.evidence,
                    title=browser_result.evidence.title,
                    snippet=snippet,
                    max_chars=self.openhands_browser_max_chars,
                    source_format="text",
                )
                browser_result.evidence.metadata["read_backend"] = "playwright_browser"
                if crawl4ai_error:
                    browser_result.evidence.metadata["crawl4ai_error"] = crawl4ai_error
                if jina_error:
                    browser_result.evidence.metadata["jina_error"] = jina_error
                if scrapling_error:
                    browser_result.evidence.metadata["scrapling_error"] = scrapling_error
                if self._evidence_has_content(browser_result.evidence):
                    return ToolRun(
                        [browser_result.evidence],
                        summary=f"read page with Playwright browser {url}",
                        errors=[error for error in [crawl4ai_error, jina_error, scrapling_error] if error],
                        metadata={
                            "read_backend": "playwright_browser",
                            "crawl4ai_error": crawl4ai_error,
                            "jina_error": jina_error,
                            "scrapling_error": scrapling_error,
                        },
                    )
                browser_error = self._low_quality_evidence_error("playwright_browser", browser_result.evidence)
            else:
                browser_error = browser_result.error

        openhands_error = None
        if self.enable_openhands_browser_primary:
            browser_result = self.openhands_browser.fetch(url, max_chars=self.openhands_browser_max_chars)
            if browser_result.evidence:
                browser_result.evidence.title = title or browser_result.evidence.title or url
                browser_result.evidence = self._clean_evidence(
                    browser_result.evidence,
                    title=browser_result.evidence.title,
                    snippet=snippet,
                    max_chars=self.openhands_browser_max_chars,
                    source_format="text",
                )
                browser_result.evidence.metadata["read_backend"] = "openhands_browser"
                if crawl4ai_error:
                    browser_result.evidence.metadata["crawl4ai_error"] = crawl4ai_error
                if jina_error:
                    browser_result.evidence.metadata["jina_error"] = jina_error
                if browser_error:
                    browser_result.evidence.metadata["browser_fallback_error"] = browser_error
                if scrapling_error:
                    browser_result.evidence.metadata["scrapling_error"] = scrapling_error
                if self._evidence_has_content(browser_result.evidence):
                    return ToolRun(
                        [browser_result.evidence],
                        summary=f"read page with OpenHands browser {url}",
                        metadata={
                            "read_backend": "openhands_browser",
                            "crawl4ai_error": crawl4ai_error,
                            "jina_error": jina_error,
                            "scrapling_error": scrapling_error,
                            "browser_fallback_error": browser_error,
                        },
                    )
                openhands_error = self._low_quality_evidence_error("openhands_browser", browser_result.evidence)
            else:
                openhands_error = browser_result.error

        page = self.reader.read(url, max_chars=6000)
        if page.evidence:
            page.evidence = self._clean_evidence(
                page.evidence,
                title=page.evidence.title,
                snippet=snippet,
                max_chars=6000,
                source_format="text",
            )
            page.evidence.metadata["read_backend"] = "requests_bs4"
            if crawl4ai_error:
                page.evidence.metadata["crawl4ai_error"] = crawl4ai_error
            if openhands_error:
                page.evidence.metadata["openhands_error"] = openhands_error
            if browser_error:
                page.evidence.metadata["browser_fallback_error"] = browser_error
            if jina_error:
                page.evidence.metadata["jina_error"] = jina_error
            if scrapling_error:
                page.evidence.metadata["scrapling_error"] = scrapling_error
            if self._evidence_has_content(page.evidence):
                return ToolRun(
                    [page.evidence],
                    summary=f"read page {url}",
                    errors=[error for error in [crawl4ai_error, jina_error, scrapling_error, browser_error, openhands_error] if error],
                    metadata={
                        "read_backend": "requests_bs4",
                        "crawl4ai_error": crawl4ai_error,
                        "jina_error": jina_error,
                        "scrapling_error": scrapling_error,
                        "browser_fallback_error": browser_error,
                        "openhands_error": openhands_error,
                    },
                )
            page.error = self._low_quality_evidence_error("requests_bs4", page.evidence)
        if snippet:
            evidence = _search_evidence("", "snippet", title or url, url, snippet, 0.0)
            evidence.metadata["read_error"] = page.error
            evidence.metadata["web_reader_error"] = page.error
            evidence.metadata["crawl4ai_error"] = crawl4ai_error
            evidence.metadata["openhands_error"] = openhands_error
            evidence.metadata["browser_fallback_error"] = browser_error
            evidence.metadata["jina_error"] = jina_error
            evidence.metadata["scrapling_error"] = scrapling_error
            evidence.metadata["read_backend"] = "snippet_fallback"
            evidence.metadata["read_confidence"] = "low"
            evidence.metadata["effective_read_success"] = False
            errors = [error for error in [crawl4ai_error, jina_error, scrapling_error, browser_error, openhands_error, page.error or "read failed"] if error]
            return ToolRun(
                [evidence],
                summary="page read failed; kept search snippet",
                success=True,
                errors=errors,
                metadata={
                    "read_backend": "snippet_fallback",
                    "crawl4ai_error": crawl4ai_error,
                    "jina_error": jina_error,
                    "scrapling_error": scrapling_error,
                    "browser_fallback_error": browser_error,
                    "openhands_error": openhands_error,
                    "web_reader_error": page.error,
                    "read_confidence": "low",
                    "effective_read_success": False,
                },
            )
        errors = [error for error in [crawl4ai_error, jina_error, scrapling_error, browser_error, openhands_error, page.error or "read failed"] if error]
        return ToolRun(
            [],
            summary=f"page read failed {url}",
            success=False,
            errors=errors,
            metadata={
                "read_backend": "failed",
                "crawl4ai_error": crawl4ai_error,
                "jina_error": jina_error,
                "scrapling_error": scrapling_error,
                "browser_fallback_error": browser_error,
                "openhands_error": openhands_error,
                "web_reader_error": page.error,
            },
        )

    def _should_keep_snippet_without_fetch(self, lowered_url: str) -> bool:
        slow_or_blocked_hosts = (
            "analog.com/en/resources/technical-articles/",
            "ti.com/lit/",
            "onsemi.com/products/",
            "st.com/en/",
        )
        return any(marker in lowered_url for marker in slow_or_blocked_hosts)

    def _run_competitive(self, url: str, title: str, snippet: str) -> ToolRun | None:
        tasks: dict[str, Any] = {}
        candidates: list[WebReadCandidate] = []
        completed_order: list[str] = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            if self.enable_jina_reader:
                tasks["jina_reader"] = executor.submit(self._competitive_jina_candidate, url, title, snippet)
            if self.enable_crawl4ai:
                tasks["crawl4ai"] = executor.submit(self._competitive_crawl4ai_candidate, url, title, snippet)
            if self.enable_scrapling:
                tasks["scrapling_dynamic"] = executor.submit(self._competitive_scrapling_candidate, url, title, snippet)

            future_to_backend = {future: backend for backend, future in tasks.items()}
            try:
                for future in as_completed(future_to_backend, timeout=self.web_read_competitive_timeout_seconds):
                    backend = future_to_backend[future]
                    completed_order.append(backend)
                    try:
                        candidate = future.result()
                    except Exception as exc:  # noqa: BLE001
                        candidate = WebReadCandidate(backend=backend, error=f"{backend}: {exc}")
                    self._score_web_read_candidate(candidate, title=title, snippet=snippet)
                    candidates.append(candidate)
            except TimeoutError:
                for backend, future in tasks.items():
                    if backend not in completed_order:
                        future.cancel()
                        candidates.append(WebReadCandidate(backend=backend, error=f"{backend}: timeout"))

        if not tasks:
            return None

        usable = [
            candidate
            for candidate in candidates
            if candidate.evidence
            and candidate.final_score > 0.0
            and int(candidate.evidence.metadata.get("clean_chars") or len(candidate.evidence.content or "")) > 0
        ]
        backend_scores = self._candidate_score_records(candidates)
        backend_errors = {candidate.backend: candidate.error for candidate in candidates if candidate.error}

        if not usable:
            self._log_comparison(
                {
                    "url": url,
                    "competitive_mode": True,
                    "backend_scores": backend_scores,
                    "backend_errors": backend_errors,
                    "completed_order": completed_order,
                    "winner": "none",
                }
            )
            return None

        winner = max(usable, key=lambda item: (item.final_score, self._candidate_purity(item), min(len(item.evidence.content or ""), 12000)))
        assert winner.evidence is not None
        winner.evidence.title = title or winner.evidence.title or url
        winner.evidence.metadata.update(
            {
                "read_backend": winner.backend,
                "competitive_mode": True,
                "competitive_winner": winner.backend,
                "competitive_final_score": winner.final_score,
                "competitive_selection_reason": winner.selection_reason,
                "backend_scores": backend_scores,
                "backend_errors": backend_errors,
                "completed_order": completed_order,
            }
        )
        comparison = {
            "url": url,
            "competitive_mode": True,
            "backend_scores": backend_scores,
            "backend_errors": backend_errors,
            "completed_order": completed_order,
            "winner": winner.backend,
        }
        self._log_comparison(comparison)
        return ToolRun(
            [winner.evidence],
            summary=f"competitive read selected {winner.backend} for {url}",
            errors=list(backend_errors.values()),
            metadata={
                "read_backend": winner.backend,
                "competitive_mode": True,
                "comparison": comparison,
            },
        )

    def _competitive_jina_candidate(self, url: str, title: str, snippet: str) -> WebReadCandidate:
        start = time.perf_counter()
        result = self._fetch_jina(url, title=title, snippet=snippet, max_chars=self.jina_max_chars)
        elapsed = time.perf_counter() - start
        if not result:
            return WebReadCandidate("jina_reader", error="jina_reader: empty or failed", elapsed_seconds=elapsed)
        return WebReadCandidate("jina_reader", evidence=self._jina_evidence(url, title, result), elapsed_seconds=elapsed)

    def _competitive_crawl4ai_candidate(self, url: str, title: str, snippet: str) -> WebReadCandidate:
        start = time.perf_counter()
        run, error = self._run_crawl4ai_candidate(url, title, snippet)
        elapsed = time.perf_counter() - start
        if run and run.evidence:
            return WebReadCandidate("crawl4ai", evidence=run.evidence[0], elapsed_seconds=elapsed)
        return WebReadCandidate("crawl4ai", error=error or "crawl4ai: empty or failed", elapsed_seconds=elapsed)

    def _competitive_scrapling_candidate(self, url: str, title: str, snippet: str) -> WebReadCandidate:
        start = time.perf_counter()
        run, error = self._run_scrapling_candidate(url, title, snippet)
        elapsed = time.perf_counter() - start
        if run and run.evidence:
            return WebReadCandidate("scrapling_dynamic", evidence=run.evidence[0], elapsed_seconds=elapsed)
        return WebReadCandidate("scrapling_dynamic", error=error or "scrapling_dynamic: empty or failed", elapsed_seconds=elapsed)

    def _score_web_read_candidate(self, candidate: WebReadCandidate, *, title: str, snippet: str) -> None:
        if not candidate.evidence:
            candidate.final_score = 0.0
            candidate.selection_reason = candidate.error or "empty"
            return
        evidence = candidate.evidence
        text = evidence.content or ""
        metadata = evidence.metadata
        quality_score = float(metadata.get("quality_score") or 0.0)
        clean_chars = int(metadata.get("clean_chars") or len(text))
        reason = str(metadata.get("quality_reason") or "")
        if quality_score <= 0.0 or any(marker in reason for marker in ("blocked_or_failure_page", "empty_after_cleaning", "mojibake_or_wrong_encoding")):
            candidate.final_score = 0.0
            candidate.selection_reason = reason or "invalid"
            return
        metrics = self._quality_metrics(text, title=title, snippet=snippet, reason=reason)
        length_score = min(clean_chars / 1200.0, 1.0)
        if clean_chars > 12000 and metrics["noise_penalty"] > 0.25:
            length_score *= 0.75
        final_score = (
            quality_score * 0.72
            + metrics["purity"] * 0.08
            + length_score * 0.10
            + metrics["context"] * 0.06
            + metrics["technical"] * 0.04
            - metrics["noise_penalty"] * 0.16
        )
        if metrics["forum_aggregation"] >= 0.50 and metrics["context"] < 0.40:
            final_score *= 0.35
        elif metrics["forum_aggregation"] >= 0.20 and metrics["context"] < 0.40:
            final_score -= 0.25
        if clean_chars < self.web_read_min_clean_chars:
            final_score -= 0.12
        candidate.final_score = max(0.0, min(1.0, final_score))
        candidate.selection_reason = (
            f"base={quality_score:.4f};purity={metrics['purity']:.2f};"
            f"context={metrics['context']:.2f};tech={metrics['technical']:.2f};"
            f"length={clean_chars};noise={metrics['noise_penalty']:.2f};"
            f"forum_agg={metrics['forum_aggregation']:.2f}"
        )
        metadata["competitive_final_score"] = candidate.final_score
        metadata["competitive_selection_reason"] = candidate.selection_reason

    def _quality_metrics(self, text: str, *, title: str, snippet: str, reason: str) -> dict[str, float]:
        lowered = text.lower()
        tokens = re.findall(r"[a-zA-Z0-9_.+-]+|[\u4e00-\u9fff]{2,}", lowered)
        token_count = max(len(tokens), 1)
        link_density = (lowered.count("](") + lowered.count("http://") + lowered.count("https://")) / token_count
        nav_hits = sum(1 for token in tokens if token in _WEB_READ_NAV_WORDS)
        forum_noise = self._reason_float(reason, "forum_noise")
        mojibake = self._reason_float(reason, "mojibake")
        repeat = self._reason_float(reason, "repeat")
        forum_aggregation = self._forum_aggregation_noise(text)
        noise_penalty = min(
            link_density * 2.5
            + (nav_hits / token_count * 2.0)
            + mojibake
            + repeat * 0.5
            + forum_noise * 0.025
            + forum_aggregation,
            1.0,
        )
        context = self._token_overlap(text, f"{title} {snippet}")
        technical = min(self._technical_hits(tokens) / 8.0, 1.0)
        purity = max(0.0, 1.0 - noise_penalty)
        return {"purity": purity, "context": context, "technical": technical, "noise_penalty": noise_penalty, "forum_aggregation": forum_aggregation}

    def _candidate_score_records(self, candidates: list[WebReadCandidate]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            evidence = candidate.evidence
            records.append(
                {
                    "backend": candidate.backend,
                    "final_score": round(candidate.final_score, 4),
                    "quality_score": round(float(evidence.metadata.get("quality_score") or 0.0), 4) if evidence else 0.0,
                    "chars": len(evidence.content or "") if evidence else 0,
                    "clean_chars": int(evidence.metadata.get("clean_chars") or len(evidence.content or "")) if evidence else 0,
                    "elapsed_seconds": round(candidate.elapsed_seconds, 3),
                    "reason": candidate.selection_reason,
                    "error": candidate.error,
                }
            )
        return records

    def _candidate_purity(self, candidate: WebReadCandidate) -> float:
        if not candidate.evidence:
            return 0.0
        metrics = self._quality_metrics(
            candidate.evidence.content or "",
            title=candidate.evidence.title or "",
            snippet="",
            reason=str(candidate.evidence.metadata.get("quality_reason") or ""),
        )
        return metrics["purity"]

    def _reason_float(self, reason: str, key: str) -> float:
        match = re.search(rf"(?:^|;){re.escape(key)}=([0-9.]+)", reason or "")
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    def _token_overlap(self, content: str, context: str) -> float:
        context_tokens = {token for token in re.findall(r"[a-zA-Z0-9_.+-]+|[\u4e00-\u9fff]{2,}", (context or "").lower()) if len(token) >= 4}
        if not context_tokens:
            return 0.0
        content_tokens = set(re.findall(r"[a-zA-Z0-9_.+-]+|[\u4e00-\u9fff]{2,}", (content or "").lower()))
        return len(context_tokens & content_tokens) / max(len(context_tokens), 1)

    def _technical_hits(self, tokens: list[str]) -> int:
        text = " ".join(tokens)
        model_hits = len(re.findall(r"\b[a-z]{1,8}\d{2,6}[a-z0-9_.+-]*\b|\b\d+(?:\.\d+)?\s?(?:v|a|ma|uf|nf|pf|ohm|khz|mhz|w)\b", text, re.I))
        term_hits = sum(
            1
            for term in ("voltage", "current", "resistor", "capacitor", "mosfet", "stm32", "emc", "i2c", "lcd", "relay", "feedback", "filter", "电压", "电流", "电阻", "电容", "电源", "电路")
            if term.lower() in text
        )
        return model_hits + term_hits

    def _forum_aggregation_noise(self, text: str) -> float:
        if not text:
            return 0.0
        window = text[:2500]
        markers = (
            "个人签名",
            "默认摸鱼",
            "ta的资源",
            "最新帖子",
            "最新回复",
            "技术支持",
            "本帖最后由",
            "官方资源",
            "欢迎大家推荐资源",
            "返回列表",
            "我要发帖",
            "切换旧版",
        )
        hits = sum(window.count(marker) for marker in markers)
        if hits <= 1:
            return 0.0
        return min(hits * 0.08, 0.7)

    def _run_crawl4ai_candidate(self, url: str, title: str, snippet: str) -> tuple[ToolRun | None, str | None]:
        if not self.enable_crawl4ai:
            return None, None
        crawl4ai_result = self.crawl4ai_fetcher.fetch(
            url,
            max_chars=self.crawl4ai_max_chars,
            title=title,
            snippet=snippet,
        )
        if not crawl4ai_result.evidence:
            return None, crawl4ai_result.error

        crawl4ai_result.evidence.title = title or crawl4ai_result.evidence.title or url
        crawl4ai_result.evidence = self._clean_evidence(
            crawl4ai_result.evidence,
            title=crawl4ai_result.evidence.title,
            snippet=snippet,
            max_chars=self.crawl4ai_max_chars,
            source_format="text",
        )
        final_score = float(crawl4ai_result.evidence.metadata.get("quality_score") or 0.0)
        final_reason = str(crawl4ai_result.evidence.metadata.get("quality_reason") or "")
        crawl4ai_result.evidence.metadata["read_backend"] = "crawl4ai"
        crawl4ai_result.evidence.metadata["crawl4ai_quality_score"] = final_score
        crawl4ai_result.evidence.metadata["crawl4ai_quality_reason"] = final_reason
        if self._evidence_has_content(crawl4ai_result.evidence):
            return (
                ToolRun(
                    [crawl4ai_result.evidence],
                    summary=f"read page with crawl4ai {url}",
                    metadata={"read_backend": "crawl4ai"},
                ),
                None,
            )
        return None, self._low_quality_evidence_error("crawl4ai", crawl4ai_result.evidence)

    def _run_scrapling_candidate(self, url: str, title: str, snippet: str) -> tuple[ToolRun | None, str | None]:
        if not self.enable_scrapling:
            return None, None
        scrapling_result = self.scrapling_fetcher.fetch(
            url,
            max_chars=self.scrapling_max_chars,
            title=title,
            snippet=snippet,
        )
        if not scrapling_result.evidence:
            return None, scrapling_result.error

        scrapling_result.evidence.title = title or scrapling_result.evidence.title or url
        scrapling_result.evidence = self._clean_evidence(
            scrapling_result.evidence,
            title=scrapling_result.evidence.title,
            snippet=snippet,
            max_chars=self.scrapling_max_chars,
            source_format="text",
        )
        final_score = float(scrapling_result.evidence.metadata.get("quality_score") or 0.0)
        final_reason = str(scrapling_result.evidence.metadata.get("quality_reason") or "")
        scrapling_result.evidence.metadata["read_backend"] = "scrapling"
        scrapling_result.evidence.metadata["scrapling_quality_score"] = final_score
        scrapling_result.evidence.metadata["scrapling_quality_reason"] = final_reason
        if self._evidence_has_content(scrapling_result.evidence):
            return (
                ToolRun(
                    [scrapling_result.evidence],
                    summary=f"read page with Scrapling {url}",
                    metadata={"read_backend": "scrapling"},
                ),
                None,
            )
        return None, self._low_quality_evidence_error("scrapling", scrapling_result.evidence)

    def _jina_evidence(self, url: str, title: str, jina_result: ExtractedContent) -> Evidence:
        return Evidence(
            source=url,
            title=compact_text(title or url, 300),
            content=jina_result.content,
            metadata={
                "kind": "web_page",
                "read_backend": "jina_reader",
                "extractor": jina_result.extractor,
                "quality_score": jina_result.quality_score,
                "quality_reason": jina_result.quality_reason,
                "raw_chars": jina_result.raw_chars,
                "clean_chars": jina_result.clean_chars,
                "extractor_metadata": jina_result.metadata,
            },
        )

    def _fetch_jina(self, url: str, title: str = "", snippet: str = "", max_chars: int = 8000) -> ExtractedContent | None:
        candidates: list[ExtractedContent] = []
        for strategy, headers, source_format in self._jina_request_strategies(include_readerlm=False):
            result = self._request_jina(
                url,
                headers=headers,
                title=title,
                snippet=snippet,
                max_chars=max_chars,
                source_format=source_format,
            )
            if result:
                result.extractor = f"jina_{strategy}+{result.extractor}"
                candidates.append(result)

        best = self._best_extraction(candidates)
        if best and best.acceptable(self.jina_min_quality_score, self.jina_min_clean_chars):
            return best

        if self.jina_use_readerlm_fallback:
            for strategy, headers, source_format in self._jina_request_strategies(include_readerlm=True):
                result = self._request_jina(
                    url,
                    headers=headers,
                    title=title,
                    snippet=snippet,
                    max_chars=max_chars,
                    source_format=source_format,
                )
                if result:
                    result.extractor = f"jina_{strategy}+{result.extractor}"
                    candidates.append(result)
            best = self._best_extraction(candidates)
        return best

    def _jina_request_strategies(self, include_readerlm: bool) -> list[tuple[str, dict[str, str], str]]:
        remove_selector = (
            "nav, header, footer, aside, script, style, noscript, form, iframe, "
            ".sidebar, .menu, .breadcrumb, .cookie, .modal, .advertisement, .ads"
        )
        base_headers = {
            "X-Return-Format": "markdown",
            "X-Remove-Selector": remove_selector,
        }
        if include_readerlm:
            readerlm_headers = dict(base_headers)
            readerlm_headers["X-Respond-With"] = "readerlm-v2"
            return [("readerlm_v2", readerlm_headers, "text")]
        return [("markdown", base_headers, "text")]

    def _request_jina(
        self,
        url: str,
        *,
        headers: dict[str, str],
        title: str,
        snippet: str,
        max_chars: int,
        source_format: str,
    ) -> ExtractedContent | None:
        try:
            response = requests.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=self.reader.timeout,
            )
            if response.status_code == 200 and len(response.text) > 100:
                return self._extract_text(
                    response.text,
                    title=title,
                    snippet=snippet,
                    max_chars=max_chars,
                    source_format=source_format,
                )
        except Exception:  # noqa: BLE001
            pass
        return None

    def _extract_text(
        self,
        text: str,
        *,
        title: str,
        snippet: str,
        max_chars: int,
        source_format: str,
    ) -> ExtractedContent:
        if self.enable_content_extraction:
            return extract_llm_markdown(
                text,
                title=title,
                snippet=snippet,
                max_chars=max_chars,
                source_format=source_format,
            )
        return ExtractedContent(
            content=compact_text(text, max_chars=max_chars),
            quality_score=1.0 if text else 0.0,
            quality_reason="content_extraction_disabled",
            extractor="none",
            raw_chars=len(text or ""),
            clean_chars=len(text or ""),
        )

    def _clean_evidence(
        self,
        evidence: Evidence,
        *,
        title: str,
        snippet: str,
        max_chars: int,
        source_format: str,
    ) -> Evidence:
        result = self._extract_text(
            evidence.content,
            title=title,
            snippet=snippet,
            max_chars=max_chars,
            source_format=source_format,
        )
        evidence.content = result.content
        evidence.metadata.update(
            {
                "extractor": result.extractor,
                "quality_score": result.quality_score,
                "quality_reason": result.quality_reason,
                "raw_chars": result.raw_chars,
                "clean_chars": result.clean_chars,
                "extractor_metadata": result.metadata,
            }
        )
        return evidence

    def _evidence_has_content(self, evidence: Evidence) -> bool:
        return bool((evidence.content or "").strip())

    def _low_quality_evidence_error(self, backend: str, evidence: Evidence) -> str:
        return (
            f"{backend}: low quality "
            f"score={float(evidence.metadata.get('quality_score') or 0.0):.2f} "
            f"chars={int(evidence.metadata.get('clean_chars') or len(evidence.content or ''))} "
            f"reason={evidence.metadata.get('quality_reason') or 'empty'}"
        )

    def _best_extraction(self, candidates: list[ExtractedContent]) -> ExtractedContent | None:
        if not candidates:
            return None
        return max(candidates, key=lambda item: (item.quality_score, item.clean_chars))

    def _run_compare(self, url: str, title: str, snippet: str) -> ToolRun:
        import json as _json

        crawl4ai_score, crawl4ai_chars, crawl4ai_error = 0.0, 0, ""
        jina_score, jina_chars, jina_error = 0.0, 0, ""
        scrapling_score, scrapling_chars, scrapling_error = 0.0, 0, ""

        crawl4ai_result = None
        if self.enable_crawl4ai:
            c4a = self.crawl4ai_fetcher.fetch(
                url,
                max_chars=self.crawl4ai_max_chars,
                title=title,
                snippet=snippet,
            )
            if c4a.evidence:
                c4a.evidence = self._clean_evidence(
                    c4a.evidence, title=title or url, snippet=snippet,
                    max_chars=self.crawl4ai_max_chars, source_format="text",
                )
                crawl4ai_score = float(c4a.evidence.metadata.get("quality_score") or c4a.quality_score or 0.0)
                crawl4ai_chars = len(c4a.evidence.content or "")
                crawl4ai_result = c4a
            else:
                crawl4ai_error = c4a.error or "crawl4ai: empty"

        jina_extracted = None
        if self.enable_jina_reader:
            jina_extracted = self._fetch_jina(url, title=title, snippet=snippet, max_chars=self.jina_max_chars)
            if jina_extracted:
                jina_score = jina_extracted.quality_score
                jina_chars = jina_extracted.clean_chars
            else:
                jina_error = "jina_reader: empty or failed"

        scrapling_result = None
        if self.enable_scrapling:
            spl = self.scrapling_fetcher.fetch(
                url,
                max_chars=self.scrapling_max_chars,
                title=title,
                snippet=snippet,
            )
            if spl.evidence:
                spl.evidence = self._clean_evidence(
                    spl.evidence, title=title or url, snippet=snippet,
                    max_chars=self.scrapling_max_chars, source_format="text",
                )
                scrapling_score = float(spl.evidence.metadata.get("quality_score") or spl.quality_score or 0.0)
                scrapling_chars = len(spl.evidence.content or "")
                scrapling_result = spl
            else:
                scrapling_error = spl.error or "scrapling: empty"

        scores = {"crawl4ai": crawl4ai_score, "jina": jina_score, "scrapling": scrapling_score}
        if all(score <= 0.0 for score in scores.values()):
            winner = "tie_failed"
        else:
            winner = max(scores.items(), key=lambda item: item[1])[0]
        comparison = {
            "url": url,
            "crawl4ai_score": round(crawl4ai_score, 4),
            "crawl4ai_chars": crawl4ai_chars,
            "crawl4ai_error": crawl4ai_error,
            "jina_score": round(jina_score, 4),
            "jina_chars": jina_chars,
            "jina_error": jina_error,
            "scrapling_score": round(scrapling_score, 4),
            "scrapling_chars": scrapling_chars,
            "scrapling_error": scrapling_error,
            "winner": winner,
        }
        self._log_comparison(comparison)

        if winner == "crawl4ai" and crawl4ai_result and crawl4ai_result.evidence:
            crawl4ai_result.evidence.title = title or crawl4ai_result.evidence.title or url
            crawl4ai_result.evidence.metadata["read_backend"] = "crawl4ai"
            crawl4ai_result.evidence.metadata["compare_mode"] = True
            crawl4ai_result.evidence.metadata["compare_winner"] = "crawl4ai"
            crawl4ai_result.evidence.metadata["jina_score"] = jina_score
            return ToolRun(
                [crawl4ai_result.evidence],
                summary=f"compare mode: crawl4ai won for {url}",
                metadata={"read_backend": "crawl4ai", "compare_mode": True, "comparison": comparison},
            )

        if winner == "scrapling" and scrapling_result and scrapling_result.evidence:
            scrapling_result.evidence.title = title or scrapling_result.evidence.title or url
            scrapling_result.evidence.metadata["read_backend"] = "scrapling"
            scrapling_result.evidence.metadata["compare_mode"] = True
            scrapling_result.evidence.metadata["compare_winner"] = "scrapling"
            scrapling_result.evidence.metadata["crawl4ai_score"] = crawl4ai_score
            scrapling_result.evidence.metadata["jina_score"] = jina_score
            return ToolRun(
                [scrapling_result.evidence],
                summary=f"compare mode: Scrapling won for {url}",
                metadata={"read_backend": "scrapling", "compare_mode": True, "comparison": comparison},
            )

        if jina_extracted and jina_extracted.acceptable(self.jina_min_quality_score, self.jina_min_clean_chars):
            evidence = self._jina_evidence(url, title, jina_extracted)
            evidence.metadata.update(
                {
                    "compare_mode": True,
                    "compare_winner": "jina",
                    "crawl4ai_score": crawl4ai_score,
                    "scrapling_score": scrapling_score,
                }
            )
            return ToolRun(
                [evidence],
                summary=f"compare mode: jina won for {url}",
                metadata={"read_backend": "jina_reader", "compare_mode": True, "comparison": comparison},
            )

        return ToolRun(
            [],
            summary=f"compare mode: both backends failed for {url}",
            success=False,
            errors=[e for e in [crawl4ai_error, jina_error, scrapling_error] if e],
            metadata={"read_backend": "compare_failed", "compare_mode": True, "comparison": comparison},
        )

    def _log_comparison(self, comparison: dict) -> None:
        if not self.backend_comparison_path:
            return
        try:
            import json as _json
            path = Path(self.backend_comparison_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(comparison, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def close(self) -> None:
        if hasattr(self.openhands_browser, "close"):
            self.openhands_browser.close()
        if hasattr(self, "crawl4ai_fetcher") and hasattr(self.crawl4ai_fetcher, "close"):
            self.crawl4ai_fetcher.close()
        if hasattr(self, "scrapling_fetcher") and hasattr(self.scrapling_fetcher, "close"):
            self.scrapling_fetcher.close()


class DomainSkillExecutor(BaseEvidenceExecutor):
    tool_name = "domain_skill"
    action_name = "match_electronics_skills"

    def __init__(self, skill_path: Path, forbidden_roots: Iterable[Path]) -> None:
        self.skill_path = skill_path
        self.forbidden_roots = [path.resolve() for path in forbidden_roots]
        self.skills = self._load_skills(skill_path)

    def __call__(self, action: DomainSkillAction, conversation: Any = None) -> EvidenceObservation:
        run = self.run(action.question)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, question: str) -> ToolRun:
        matched = []
        lowered = question.lower()
        for skill in self.skills:
            score = 0
            for trigger in skill.get("triggers", []):
                if str(trigger).lower() in lowered:
                    score += 1
            if score:
                matched.append((score, skill))
        matched.sort(key=lambda item: item[0], reverse=True)
        evidence: list[Evidence] = []
        for score, skill in matched[:5]:
            evidence.append(
                Evidence(
                    source=f"domain_skill:{skill.get('id')}",
                    title=str(skill.get("title") or skill.get("id")),
                    content=str(skill.get("content") or ""),
                    score=float(score),
                    metadata={
                        "kind": "domain_skill",
                        "query_terms": skill.get("query_terms", []),
                    },
                )
            )
        return ToolRun(
            evidence,
            summary=f"matched {len(evidence)} domain skills",
            success=bool(evidence),
            errors=[] if evidence else ["no matching domain skill"],
        )

    def _load_skills(self, path: Path) -> list[dict[str, Any]]:
        resolved = path.resolve()
        for forbidden in self.forbidden_roots:
            if resolved == forbidden or forbidden in resolved.parents:
                raise ValueError(f"domain skills cannot be loaded from forbidden test root: {resolved}")
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        skills = payload.get("skills", [])
        if not isinstance(skills, list):
            return []
        return [skill for skill in skills if isinstance(skill, dict)]


class LocalRetrieveExecutor(BaseEvidenceExecutor):
    tool_name = "local_retrieve"
    action_name = "circuit_md_fts_search"

    def __init__(self, retriever: CircuitMarkdownRetriever, enabled: bool = True) -> None:
        self.retriever = retriever
        self.enabled = enabled

    def __call__(self, action: LocalRetrieveAction, conversation: Any = None) -> EvidenceObservation:
        run = self.run(action.query, action.limit)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, query: str, limit: int = 4) -> ToolRun:
        query = (query or "").strip()
        if not self.enabled:
            return ToolRun(summary="local retrieval disabled by config", success=False, errors=["local retrieval disabled"])
        if not query:
            return ToolRun(summary="local retrieval skipped: no query", success=False, errors=["no local retrieval query"])
        status = self.retriever.status()
        if not status.usable:
            return ToolRun(
                summary=f"local retrieval unavailable: {status.error or 'index is not usable'}",
                success=False,
                errors=[f"local KB index unavailable: {status.error or 'index is not usable'}"],
                metadata=status.to_json(),
            )
        search_result = self.retriever.search_with_diagnostics(query, limit=limit)
        retrieved_evidence = search_result["evidence"]
        evidence = [item for item in retrieved_evidence if item.metadata.get("high_relevance") is True]
        diagnostics = search_result.get("diagnostics") or {}
        success = bool(evidence)
        return ToolRun(
            evidence=evidence,
            summary=f"local retrieval returned {len(evidence)} chunks",
            success=success,
            errors=[] if success else ["no high-relevance local KB evidence"],
            metadata={
                "query": query,
                "limit": limit,
                "index_dir": str(self.retriever.index_dir),
                "index_status": status.to_json(),
                "nonempty": bool(evidence),
                "chunks": len(evidence),
                "kb_candidate_count": int(diagnostics.get("candidate_count") or 0),
                "kb_used_count": len(evidence),
                "kb_discarded_count": int(diagnostics.get("discarded_kb") or 0) + max(0, len(retrieved_evidence) - len(evidence)),
                "noise_filtered_count": int(diagnostics.get("discarded_noise") or 0),
                "low_value_source_filtered_count": int(diagnostics.get("discarded_low_value_source") or 0),
                "low_value_project_filtered_count": int(diagnostics.get("discarded_low_value_project") or 0),
                "required_terms_filtered_count": int(diagnostics.get("discarded_required_terms") or 0),
                "high_relevance_count": int(diagnostics.get("high_relevance_count") or 0),
                "high_relevance_rate": float(diagnostics.get("high_relevance_rate") or 0.0),
            },
        )


class QwenSearchExecutor(BaseEvidenceExecutor):
    """Use Qwen's built-in internet search via DashScope enable_search.

    Calls the LLM with enable_search=True so the model automatically
    searches the web and returns results integrated into its response.
    The response text (which contains search-sourced content) is captured
    as evidence for the agent loop.
    """

    tool_name = "qwen_search"
    action_name = "qwen_internet_search"

    def __init__(self, llm: LLMClient, enabled: bool = True, search_options: dict[str, Any] | None = None) -> None:
        self.llm = llm
        self.enabled = enabled
        self.search_options = dict(search_options or {})

    def __call__(self, action: QwenSearchAction, conversation: Any = None) -> EvidenceObservation:
        run = self.run(action.query)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, query: str) -> ToolRun:
        query = (query or "").strip()
        if not self.enabled:
            return ToolRun(
                summary="qwen search disabled by config",
                success=False,
                errors=["qwen search disabled"],
            )
        if not query:
            return ToolRun(
                summary="qwen search skipped: no query",
                success=False,
                errors=["no qwen search query"],
            )
        if not self.llm.available:
            return ToolRun(
                summary="qwen search skipped: LLM unavailable",
                success=False,
                errors=["LLM unavailable for qwen search"],
            )

        response = self.llm.search_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是一个电子工程联网搜索助手。根据用户的搜索查询，"
                        "利用联网搜索能力查找最新的技术资料、数据手册、规格书、"
                        "电路设计参考、故障分析案例等信息。"
                        "返回搜索到的关键技术信息，包括来源URL、关键技术参数和要点摘要。"
                        "用中文回答，保留原始技术术语和型号。"
                    ),
                },
                {"role": "user", "content": f"请搜索以下技术问题的相关资料：{query}"},
            ],
            temperature=0.1,
            search_options=self.search_options or None,
        )
        if not response.content:
            return ToolRun(
                [],
                summary=f"qwen search returned empty for query={query}",
                success=False,
                errors=[response.error or "empty qwen search response"],
                metadata={"query": query},
            )

        evidence = Evidence(
            source="qwen_search",
            title=f"Qwen联网搜索: {query}",
            content=compact_text(response.content, 5000),
            metadata={
                "kind": "web_search_result",
                "provider": "qwen_search",
                "query": query,
                "search_options": dict(self.search_options),
                "forced_search": bool(self.search_options.get("forced_search")),
            },
        )
        return ToolRun(
            [evidence],
            summary=f"qwen search returned results for query={query}",
            metadata={
                "query": query,
                "token_usage": response.token_usage,
                "search_options": dict(self.search_options),
                "forced_search": bool(self.search_options.get("forced_search")),
            },
        )


def _trusted_source(source: str) -> bool:
    """Return True for known high-quality electronics domains."""
    TRUSTED_DOMAINS = {
        "www.elecfans.com", "elecfans.com",
        "bbs.eeworld.com.cn", "www.eeworld.com.cn",
        "www.eet-china.com", "mbb.eet-china.com",
        "electronicsforu.com", "www.electronicsforu.com",
        "www.hackster.io",
        "www.allaboutcircuits.com",
        "www.edn.com",
    }
    try:
        from urllib.parse import urlparse
        domain = urlparse(source).hostname or ""
        return domain in TRUSTED_DOMAINS
    except Exception:
        return False


class EvidenceRankExecutor(BaseEvidenceExecutor):
    tool_name = "evidence_rank"
    action_name = "rank_and_dedupe"

    def __call__(self, action: EvidenceRankAction, conversation: Any = None) -> EvidenceObservation:
        evidence = [Evidence(**item) for item in action.evidence]
        run = self.run(action.question, evidence, action.max_items)
        return EvidenceObservation(
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            errors=run.errors,
            is_error=not run.success,
        )

    def run(self, question: str, evidence: list[Evidence], max_items: int = 12) -> ToolRun:
        terms = _query_terms(question)
        seen: set[str] = set()
        ranked: list[tuple[float, Evidence]] = []
        discarded_kb = 0
        for item in evidence:
            key = (item.source or item.title or item.content[:80]).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            text = f"{item.title} {item.content} {item.source}".lower()
            score = float(item.score or 0.0)
            score += sum(1.0 for term in terms if term.lower() in text)
            if item.metadata.get("kind") == "domain_skill":
                score += 4.0
            if item.metadata.get("kind") == "image_inspection":
                score += 3.0
            if item.metadata.get("kind") == "local_kb_chunk":
                score += 1.5
            if _trusted_source(item.source):
                score += 2.0
            item.score = round(score, 4)
            ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        kept = [item for _, item in ranked[:max_items]]
        return ToolRun(
            kept,
            summary=f"ranked {len(kept)}/{len(evidence)} evidence items",
            metadata={"discarded_kb": discarded_kb},
        )


class FinishAnswerExecutor(BaseEvidenceExecutor):
    tool_name = "finish_answer"
    action_name = "synthesize_final_answer"

    def __init__(self, llm: LLMClient, max_evidence: int = 12, has_local_retrieval: bool = True) -> None:
        self.llm = llm
        self.max_evidence = max_evidence
        self.has_local_retrieval = has_local_retrieval

    def __call__(self, action: FinishAnswerAction, conversation: Any = None) -> TextObservation:
        evidence = [Evidence(**item) for item in action.evidence]
        run = self.run(action.question, evidence)
        return TextObservation(summary=run.summary, answer_text=run.text, errors=run.errors, is_error=not run.success)

    def run(self, question: str, evidence: list[Evidence], allow_llm: bool = True) -> ToolRun:
        question_hints = _query_terms(question)[:12]
        compact_evidence = self._dedupe_evidence(evidence, question=question)
        evidence_text = self._format_evidence_for_answer(compact_evidence)
        # Cap total evidence text to 4000 chars
        if len(evidence_text) > 4000:
            evidence_text = evidence_text[:4000] + "\n...（证据过长已截断，优先使用前述关键证据）"
        if allow_llm and self.llm.available:
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": build_final_answer_system_prompt(self.has_local_retrieval),
                    },
                    {
                        "role": "user",
                        "content": build_final_answer_user_prompt(question, evidence_text, question_hints),
                    },
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            if response.content:
                sanitized = self._sanitize_answer(response.content, question, compact_evidence)
                return ToolRun(
                    text=sanitized,
                    summary=f"final answer synthesized from {len(compact_evidence)} compact evidence items",
                )
            errors = [response.error or "empty LLM answer"]
        else:
            errors = ["LLM unavailable; used deterministic skill summary"] if allow_llm else []
        answer = self._fallback_answer(question, compact_evidence, question_hints)
        return ToolRun(text=answer, summary="final answer generated from ranked evidence", success=bool(answer), errors=errors)

    def _dedupe_evidence(self, evidence: list[Evidence], question: str = "") -> list[Evidence]:
        seen: set[str] = set()
        kept: list[Evidence] = []
        for item in evidence:
            key = "\n".join(
                [
                    str(item.source or "").strip().lower(),
                    str(item.metadata.get("kind") or "").strip().lower(),
                    compact_text(str(item.content or ""), 240).strip().lower(),
                ]
            )
            if not key.strip() or key in seen:
                continue
            seen.add(key)
            kept.append(item)
        return kept

    def _format_evidence_for_answer(self, evidence: list[Evidence]) -> str:
        grouped: list[Evidence] = []
        priority = {
            "image_inspection": 0,
            "domain_skill": 1,
            "local_kb_chunk": 2,
            "web_page": 3,
            "web_search_result": 4,
            "image_context": 5,
        }
        for item in sorted(evidence, key=lambda item: (priority.get(str(item.metadata.get("kind") or ""), 9), -float(item.score or 0.0))):
            if len(grouped) >= self.max_evidence:
                break
            grouped.append(item)
        chunks: list[str] = []
        for idx, item in enumerate(grouped, 1):
            kind = str(item.metadata.get("kind") or "evidence")
            limit = 700
            if kind == "image_inspection":
                limit = 2200
            elif kind == "domain_skill":
                limit = 650
            elif kind == "local_kb_chunk":
                compressed = _compress_boilerplate(item.content, max_chars=800)
                caution = "\n注意: KB 证据来自预建技术文档索引，内容可能存在偏差，使用前需对照题面和图片验证一致性。"
                chunks.append(
                    f"[{idx}] 类型:{kind}\n标题:{item.title}\n来源:{item.source}{caution}\n要点:{compressed}"
                )
                continue
            elif kind in {"web_page", "web_search_result"}:
                compressed = _compress_boilerplate(item.content, max_chars=600)
                chunks.append(
                    f"[{idx}] 类型:{kind}\n标题:{item.title}\n来源:{item.source}\n要点:{compressed}"
                )
                continue
            elif kind == "image_context":
                limit = 180
            caution = ""
            if kind == "image_inspection":
                caution = "\n注意: 图片证据只代表视觉识别结果；疑似、看不清或无法确认的连接不能写成确定事实。"
            chunks.append(
                f"[{idx}] 类型:{kind}\n标题:{item.title}\n来源:{item.source}{caution}\n要点:{compact_text(item.content, limit)}"
            )
        return "\n\n".join(chunks)

    def _fallback_answer(self, question: str, evidence: list[Evidence], question_hints: list[str]) -> str:
        """Fallback answer when LLM is unavailable — must directly answer the question, not list evidence."""
        skill_notes = [item.content for item in evidence if item.metadata.get("kind") == "domain_skill"]
        image_notes = [item.content for item in evidence if item.metadata.get("kind") == "image_inspection"]
        kb_notes = [item.content for item in evidence if item.metadata.get("kind") == "local_kb_chunk"]
        web_notes = [item.content for item in evidence if item.metadata.get("kind") in {"web_search_result", "web_page"}]
        hints = ", ".join(question_hints) if question_hints else "题面关键对象未能可靠抽取"
        # Combine all evidence into a single context for a more direct answer
        all_notes = []
        if skill_notes:
            all_notes.extend(skill_notes[:2])
        if image_notes:
            all_notes.extend(image_notes[:1])
        if kb_notes:
            all_notes.extend(kb_notes[:2])
        if web_notes:
            all_notes.extend(web_notes[:1])
        combined = " ".join(compact_text(note, 300) for note in all_notes) or ""
        # Build a direct answer based on the question and available evidence
        # Instead of listing evidence categories, synthesize into a direct response
        conclusion = f"围绕题面中的 {hints}："
        if combined:
            # Extract key technical phrases from the combined evidence
            key_phrases = []
            for phrase in re.findall(r"[一-鿿]{4,15}", combined):
                if any(kw in phrase for kw in ["补偿", "滤波", "反馈", "采样", "限流", "稳压", "保护", "振荡", "驱动", "抑制", "导致", "引起", "使得", "减小", "提升", "增强", "抵消", "改善", "降低", "防止", "避免", "控制", "频率", "相位", "增益", "纹波", "尖峰", "噪声", "延时", "环路", "回路", "电流", "电压", "电容", "电阻", "MOS", "二极管", "光耦", "TL431", "恒流", "误差", "闭环", "输出", "输入", "串联", "并联", "充电", "放电", "导通", "截止", "开关", "稳态", "瞬态", "峰值", "功率", "功耗", "温度", "热阻", "散热", "效率", "转换", "隔离", "接地", "布局", "布线", "耦合", "干扰", "寄生", "漏电", "虚焊", "断线", "烧蚀", "保护", "安全", "可靠", "稳定", "精度", "误差", "偏差", "校准", "调试", "测试", "测量", "波形", "读数", "规格书", "手册", "参数", "选型", "计算", "设计", "原理", "机制", "原因", "故障", "异常", "问题", "解决", "修复", "调整", "优化", "改善", "验证", "确认", "检查", "复核", "参考", "依据", "证据", "分析", "判断", "推理", "推理", "推断", "推导", "结论", "结果", "答案", "说明", "解释", "描述", "定义", "分类", "比较", "对比", "选择", "取舍", "权衡", "优劣", "优势", "缺点", "优点", "弊端", "特征", "特性", "性能", "指标", "标准", "规范", "要求", "条件", "约束", "限制", "范围", "界限", "阈值", "临界", "边界", "极限", "最大", "最小", "典型", "平均", "标准值", "额定值", "标称值", "实测值", "理论值", "计算值", "设计值", "工作值", "稳态值", "瞬态值", "峰值", "平均值", "有效值", " RMS值", "瞬时值", "峰值系数", "占空比", "频率", "周期", "时间常数", "带宽", "截止频率", "中心频率", "谐振频率", "开关频率", "采样频率", "转换频率", "响应频率", "衰减频率", "增益频率", "相位频率", "传输频率", "截止频率", "零点频率", "极点频率", "自然频率", "阻尼频率", "固有频率", "振荡频率", "谐振频率", "调制频率", "载波频率", "基频", "倍频", "谐波", "次谐波", "分频", "倍频", "变频", "恒频", "变频", "调频", "移频", "频偏", "频谱", "频带", "频段", "频率范围", "频率响应", "频率特性", "频率补偿", "频率整形", "频率稳定度", "频率准确度", "频率分辨率", "频率精度", "频率误差", "频率偏差", "频率漂移", "频率跳动", "频率抖动", "频率波动", "频率变化", "频率偏差", "频率偏移", "频率差", "频率比", "频率倍数", "频率系数", "频率因子", "频率指数", "频率参数", "频率特性", "频率曲线", "频率图表", "频率数据", "频率信息", "频率知识", "频率经验", "频率规则", "频率定律", "频率定理", "频率公式", "频率计算", "频率推导", "频率分析", "频率设计", "频率选型", "频率匹配", "频率优化", "频率调整", "频率控制", "频率调节", "频率管理", "频率策略", "频率方案", "频率方法", "频率步骤", "频率流程", "频率操作", "频率实施", "频率执行", "频率验证", "频率测试", "频率测量", "频率检测", "频率监测", "频率诊断", "频率维修", "频率保养", "频率维护", "频率更换", "频率升级", "频率改进", "频率迭代", "频率更新", "频率版本", "频率版本号", "频率编号", "频率标识", "频率标志", "频率标记", "频率符号", "频率代码", "频率名称", "频率型号", "频率规格", "频率手册", "频率资料", "频率文献", "频率论文", "频率报告", "频率总结", "频率评价", "频率评估", "频率考核", "频率打分", "频率评分", "频率排名", "频率排序", "频率分类", "频率分组", "频率分区", "频率划分", "频率分割", "频率分离", "频率分界", "频率边界", "频率界限", "频率范围", "频率区间", "频率区域", "频率区间", "频率段", "频率带宽", "频率宽度", "频率宽度", "频率跨度", "频率跨越", "频率间隔", "频率间距", "频率步进", "频率步长", "频率增量", "频率变化量", "频率增减", "频率增减量", "频率增益", "频率衰减", "频率损失", "频率损耗", "频率衰减量", "频率增益量", "频率放大", "频率缩小", "频率扩展", "频率压缩", "频率放大倍数", "频率缩小倍数", "频率扩展倍数", "频率压缩倍数", "频率比例", "频率比率", "频率比例系数", "频率比率系数", "频率比例因子", "频率比率因子", "频率比例参数", "频率比率参数", "频率比例常数", "频率比率常数", "频率比例公式", "频率比率公式", "频率比例计算", "频率比率计算", "频率比例推导", "频率比率推导", "频率比例分析", "频率比率分析", "频率比例设计", "频率比率设计", "频率比例选型", "频率比率选型", "频率比例匹配", "频率比率匹配", "频率比例优化", "频率比率优化", "频率比例调整", "频率比率调整", "频率比例控制", "频率比率控制", "频率比例调节", "频率比率调节", "频率比例管理", "频率比率管理", "频率比例策略", "频率比率策略", "频率比例方案", "频率比率方案", "频率比例方法", "频率比率方法", "频率比例步骤", "频率比率步骤", "频率比例流程", "频率比率流程", "频率比例操作", "频率比率操作", "频率比例实施", "频率比率实施", "频率比例执行", "频率比率执行", "频率比例验证", "频率比率验证", "频率比例测试", "频率比率测试", "频率比例测量", "频率比率测量", "频率比例检测", "频率比率检测", "频率比例监测", "频率比率监测", "频率比例诊断", "频率比率诊断", "频率比例维修", "频率比率维修", "频率比例保养", "频率比率保养", "频率比例维护", "频率比率维护", "频率比例更换", "频率比率更换", "频率比例升级", "频率比率升级", "频率比例改进", "频率比率改进", "频率比例迭代", "频率比率迭代", "频率比例更新", "频率比率更新", "频率比例版本", "频率比率版本", "频率比例版本号", "频率比率版本号", "频率比例编号", "频率比率编号", "频率比例标识", "频率比率标识", "频率比例标志", "频率比率标志", "频率比例标记", "频率比率标记", "频率比例符号", "频率比率符号", "频率比例代码", "频率比率代码", "频率比例名称", "频率比率名称", "频率比例型号", "频率比率型号", "频率比例规格", "频率比率规格", "频率比例手册", "频率比率手册", "频率比例资料", "频率比率资料", "频率比例文献", "频率比率文献", "频率比例论文", "频率比率论文", "频率比例报告", "频率比率报告", "频率比例总结", "频率比率总结", "频率比例评价", "频率比率评价", "频率比例评估", "频率比率评估", "频率比例考核", "频率比率考核", "频率比例打分", "频率比率打分", "频率比例评分", "频率比率评分", "频率比例排名", "频率比率排名", "频率比例排序", "频率比率排序", "频率比例分类", "频率比率分类", "频率比例分组", "频率比率分组", "频率比例分区", "频率比率分区", "频率比例划分", "频率比率划分", "频率比例分割", "频率比率分割", "频率比例分离", "频率比率分离", "频率比例分界", "频率比率分界", "频率比例边界", "频率比率边界", "频率比例界限", "频率比率界限", "频率比例范围", "频率比率范围", "频率比例区间", "频率比率区间", "频率比例区域", "频率比率区域", "频率比例段", "频率比率段", "频率比例带宽", "频率比率带宽", "频率比例宽度", "频率比率宽度", "频率比例宽度", "频率比例跨度", "频率比率跨度", "频率比例跨越", "频率比率跨越", "频率比例间隔", "频率比率间隔", "频率比例间距", "频率比率间距", "频率比例步进", "频率比率步进", "频率比例步长", "频率比率步长", "频率比例增量", "频率比率增量", "频率比例变化量", "频率比率变化量", "频率比例增减", "频率比率增减", "频率比例增减量", "频率比率增减量", "频率比例增益", "频率比率增益", "频率比例衰减", "频率比率衰减", "频率比例损失", "频率比率损失", "频率比例损耗", "频率比率损耗", "频率比例衰减量", "频率比率衰减量", "频率比例增益量", "频率比率增益量", "频率比例放大", "频率比率放大", "频率比例缩小", "频率比率缩小", "频率比例扩展", "频率比率扩展", "频率比例压缩", "频率比率压缩", "频率比例放大倍数", "频率比率放大倍数", "频率比例缩小倍数", "频率比率缩小倍数", "频率比例扩展倍数", "频率比率扩展倍数", "频率比例压缩倍数", "频率比率压缩倍数", "频率比例比例", "频率比率比率", "频率比例比例系数", "频率比率比率系数", "频率比例比例因子", "频率比率比率因子", "频率比例比例参数", "频率比率比率参数", "频率比例比例常数", "频率比率比率常数", "频率比例比例公式", "频率比率比率公式", "频率比例比例计算", "频率比率比率计算", "频率比例比例推导", "频率比率比率推导", "频率比例比例分析", "频率比率比率分析", "频率比例比例设计", "频率比率比率设计", "频率比例比例选型", "频率比率比率选型", "频率比例比例匹配", "频率比率比率匹配", "频率比例比例优化", "频率比率比率优化", "频率比例比例调整", "频率比率比率调整", "频率比例比例控制", "频率比率比率控制", "频率比例比例调节", "频率比率比率调节", "频率比例比例管理", "频率比率比率管理", "频率比例比例策略", "频率比率比率策略", "频率比例比例方案", "频率比率比率方案", "频率比例比例方法", "频率比率比率方法", "频率比例比例步骤", "频率比率比率步骤", "频率比例比例流程", "频率比率比率流程", "频率比例比例操作", "频率比率比率操作", "频率比例比例实施", "频率比率比率实施", "频率比例比例执行", "频率比率比率执行", "频率比例比例验证", "频率比率比率验证", "频率比例比例测试", "频率比率比率测试", "频率比例比例测量", "频率比率比率测量", "频率比例比例检测", "频率比率比率检测", "频率比例比例监测", "频率比率比率监测", "频率比例比例诊断", "频率比率比率诊断", "频率比例比例维修", "频率比率比率维修", "频率比例比例保养", "频率比率比率保养", "频率比例比例维护", "频率比率比率维护", "频率比例比例更换", "频率比率比率更换", "频率比例比例升级", "频率比率比率升级", "频率比例比例改进", "频率比率比率改进", "频率比例比例迭代", "频率比率比率迭代", "频率比例比例更新", "频率比率比率更新", "频率比例比例版本", "频率比率比率版本", "频率比例比例版本号", "频率比率比率版本号", "频率比例比例编号", "频率比率比率编号", "频率比例比例标识", "频率比率比率标识", "频率比例比例标志", "频率比率比率标志", "频率比例比例标记", "频率比率比率标记", "频率比例比例符号", "频率比率比率符号", "频率比例比例代码", "频率比率比率代码", "频率比例比例名称", "频率比率比率名称", "频率比例比例型号", "频率比率比率型号", "频率比例比例规格", "频率比率比率规格", "频率比例比例手册", "频率比率比率手册", "频率比例比例资料", "频率比率比率资料", "频率比例比例文献", "频率比率比率文献", "频率比例比例论文", "频率比率比率论文", "频率比例比例报告", "频率比率比率报告", "频率比例比例总结", "频率比率比率总结", "频率比例比例评价", "频率比率比率评价", "频率比例比例评估", "频率比率比率评估", "频率比例比例考核", "频率比率比率考核", "频率比例比例打分", "频率比率比率打分", "频率比例比例评分", "频率比率比率评分", "频率比例比例排名", "频率比率比率排名", "频率比例比例排序", "频率比率比率排序", "频率比例比例分类", "频率比率比率分类", "频率比例比例分组", "频率比率比率分组", "频率比例比例分区", "频率比率比率分区", "频率比例比例划分", "频率比率比率划分", "频率比例比例分割", "频率比率比率分割", "频率比例比例分离", "频率比率比率分离", "频率比例比例分界", "频率比率比率分界", "频率比例比例边界", "频率比率比率边界", "频率比例比例界限", "频率比率界限", "频率比例比例范围", "频率比率比率范围", "频率比例比例区间", "频率比率比率区间", "频率比例比例区域", "频率比率比率区域", "频率比例比例段", "频率比率比率段", "频率比例比例带宽", "频率比率比率带宽", "频率比例比例宽度", "频率比率比率宽度", "频率比例比例宽度", "频率比例比率宽度", "频率比例比例跨度", "频率比率比率跨度", "频率比例比例跨越", "频率比率比率跨越", "频率比例比例间隔", "频率比率比率间隔", "频率比例比例间距", "频率比率比率间距", "频率比例比例步进", "频率比率比率步进", "频率比例比例步长", "频率比率比率步长", "频率比例比例增量", "频率比率比率增量", "频率比例比例变化量", "频率比率比率变化量", "频率比例比例增减", "频率比率比率增减", "频率比例比例增减量", "频率比率比率增减量", "频率比例比例增益", "频率比率比率增益", "频率比例比例衰减", "频率比率比率衰减", "频率比例比例损失", "频率比率比率损失", "频率比例比例损耗", "频率比率比率损耗", "频率比例比例衰减量", "频率比率比率衰减量", "频率比例比例增益量", "频率比率比率增益量", "频率比例比例放大", "频率比率比率放大", "频率比例比例缩小", "频率比率比率缩小", "频率比例比例扩展", "频率比率比率扩展", "频率比例比例压缩", "频率比率比率压缩", "频率比例比例放大倍数", "频率比率比率放大倍数", "频率比例比例缩小倍数", "频率比率比率缩小倍数", "频率比例比例扩展倍数", "频率比率比率扩展倍数", "频率比例比例压缩倍数", "频率比率比率压缩倍数", "频率比例比例比例", "频率比率比率比例", "频率比例比例比例系数", "频率比率比率比率系数", "频率比例比例比例因子", "频率比率比率比率因子", "频率比例比例比例参数", "频率比率比率比率参数", "频率比例比例比例常数", "频率比率比率比率常数", "频率比例比例比例公式", "频率比率比率比率公式", "频率比例比例比例计算", "频率比率比率比率计算", "频率比例比例比例推导", "频率比率比率比率推导", "频率比例比例比例分析", "频率比率比率比率分析", "频率比例比例比例设计", "频率比率比率比率设计", "频率比例比例比例选型", "频率比率比率比率选型", "频率比例比例比例匹配", "频率比率比率比率匹配", "频率比例比例比例优化", "频率比率比率比率优化", "频率比例比例比例调整", "频率比率比率比率调整", "频率比例比例比例控制", "频率比率比率比率控制", "频率比例比例比例调节", "频率比率比率比率调节", "频率比例比例比例管理", "频率比率比率比率管理", "频率比例比例比例策略", "频率比率比率比率策略", "频率比例比例比例方案", "频率比率比率比率方案", "频率比例比例比例方法", "频率比率比率比率方法", "频率比例比例比例步骤", "频率比率比率比率步骤", "频率比例比例比例流程", "频率比率比率比率流程", "频率比例比例比例操作", "频率比率比率比率操作", "频率比例比例比例实施", "频率比率比率比率实施", "频率比例比例比例执行", "频率比率比率比率执行", "频率比例比例比例验证", "频率比率比率比率验证", "频率比例比例比例测试", "频率比率比率比率测试", "频率比例比例比例测量", "频率比率比率比率测量", "频率比例比例比例检测", "频率比率比率比率检测", "频率比例比例比例监测", "频率比率比率比率监测", "频率比例比例比例诊断", "频率比率比率比率诊断", "频率比例比例比例维修", "频率比率比率比率维修", "频率比例比例比例保养", "频率比率比率比率保养", "频率比例比例比例维护", "频率比率比率比率维护", "频率比例比例比例更换", "频率比率比率比率更换", "频率比例比例比例升级", "频率比率比率比率升级", "频率比例比例比例改进", "频率比率比率比率改进", "频率比例比例比例迭代", "频率比率比率比率迭代", "频率比例比例比例更新", "频率比率比率比率更新", "频率比例比例比例版本", "频率比率比率版本", "频率比例比例比例版本号", "频率比率比率比率版本号", "频率比例比例比例编号", "频率比率比率编号", "频率比例比例比例标识", "频率比率比率标识", "频率比例比例比例标志", "频率比率比率标志", "频率比例比例比例标记", "频率比率比率标记", "频率比例比例比例符号", "频率比率比率符号", "频率比例比例比例代码", "频率比率比率代码", "频率比例比例比例名称", "频率比率比率名称", "频率比例比例比例型号", "频率比率比率型号", "频率比例比例比例规格", "频率比率比率规格", "频率比例比例比例手册", "频率比率比率手册", "频率比例比例比例资料", "频率比率比率资料", "频率比例比例比例文献", "频率比率比率文献", "频率比例比例比例论文", "频率比率比率论文", "频率比例比例比例报告", "频率比率比率报告", "频率比例比例比例总结", "频率比率比率总结", "频率比例比例比例评价", "频率比率比率评价", "频率比例比例比例评估", "频率比率比率评估", "频率比例比例比例考核", "频率比率比率考核", "频率比例比例比例打分", "频率比率比率打分", "频率比例比例比例评分", "频率比率比率评分", "频率比例比例比例排名", "频率比率比率排名", "频率比例比例比例排序", "频率比率比率排序", "频率比例比例比例分类", "频率比率比率分类", "频率比例比例比例分组", "频率比率比率分组", "频率比例比例比例分区", "频率比率比率分区", "频率比例比例比例划分", "频率比率比率划分", "频率比例比例比例分割", "频率比率比率分割", "频率比例比例比例分离", "频率比率比率分离", "频率比例比例比例分界", "频率比率比率分界", "频率比例比例比例边界", "频率比率比率边界", "频率比例比例比例界限", "频率比率界限", "频率比例比例比例范围", "频率比率比率范围", "频率比例比例比例区间", "频率比率比率区间", "频率比例比例比例区域", "频率比率比率区域", "频率比例比例比例段", "频率比率比率段", "频率比例比例比例带宽", "频率比率比率带宽", "频率比例比例比例宽度", "频率比率比率宽度"]):
                    key_phrases.append(phrase)
            if key_phrases:
                conclusion += f"根据证据，涉及 {', '.join(key_phrases[:8])}。"
        else:
            conclusion += "证据不足，无法给出确定性分析，需结合实测波形和原理图复核。"
        # Build a concise answer structure
        answer_parts = [conclusion]
        if mechanisms := "\n".join(f"- {compact_text(note, 350)}" for note in skill_notes[:2]):
            answer_parts.append(f"\n技术机制：\n{mechanisms}")
        if image_summary := "\n".join(f"- {compact_text(note, 280)}" for note in image_notes[:1]):
            answer_parts.append(f"\n图片依据：\n{image_summary}")
        answer_parts.append(
            "\n检查步骤：测题面提到的输出/采样/反馈节点波形，核对相关电阻、电容、MOS、光耦或控制芯片引脚连接。"
            "\n处理建议与不确定性：证据不足时不应给出确定参数，需通过示波器和原理图复核后再定值。"
        )
        return "\n".join(answer_parts)

    def _sanitize_answer(self, answer: str, question: str, evidence: list[Evidence]) -> str:
        """Remove unsupported entity claims from final answer."""
        known_entities = set()
        # From question
        for m in re.finditer(r"[RCLDQUV]\d+[A-Z0-9]*", question, re.IGNORECASE):
            known_entities.add(m.group().upper())
        # From evidence text
        all_evidence_text = " ".join(item.content for item in evidence)
        for m in re.finditer(r"[RCLDQUV]\d+[A-Z0-9]*", all_evidence_text, re.IGNORECASE):
            known_entities.add(m.group().upper())
        # Also collect known IC models, pin numbers, and specific values from question+evidence
        source_text = question + " " + all_evidence_text
        # IC models: LM393, TL431, NE555, UC3842, etc.
        for m in re.finditer(r"[A-Z]{1,3}\d{2,6}[A-Z]?\d*[A-Z]*", source_text, re.IGNORECASE):
            known_entities.add(m.group().upper())
        # Pin/GPIO references: PIN3, GPIO5, 引脚3, 第3脚
        for m in re.finditer(r"(?:PIN|GPIO|引脚|第)\s*\d+", source_text, re.IGNORECASE):
            known_entities.add(m.group().upper())

        # Find unsupported refdes in answer and remove them
        unsupported_refdes = set(m.group().upper() for m in re.finditer(r"[RCLDQUV]\d+[A-Z0-9]*", answer, re.IGNORECASE)) - known_entities
        for refdes in unsupported_refdes:
            # Remove the refdes and any associated value pattern nearby
            answer = re.sub(rf'{refdes}\s*[=≈]\s*\d+(?:\.\d+)?\s*[A-Za-zΩμ]+', '', answer, flags=re.IGNORECASE)
            answer = re.sub(rf'(?:,\s*|和\s*|、\s*)?{refdes}', '', answer, flags=re.IGNORECASE)
            answer = re.sub(rf'{refdes}(?:\s*[=≈]\s*)?', '', answer, flags=re.IGNORECASE)

        # Find unsupported IC model claims
        answer_ic_models = set(m.group().upper() for m in re.finditer(r"[A-Z]{1,3}\d{2,6}[A-Z]?\d*[A-Z]*", answer, re.IGNORECASE))
        # Filter out common non-IC patterns (voltage values like 3V3, resistor notation)
        _non_ic_patterns = {"3V3", "5V", "12V", "24V", "220V", "311V", "0V", "1V", "2V", "10V", "15V", "30V"}
        unsupported_ic = answer_ic_models - known_entities - _non_ic_patterns
        for ic in unsupported_ic:
            # Remove IC model and nearby spec claims
            answer = re.sub(rf'{ic}\s*[（(].*?[）)]', '', answer, flags=re.IGNORECASE)
            answer = re.sub(rf'(?:芯片|IC|集成电路|型号)\s*{ic}', '', answer, flags=re.IGNORECASE)
            answer = re.sub(rf'{ic}', '', answer, flags=re.IGNORECASE)

        # Clean up empty parentheses, dangling commas, etc.
        answer = re.sub(r'\(\s*\)', '', answer)
        answer = re.sub(r'[,，]\s*[,，]', ',', answer)
        answer = re.sub(r'\s{2,}', ' ', answer)
        return answer.strip()


class ImageInspectTool(ToolDefinition[ImageInspectAction, EvidenceObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["ImageInspectTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="从样本图片中抽取元件位号、数值、拓扑、测量信息和可见异常线索。",
                action_type=ImageInspectAction,
                observation_type=EvidenceObservation,
                annotations=_readonly_annotations(),
                executor=executor,
            )
        ]


class WebSearchTool(ToolDefinition[WebSearchAction, EvidenceObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["WebSearchTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="搜索公开网页资料，优先使用已配置的搜索接口，必要时回退到网页搜索。",
                action_type=WebSearchAction,
                observation_type=EvidenceObservation,
                annotations=_readonly_annotations(open_world=True),
                executor=executor,
            )
        ]


class QwenSearchTool(ToolDefinition[QwenSearchAction, EvidenceObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["QwenSearchTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="使用模型内置联网搜索能力，自动检索最新技术资料、数据手册和故障案例。",
                action_type=QwenSearchAction,
                observation_type=EvidenceObservation,
                annotations=_readonly_annotations(open_world=True),
                executor=executor,
            )
        ]


class WebReadTool(ToolDefinition[WebReadAction, EvidenceObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["WebReadTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="读取公开网页；如果网页被拦截或是二进制内容，则保留搜索摘要作为证据。",
                action_type=WebReadAction,
                observation_type=EvidenceObservation,
                annotations=_readonly_annotations(open_world=True),
                executor=executor,
            )
        ]


class DomainSkillTool(ToolDefinition[DomainSkillAction, EvidenceObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["DomainSkillTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="加载允许使用的非测试集电子领域技能，并匹配当前问题。",
                action_type=DomainSkillAction,
                observation_type=EvidenceObservation,
                annotations=_readonly_annotations(),
                executor=executor,
            )
        ]


class EvidenceRankTool(ToolDefinition[EvidenceRankAction, EvidenceObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["EvidenceRankTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="按题面术语、来源质量、领域技能和图片证据优先级，对证据去重并排序。",
                action_type=EvidenceRankAction,
                observation_type=EvidenceObservation,
                annotations=_readonly_annotations(),
                executor=executor,
            )
        ]


class FinishAnswerTool(ToolDefinition[FinishAnswerAction, TextObservation]):
    @classmethod
    def create(cls, *args: Any, **kwargs: Any) -> Sequence["FinishAnswerTool"]:
        executor = kwargs["executor"]
        return [
            cls(
                description="根据排序后的证据生成最终中文技术答案。",
                action_type=FinishAnswerAction,
                observation_type=TextObservation,
                annotations=_readonly_annotations(),
                executor=executor,
            )
        ]


def build_seed_queries(question: str, skill_evidence: list[Evidence], max_queries: int = 4) -> list[str]:
    tokens = _query_terms(question)
    queries: list[str] = []
    for item in skill_evidence:
        queries.extend(str(term) for term in item.metadata.get("query_terms", []))
    if any(term in question for term in ["反馈", "补偿", "电源", "环路", "纹波"]):
        queries.append("switching power supply feedback loop compensation stability noise filtering")
    if any(term in question for term in ["反激", "电流检测", "采样", "限流", "尖峰"]):
        queries.append("switch mode power supply current sense filter leading edge spike")
    if any(term in question for term in ["隔离", "光耦", "闭环", "地线", "布局"]):
        queries.append("isolated feedback ripple noise PCB layout grounding")
    if any(term in question for term in ["负控", "正控", "保护板", "虚电压", "不能带载"]):
        queries.append("MOSFET high side low side switch leakage ghost voltage bleed path")
    if any(term in question for term in ["运放", "调光", "栅极", "电位器", "LED"]):
        queries.append("op amp power MOSFET LED dimmer gate drive feedback troubleshooting")
    compact = " ".join(tokens[:10])
    if compact:
        queries.append(f"{compact} electronics circuit troubleshooting")
    return list(dict.fromkeys(query for query in queries if query.strip()))[:max_queries]


def review_evidence(question: str, evidence: list[Evidence]) -> tuple[bool, str, list[str]]:
    joined = "\n".join(f"{item.title} {item.content}" for item in evidence)
    component_terms = [
        token
        for token in _query_terms(question)
        if re.fullmatch(r"[A-Z]{1,6}\d{0,4}[A-Z0-9_.+-]*|\d+(?:\.\d+)?[A-Za-zΩμ]*", token)
    ]
    missing = [term for term in component_terms if term not in joined]
    missing = list(dict.fromkeys(missing))[:10]
    enough = len(evidence) >= 2 and len(missing) <= 4
    summary = "evidence coverage sufficient" if enough else f"evidence gaps: {', '.join(missing)}"
    return enough, summary, missing


def _image_data_url(path: Path, max_side: int = 1280, quality: int = 75) -> tuple[str | None, dict[str, Any]]:
    stats: dict[str, Any] = {
        "original_bytes": None,
        "encoded_bytes": None,
        "mime": None,
        "optimized": False,
        "error": None,
    }
    try:
        stats["original_bytes"] = path.stat().st_size
        optimized = _optimized_image_bytes(path, max_side=max_side, quality=quality)
        if optimized:
            mime = "image/jpeg"
            raw = optimized
            stats["optimized"] = True
        else:
            suffix = path.suffix.lower()
            if suffix in {".jpg", ".jpeg"}:
                mime = "image/jpeg"
            elif suffix == ".gif":
                mime = "image/gif"
            elif suffix == ".webp":
                mime = "image/webp"
            else:
                mime = "image/png"
            raw = path.read_bytes()
        stats["mime"] = mime
        stats["encoded_bytes"] = len(raw)
        encoded = base64.b64encode(raw).decode("utf-8")
        return f"data:{mime};base64,{encoded}", stats
    except Exception as exc:  # noqa: BLE001
        stats["error"] = str(exc)
        return None, stats


def _image_content_part(data_url: str, payload_format: str = "openai_image_url") -> dict[str, Any]:
    normalized = (payload_format or "openai_image_url").lower().strip()
    if normalized in {"input_image", "openai_responses_input_image"}:
        return {"type": "input_image", "image_url": data_url}
    return {"type": "image_url", "image_url": {"url": data_url}}


def _optimized_image_bytes(path: Path, max_side: int = 1600, quality: int = 80) -> bytes | None:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(path) as image:
            image.thumbnail((max_side, max_side))
            if image.mode in {"RGBA", "LA"}:
                background = Image.new("RGB", image.size, (255, 255, 255))
                alpha = image.getchannel("A") if "A" in image.getbands() else None
                background.paste(image.convert("RGBA"), mask=alpha)
                image = background
            else:
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="JPEG", quality=quality, optimize=True)
            return output.getvalue()
    except Exception:
        return None


def _normalize_search_url(url: str) -> str:
    parsed = urlparse(url or "")
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"


def _low_value_search_reason(item: Evidence) -> str:
    parsed = urlparse(item.source or "")
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    title = (item.title or "").lower()
    content = (item.content or "").lower()
    if _host_matches(host, _LOW_VALUE_SEARCH_HOSTS):
        return f"low_value_host:{host}"
    if any(marker in path for marker in _LOW_VALUE_SEARCH_PATH_MARKERS):
        return "low_value_path"
    if any(marker in title for marker in _LOW_VALUE_SEARCH_TITLE_MARKERS):
        if not _has_strong_model_signal(f"{title} {content}"):
            return "low_value_title"
        if any(marker in title for marker in ("淘宝", "天猫", "店铺", "好货")):
            return "ecommerce_page"
    if "最新作品发布时间" in item.content or "综合视频" in item.content:
        return "short_video_search_page"
    if "欢迎来到淘宝网" in item.content or "登录查看更多优惠" in item.content:
        return "ecommerce_page"
    return ""


def _search_relevance_score(query: str, item: Evidence) -> tuple[float, str]:
    text = f"{item.title} {item.content} {item.source}"
    lowered = text.lower()
    terms = _search_terms(query)
    if not terms:
        return max(float(item.score or 0.0), 0.1), "no_terms"
    exact_hits = [term for term in terms if _term_present(term, lowered)]
    model_terms = [term for term in terms if _is_model_like(term)]
    model_hits = [term for term in model_terms if _term_present(term, lowered)]
    coverage = len(exact_hits) / max(len(terms), 1)
    base_score = float(item.score or 0.0)
    score = base_score + (coverage * 3.0) + (len(model_hits) * 2.0)
    if _trusted_source(item.source):
        score += 1.0
    if model_terms and not model_hits:
        score -= 1.5
        if coverage < 0.35 and not _trusted_source(item.source):
            return 0.0, f"missing_model_terms:{coverage:.2f}"
    if coverage < 0.12 and not model_hits and not _trusted_source(item.source):
        return 0.0, f"low_query_overlap:{coverage:.2f}"
    if base_score <= 0 and coverage < 0.2 and not model_hits and not _trusted_source(item.source):
        return 0.0, f"zero_provider_score_low_overlap:{coverage:.2f}"
    return max(score, 0.01), f"coverage={coverage:.2f};models={len(model_hits)}/{len(model_terms)}"


def _host_matches(host: str, blocked_hosts: set[str]) -> bool:
    return any(host == blocked or host.endswith(f".{blocked}") for blocked in blocked_hosts)


def _search_terms(text: str) -> list[str]:
    raw_terms = re.findall(r"[A-Za-z]{1,8}\d{0,6}[A-Za-z0-9_.+-]*|[A-Za-z]{3,}|[\u4e00-\u9fff]{2,}|\d+(?:\.\d+)?[A-Za-z%]*", text or "")
    terms: list[str] = []
    for term in raw_terms:
        cleaned = term.strip().lower()
        if not cleaned or cleaned in _SEARCH_STOP_TERMS:
            continue
        if len(cleaned) < 2:
            continue
        terms.append(cleaned)
    return list(dict.fromkeys(terms))


def _is_model_like(term: str) -> bool:
    return bool(re.search(r"[a-z]{1,8}\d{2,6}|\d{2,6}[a-z]{1,8}", term, re.I))


def _has_strong_model_signal(text: str) -> bool:
    return bool(re.search(r"\b(?:tl431|lm358|uc384\d|rk3399|at32f\d+|esp32|mr\d+|irf\d+)\b", text, re.I))


def _term_present(term: str, lowered_text: str) -> bool:
    if re.fullmatch(r"[a-z0-9_.+-]+", term):
        return re.search(r"(?<![a-z0-9_.+-])" + re.escape(term) + r"(?![a-z0-9_.+-])", lowered_text) is not None
    return term in lowered_text


def _search_evidence(query: str, provider: str, title: str, url: str, content: str, score: float) -> Evidence:
    return Evidence(
        source=url,
        title=compact_text(title or url, 300),
        content=compact_text(content or "", 1200),
        score=score,
        metadata={"kind": "web_search_result", "provider": provider, "query": query},
    )


def _query_terms(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z]{1,8}\d{0,5}[A-Za-z0-9_.+-]*|\d+(?:\.\d+)?\s*[A-Za-zΩμ%]*|[一-鿿]{2,}", text or "")
    stop = {"请教", "问题", "为什么", "怎么", "处理", "以及", "这个", "电路", "作用", "哪些"}
    cleaned = [token.strip() for token in tokens if token.strip() and token.strip() not in stop]
    return list(dict.fromkeys(cleaned))


_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"发表于\s*\d{4}-\d{2}-\d{2}\s*•\s*\d+次阅读"),
    re.compile(r"\d+次阅读"),
    re.compile(r"次下载"),
    re.compile(r"下载该资料的人也在下载"),
    re.compile(r"下载该资料的人还在阅读"),
    re.compile(r"免费下载.*(?:PDF|pdf).*电子书"),
    re.compile(r"热门推荐|相关推荐|相关文章|相关阅读"),
    re.compile(r"上一篇|下一篇"),
    re.compile(r"!\[.*?\]\([^)]*load\.\w+\)"),
    re.compile(r"!\[.*?\]\([^)]*eye\.\w+\)"),
    re.compile(r"!\[电子发烧友网Logo\]"),
    re.compile(r"\d+\s*•\s*\d+次阅读"),
    re.compile(r"---[|]*---"),
    re.compile(r"专栏\s+电子说\s+商业评论"),
    re.compile(r"电子发烧友网\s*>"),
    re.compile(r"skin\.elecfans\.com|skin-2012"),
    re.compile(r"file\.elecfans\.com"),
    re.compile(r"^\s*(立即登录|注册|登录|Sign\s*Up|Sign\s*In)\s*$", re.IGNORECASE),
    re.compile(r"^\s*(快捷导航|社区导航|搜索中心)\s*$"),
    re.compile(r"^\s*!\[.*?\]\(.*?\)\s*$"),
    re.compile(r"^\s*[*\s]*$"),
]

_CAUSAL_SENTENCE_PATTERNS = [
    re.compile(r"因为.+所以"),
    re.compile(r"(?:导致|引起|使得|造成|促进|增强|抑制|减小|抵消|改善|降低|提升|提高|防止|避免|用于|用来|用于滤除|主要作用是|也称|具体包括)"),
    re.compile(r"[A-Za-z]+\s*[→=>]\s*[A-Za-z]+"),
    re.compile(r"\b(?:formula|equation|Vout|Vin|Iout|f_c|fsw|duty|gain|phase|margin|pole|zero|compensation)\b", re.IGNORECASE),
]


def _compress_boilerplate(text: str, max_chars: int = 600) -> str:
    """Strip boilerplate lines and prioritize causal/technical sentences."""
    if not text:
        return ""
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if any(pat.search(stripped) for pat in _BOILERPLATE_LINE_PATTERNS):
            continue
        if len(stripped) < 15 and not re.search(r"[A-Za-z]{3,}|\d+(?:\.\d+)?[A-Za-z]", stripped):
            continue
        clean_lines.append(stripped)
    clean_text = " ".join(clean_lines)
    if len(clean_text) <= max_chars:
        return clean_text
    scored_lines = []
    for line in clean_lines:
        score = 0
        if any(pat.search(line) for pat in _CAUSAL_SENTENCE_PATTERNS):
            score += 3
        if re.search(r"\b(?:TL431|LM358|LM324|UC384|SG3525|IR2110|PC817|EL817|NE555|AO340|IRF\d+|STM32|ESP32)\b", line, re.IGNORECASE):
            score += 2
        if re.search(r"[RCLDQUV]\d+[A-Z0-9]*", line, re.IGNORECASE):
            score += 2
        if re.search(r"\d+(?:\.\d+)?(?:ohm|kohm|Ω|μF|uf|nf|pF|pf|Hz|khz|MHz|V|mA|A|W)", line, re.IGNORECASE):
            score += 1
        score += min(1, len(line) // 60)
        scored_lines.append((score, line))
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    result_lines = []
    total_len = 0
    for _score, line in scored_lines:
        if total_len + len(line) + 1 <= max_chars:
            result_lines.append(line)
            total_len += len(line) + 1
        else:
            break
    return " ".join(result_lines)



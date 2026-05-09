from __future__ import annotations

import base64
from io import BytesIO
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

import requests
import yaml
from pydantic import Field

from agent.prompts import (
    build_final_answer_system_prompt,
    VISION_SYSTEM_PROMPT,
    build_final_answer_user_prompt,
    build_vision_user_prompt,
)
from llm_client import LLMClient
from schemas import Evidence, StandardSample, ToolEvent
from tools.utils import compact_text, timer
from tools.web_reader import WebReader
from tools.web_search import WebSearch
from tools.openhands_browser import OpenHandsBrowserConfig, OpenHandsBrowserFetcher
from tools.circuit_kb import (
    CircuitMarkdownRetriever,
    classify_query_terms,
    is_boilerplate_text,
    is_circuitmaker_project_source,
    is_low_value_source,
)

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
    ) -> None:
        self.provider_order = list(provider_order or ["html"])
        self.api_key_envs = dict(api_key_envs or {})
        self.api_keys = {key.lower(): value for key, value in dict(api_keys or {}).items() if value}
        self.timeout = timeout
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
                if results:
                    return ToolRun(
                        results,
                        summary=f"html search returned {len(results)} results",
                        metadata={"provider": "html", "query": query},
                    )
                errors.append(f"html: {error or 'no results'}")
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
            return ToolRun(
                [evidence],
                summary="kept search snippet for pdf/binary/known-slow page",
                metadata={"read_backend": "snippet_fallback"},
            )

        openhands_error = None
        if self.enable_openhands_browser_primary:
            browser_result = self.openhands_browser.fetch(url, max_chars=self.openhands_browser_max_chars)
            if browser_result.evidence:
                browser_result.evidence.title = title or browser_result.evidence.title or url
                browser_result.evidence.metadata["read_backend"] = "openhands_browser"
                return ToolRun(
                    [browser_result.evidence],
                    summary=f"read page with OpenHands browser {url}",
                    metadata={"read_backend": "openhands_browser"},
                )
            openhands_error = browser_result.error

        page = self.reader.read(url, max_chars=6000)
        if page.evidence:
            page.evidence.metadata["read_backend"] = "requests_bs4"
            if openhands_error:
                page.evidence.metadata["openhands_error"] = openhands_error
            return ToolRun(
                [page.evidence],
                summary=f"read page {url}",
                errors=[openhands_error] if openhands_error else [],
                metadata={"read_backend": "requests_bs4", "openhands_error": openhands_error},
            )
        if snippet:
            evidence = _search_evidence("", "snippet", title or url, url, snippet, 0.0)
            evidence.metadata["read_error"] = page.error
            evidence.metadata["web_reader_error"] = page.error
            evidence.metadata["openhands_error"] = openhands_error
            evidence.metadata["read_backend"] = "snippet_fallback"
            errors = [error for error in [openhands_error, page.error or "read failed"] if error]
            return ToolRun(
                [evidence],
                summary="page read failed; kept search snippet",
                success=True,
                errors=errors,
                metadata={
                    "read_backend": "snippet_fallback",
                    "openhands_error": openhands_error,
                    "web_reader_error": page.error,
                },
            )
        errors = [error for error in [openhands_error, page.error or "read failed"] if error]
        return ToolRun(
            [],
            summary=f"page read failed {url}",
            success=False,
            errors=errors,
            metadata={
                "read_backend": "failed",
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

    def close(self) -> None:
        if hasattr(self.openhands_browser, "close"):
            self.openhands_browser.close()


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

    def __init__(self, llm: LLMClient, enabled: bool = True) -> None:
        self.llm = llm
        self.enabled = enabled

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
            },
        )
        return ToolRun(
            [evidence],
            summary=f"qwen search returned results for query={query}",
            metadata={"query": query, "token_usage": response.token_usage},
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




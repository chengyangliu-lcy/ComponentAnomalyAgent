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
    FINAL_ANSWER_SYSTEM_PROMPT,
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
            if item.metadata.get("kind") == "local_kb_chunk":
                item.metadata["discarded_kb"] = True
                discarded_kb += 1
                continue
            text = f"{item.title} {item.content} {item.source}".lower()
            score = float(item.score or 0.0)
            score += sum(1.0 for term in terms if term.lower() in text)
            if item.metadata.get("kind") == "domain_skill":
                score += 4.0
            if item.metadata.get("kind") == "image_inspection":
                score += 3.0
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

    def __init__(self, llm: LLMClient, max_evidence: int = 12) -> None:
        self.llm = llm
        self.max_evidence = max_evidence

    def __call__(self, action: FinishAnswerAction, conversation: Any = None) -> TextObservation:
        evidence = [Evidence(**item) for item in action.evidence]
        run = self.run(action.question, evidence)
        return TextObservation(summary=run.summary, answer_text=run.text, errors=run.errors, is_error=not run.success)

    def run(self, question: str, evidence: list[Evidence], allow_llm: bool = True) -> ToolRun:
        question_hints = _query_terms(question)[:12]
        compact_evidence = self._dedupe_evidence(evidence, question=question)
        evidence_text = self._format_evidence_for_answer(compact_evidence)
        if allow_llm and self.llm.available:
            response = self.llm.chat(
                [
                    {
                        "role": "system",
                        "content": FINAL_ANSWER_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_final_answer_user_prompt(question, evidence_text, question_hints),
                    },
                ],
                temperature=0.1,
            )
            if response.content:
                return ToolRun(
                    text=response.content,
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
            if item.metadata.get("kind") == "local_kb_chunk":
                item.metadata["discarded_kb"] = True
                continue
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
            elif kind in {"web_page", "web_search_result"}:
                limit = 600
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
        skill_notes = [item.content for item in evidence if item.metadata.get("kind") == "domain_skill"]
        image_notes = [item.content for item in evidence if item.metadata.get("kind") == "image_inspection"]
        web_notes = [item.content for item in evidence if item.metadata.get("kind") in {"web_search_result", "web_page"}]
        hints = ", ".join(question_hints) if question_hints else "题面关键对象未能可靠抽取"
        mechanisms = "\n".join(f"- {compact_text(note, 420)}" for note in skill_notes[:3]) or "- 现有领域证据不足，需要结合题面和实测波形判断。"
        image_summary = "\n".join(f"- {compact_text(note, 360)}" for note in image_notes[:2]) or "- 未获得可用图片细节，需以实物图或原理图复核节点。"
        web_summary = "\n".join(f"- {compact_text(note, 320)}" for note in web_notes[:2]) or "- 未获得可靠公开资料补充。"
        return (
            f"结论：围绕题面中的 {hints} 判断，优先把异常归因到已出现的反馈、采样、驱动、滤波或开关路径上，"
            "再按对应节点做验证；不要只做通用排查。\n\n"
            "原因机制：\n"
            f"{mechanisms}\n\n"
            "图片依据：\n"
            f"{image_summary}\n\n"
            "公开资料依据：\n"
            f"{web_summary}\n\n"
            "检查步骤：先测题面提到的输出/采样/反馈节点波形，再核对相关电阻、电容、MOS、光耦、TL431或控制芯片引脚的实际连接，最后验证修改前后的纹波、尖峰、输出电压和温升。\n"
            "处理建议与不确定性：按已命中的元件和节点调整补偿、滤波、泄放、驱动或布局；如果证据没有显示具体数值，不应给出确定参数，应通过示波器和原理图复核后再定值。"
        )


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
    tokens = re.findall(r"[A-Za-z]{1,8}\d{0,5}[A-Za-z0-9_.+-]*|\d+(?:\.\d+)?\s*[A-Za-zΩμ%]*|[\u4e00-\u9fff]{2,}", text or "")
    stop = {"请教", "问题", "为什么", "怎么", "处理", "以及", "这个", "电路", "作用", "哪些"}
    cleaned = [token.strip() for token in tokens if token.strip() and token.strip() not in stop]
    return list(dict.fromkeys(cleaned))


def _kb_evidence_allowed(question: str, item: Evidence, min_relevance: float = 5.0) -> bool:
    relevance = float(item.metadata.get("kb_relevance") or item.metadata.get("rerank_score") or item.score or 0.0)
    if relevance < min_relevance:
        return False
    if item.metadata.get("high_relevance") is False:
        return False
    if is_low_value_source(item.source) or is_boilerplate_text(f"{item.title}\n{item.source}\n{item.content}"):
        return False
    matched_terms = item.metadata.get("matched_query_terms") if isinstance(item.metadata, dict) else {}
    if is_circuitmaker_project_source(item.source):
        strong_matched = []
        if isinstance(matched_terms, dict):
            for key in ("models", "refdes", "values"):
                strong_matched.extend(matched_terms.get(key) or [])
        if not strong_matched and relevance < min_relevance + 2.0:
            return False
    if not question:
        return True
    text = f"{item.title} {item.source} {item.content}".lower()
    profile = classify_query_terms(question)
    terms = [term.lower() for term in _query_terms(question)]
    strong_terms = [term.lower() for term in [*profile["models"], *profile["refdes"], *profile["values"]]]
    if strong_terms:
        return any(term in text for term in strong_terms)
    matched = sum(1 for term in terms if len(term) >= 3 and term in text)
    return matched >= 2


def _trusted_source(source: str) -> bool:
    host = urlparse(source).hostname or ""
    trusted_markers = (
        "ti.com",
        "analog.com",
        "onsemi.com",
        "st.com",
        "infineon.com",
        "electronics.stackexchange.com",
        "edn.com",
        "ridleyengineering.com",
    )
    return any(marker in host for marker in trusted_markers)


def _readonly_annotations(open_world: bool = False) -> Any:
    if ToolAnnotations is None:
        return None
    return ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=open_world,
    )


def evidence_to_json(items: list[Evidence]) -> list[dict[str, Any]]:
    return [item.to_json() for item in items]


def evidence_from_json(items: list[dict[str, Any]]) -> list[Evidence]:
    return [Evidence(**item) for item in items]

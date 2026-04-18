from __future__ import annotations

from typing import Any, Dict, List

from schemas import AgentPlan, Evidence, StandardSample, ToolEvent
from tools.browser import BrowserFallback
from tools.file_reader import FileReader
from tools.image_resolver import ImageResolver
from tools.logger import TraceLogger
from tools.retriever import LocalRetriever
from tools.utils import timer
from tools.web_reader import WebReader
from tools.web_search import WebSearch


class Executor:
    def __init__(
        self,
        config: Dict[str, Any],
        image_resolver: ImageResolver,
        retriever: LocalRetriever,
        file_reader: FileReader,
        web_search: WebSearch,
        web_reader: WebReader,
        browser: BrowserFallback,
    ) -> None:
        self.config = config
        self.image_resolver = image_resolver
        self.retriever = retriever
        self.file_reader = file_reader
        self.web_search = web_search
        self.web_reader = web_reader
        self.browser = browser

    def execute(self, sample: StandardSample, plan: AgentPlan, trace: TraceLogger) -> List[Evidence]:
        evidence: List[Evidence] = []
        if plan.needs_local_retrieval:
            evidence.extend(self._local_search(sample, plan, trace))
        if plan.needs_images:
            evidence.extend(self._image_context(sample, trace))
        if plan.needs_web_search:
            evidence.extend(self._web_search(plan, trace))
        return evidence

    def _local_search(self, sample: StandardSample, plan: AgentPlan, trace: TraceLogger) -> List[Evidence]:
        with timer() as elapsed:
            try:
                docs: List[Evidence] = []
                for query in plan.queries or [sample.question_text]:
                    docs.extend(
                        self.retriever.search(
                            query=query,
                            limit=int(self.config.get("max_local_docs", 4)),
                            post_id=sample.post_id,
                        )
                    )
                unique: Dict[str, Evidence] = {}
                for doc in docs:
                    unique.setdefault(doc.source, doc)
                docs = sorted(unique.values(), key=lambda item: item.score, reverse=True)[: int(self.config.get("max_local_docs", 4))]
                trace.add(
                    ToolEvent(
                        tool_name="retriever",
                        action="local_markdown_search",
                        success=True,
                        elapsed_seconds=elapsed["elapsed"],
                        summary=f"found {len(docs)} local documents",
                        inputs={"queries": plan.queries, "post_id": sample.post_id},
                        outputs={"sources": [doc.source for doc in docs]},
                    )
                )
                return docs
            except Exception as exc:  # noqa: BLE001
                trace.add(
                    ToolEvent(
                        tool_name="retriever",
                        action="local_markdown_search",
                        success=False,
                        elapsed_seconds=elapsed["elapsed"],
                        summary="local retrieval failed",
                        error=str(exc),
                    )
                )
                return []

    def _image_context(self, sample: StandardSample, trace: TraceLogger) -> List[Evidence]:
        with timer() as elapsed:
            found = [image for image in sample.images if image.exists and image.path]
            missing = [image.original_url for image in sample.images if not image.exists]
            trace.add(
                ToolEvent(
                    tool_name="image_resolver",
                    action="resolve_local_images",
                    success=not missing,
                    elapsed_seconds=elapsed["elapsed"],
                    summary=f"resolved {len(found)}/{len(sample.images)} images",
                    inputs={"post_id": sample.post_id},
                    outputs={"images": [image.to_json() for image in sample.images]},
                    error=f"missing images: {missing[:5]}" if missing else None,
                )
            )
            if not found:
                return []
            image_list = "\n".join(str(image.path) for image in found if image.path)
            return [
                Evidence(
                    source="local_images",
                    title=f"{sample.post_id} image references",
                    content=f"本样本包含 {len(found)} 张输入图片，可用于多模态模型输入：\n{image_list}",
                    metadata={"kind": "image_context"},
                )
            ]

    def _web_search(self, plan: AgentPlan, trace: TraceLogger) -> List[Evidence]:
        all_evidence: List[Evidence] = []
        max_queries = int(self.config.get("max_web_queries", 3))
        for query in plan.queries[:max_queries]:
            with timer() as elapsed:
                results, error = self.web_search.search(query, limit=int(self.config.get("max_web_results", 5)))
                trace.add(
                    ToolEvent(
                        tool_name="web_search",
                        action="html_search",
                        success=not error,
                        elapsed_seconds=elapsed["elapsed"],
                        summary=f"query={query}; results={len(results)}",
                        inputs={"query": query},
                        outputs={"sources": [result.source for result in results]},
                        error=error,
                    )
                )
            all_evidence.extend(results)
            successful_reads = 0
            for result in results:
                if successful_reads >= int(self.config.get("max_web_pages_to_read", 2)):
                    break
                if self._should_skip_page_read(result.source):
                    trace.add(
                        ToolEvent(
                            tool_name="web_reader",
                            action="skip_page_read",
                            success=True,
                            summary=result.source,
                            inputs={"url": result.source},
                            outputs={"reason": "search result snippet kept as evidence; page likely binary or blocked"},
                        )
                    )
                    continue
                with timer() as read_elapsed:
                    page = self.web_reader.read(result.source)
                    if not page.evidence and self.config.get("enable_browser_fallback", False):
                        browser_result = self.browser.fetch(result.source)
                        page.evidence = browser_result.evidence
                        page.error = page.error or browser_result.error
                    trace.add(
                        ToolEvent(
                            tool_name="web_reader",
                            action="read_page",
                            success=bool(page.evidence),
                            elapsed_seconds=read_elapsed["elapsed"],
                            summary=result.source,
                            inputs={"url": result.source},
                            outputs={"title": page.evidence.title if page.evidence else None},
                            error=page.error,
                        )
                    )
                    if page.evidence:
                        all_evidence.append(page.evidence)
                        successful_reads += 1
        return all_evidence

    def _should_skip_page_read(self, url: str) -> bool:
        lowered = url.lower()
        if lowered.endswith(".pdf") or "/pdf/" in lowered or "/resource/en/datasheet/" in lowered:
            return True
        blocked_product_pages = ("onsemi.com/products/", "st.com/en/power-management/")
        return any(marker in lowered for marker in blocked_product_pages)

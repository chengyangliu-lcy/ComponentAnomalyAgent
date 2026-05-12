from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from agent.executor import Executor
from agent.openhands_runtime import OpenHandsEvidenceRuntime
from agent.planner import Planner
from agent.reflector import Reflector
from agent.synthesizer import AnswerSynthesizer
from configs.config import RuntimeConfig
from llm_client import LLMClient
from schemas import InferenceResult, StandardSample
from tools.browser import BrowserFallback
from tools.dense_retriever import DenseRetriever
from tools.file_reader import FileReader
from tools.image_resolver import ImageResolver
from tools.logger import TraceLogger
from tools.retriever import LocalRetriever
from tools.web_reader import WebReader
from tools.web_search import WebSearch


class AgentPipeline:
    def __init__(self, config: RuntimeConfig, shared_dense_retriever: Optional[DenseRetriever] = None) -> None:
        self.config = config
        agent_cfg: Dict[str, Any] = dict(config.raw.get("agent", {}))
        self.strategy = str(agent_cfg.get("strategy", "legacy_single_agent_web_reasoning"))
        self.openhands_runtime = (
            OpenHandsEvidenceRuntime(config, shared_dense_retriever=shared_dense_retriever)
            if self.strategy in {"openhands_evidence_agent", "agentic_tool_loop"}
            else None
        )
        model_cfg = config.raw.get("model", {})
        timeout = int(config.raw.get("runtime", {}).get("request_timeout_seconds", 20))
        image_resolver = ImageResolver(config.image_root)
        if (
            agent_cfg.get("enable_local_retrieval")
            and not agent_cfg.get("allow_test_corpus_retrieval", False)
            and config.local_corpus_root.resolve() == config.image_root.resolve()
        ):
            agent_cfg["enable_local_retrieval"] = False
        retriever = LocalRetriever(config.local_corpus_root)
        llm = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.agent_model,
            temperature=float(model_cfg.get("temperature", 0.2)),
            max_tokens=int(model_cfg.get("max_tokens", 2000)),
            timeout=timeout,
        )
        self.planner = Planner(agent_cfg)
        self.executor = Executor(
            config=agent_cfg,
            image_resolver=image_resolver,
            retriever=retriever,
            file_reader=FileReader(),
            web_search=WebSearch(
                timeout=timeout,
                provider=str(agent_cfg.get("search_provider", "duckduckgo")),
                cache_ttl_seconds=int(agent_cfg.get("search_cache_ttl_seconds", 900)),
            ),
            web_reader=WebReader(timeout=timeout),
            browser=BrowserFallback(),
        )
        self.reflector = Reflector()
        self.synthesizer = AnswerSynthesizer(
            llm,
            send_images=bool(agent_cfg.get("send_images_to_llm", True)),
            max_images=int(agent_cfg.get("max_llm_images", 4)),
        )

    def run_sample(self, sample: StandardSample) -> InferenceResult:
        if self.openhands_runtime is not None:
            try:
                return self.openhands_runtime.run_sample(sample)
            except Exception as exc:  # noqa: BLE001
                if not bool(self.config.raw.get("agent", {}).get("fallback_to_legacy_on_runtime_error", True)):
                    raise
                legacy = self._run_legacy_sample(sample)
                legacy.errors.insert(0, f"{self.strategy} failed; legacy fallback used: {exc}")
                legacy.reasoning_summary = f"Agentic runtime failed and fell back to legacy pipeline. {legacy.reasoning_summary}"
                return legacy
        return self._run_legacy_sample(sample)

    def _run_legacy_sample(self, sample: StandardSample) -> InferenceResult:
        start = time.perf_counter()
        trace = TraceLogger()
        errors: list[str] = []
        plan = self.planner.plan(sample)
        evidence = self.executor.execute(sample, plan, trace)
        enough, reflection_summary, missing = self.reflector.assess(sample, plan, evidence)
        max_rounds = int(self.config.raw.get("agent", {}).get("max_reflection_rounds", 1))
        rounds = 0
        while not enough and rounds < max_rounds:
            plan.queries = self.reflector.supplemental_queries(sample, missing)
            evidence.extend(self.executor.execute(sample, plan, trace))
            enough, reflection_summary, missing = self.reflector.assess(sample, plan, evidence)
            rounds += 1
        answer, synth_summary, _single_usage, synth_errors = self.synthesizer.synthesize(sample, evidence)
        token_usage = self.synthesizer.llm.cumulative_usage
        errors.extend(error for error in synth_errors if error)
        elapsed = time.perf_counter() - start
        return InferenceResult(
            sample_id=sample.sample_id,
            question=sample.question_text,
            answer=answer,
            tools_used=trace.tool_names(),
            web_searched=any(event.tool_name == "web_search" for event in trace.events),
            tool_trace=trace.events,
            reasoning_summary=f"{reflection_summary} {synth_summary}",
            elapsed_seconds=round(elapsed, 4),
            token_usage=token_usage,
            errors=errors,
            plan=plan,
        )

    def run_by_id(self, samples: list[StandardSample], sample_id: str) -> InferenceResult:
        for sample in samples:
            if sample.sample_id == sample_id or sample.post_id == sample_id:
                return self.run_sample(sample)
        raise KeyError(f"sample not found: {sample_id}")


def result_path(outputs_dir: Path, experiment_name: str, filename: str) -> Path:
    return outputs_dir / experiment_name / filename

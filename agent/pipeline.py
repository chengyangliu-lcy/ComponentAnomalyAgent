from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from agent.executor import Executor
from agent.planner import Planner
from agent.reflector import Reflector
from agent.synthesizer import AnswerSynthesizer
from configs.config import RuntimeConfig
from llm_client import LLMClient
from schemas import InferenceResult, StandardSample
from tools.browser import BrowserFallback
from tools.file_reader import FileReader
from tools.image_resolver import ImageResolver
from tools.logger import TraceLogger
from tools.retriever import LocalRetriever
from tools.web_reader import WebReader
from tools.web_search import WebSearch


class AgentPipeline:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        agent_cfg: Dict[str, Any] = dict(config.raw.get("agent", {}))
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
        )
        self.planner = Planner(agent_cfg)
        self.executor = Executor(
            config=agent_cfg,
            image_resolver=image_resolver,
            retriever=retriever,
            file_reader=FileReader(),
            web_search=WebSearch(timeout=timeout),
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
        answer, synth_summary, token_usage, synth_errors = self.synthesizer.synthesize(sample, evidence)
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

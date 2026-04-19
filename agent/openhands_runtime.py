from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.prompts import PLANNER_SYSTEM_PROMPT, planner_guidance
from configs.config import RuntimeConfig
from llm_client import LLMClient
from schemas import AgentPlan, Evidence, InferenceResult, StandardSample, ToolEvent
from tools.evidence_tools import (
    APIWebSearchExecutor,
    DomainSkillExecutor,
    EvidenceRankExecutor,
    FinishAnswerExecutor,
    ImageInspectAction,
    ImageInspectExecutor,
    RobustWebReadExecutor,
    ToolRun,
    build_seed_queries,
    review_evidence,
)
from tools.logger import TraceLogger
from tools.utils import compact_text, timer


ALLOWED_TOOLS = {
    "inspect_image",
    "match_domain_skill",
    "web_search",
    "web_read",
    "rank_evidence",
    "review_evidence",
    "finish_answer",
}

DEFAULT_MAX_DOMAIN_SKILL_CALLS = 1


@dataclass
class AgentAction:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    stop: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentObservation:
    success: bool
    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    recoverable: bool = True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    sample: StandardSample
    image_paths: list[str]
    evidence: list[Evidence] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    selected_actions: list[dict[str, Any]] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    read_urls: set[str] = field(default_factory=set)
    tool_counts: dict[str, int] = field(default_factory=dict)
    iteration: int = 0
    consecutive_errors: int = 0
    planner_failures: int = 0
    answer: str = ""
    final_stop_reason: str = ""


class OpenHandsEvidenceRuntime:
    """Read-only OpenHands-style agent loop for evidence gathering.

    OpenHands SDK remains the tool-contract compatibility layer. The runtime
    itself is an action/observation loop: a planner chooses one tool per turn,
    observes the result, and decides whether to continue or finish.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.agent_cfg: dict[str, Any] = dict(config.raw.get("agent", {}))
        self.web_cfg: dict[str, Any] = dict(config.raw.get("web", {}))
        model_cfg = config.raw.get("model", {})
        request_timeout = int(config.raw.get("runtime", {}).get("request_timeout_seconds", 20))
        tool_timeout = int(self.agent_cfg.get("tool_timeout_seconds", request_timeout))
        planner_timeout = int(self.agent_cfg.get("planner_timeout_seconds", min(tool_timeout, 15)))
        vision_timeout = int(self.agent_cfg.get("vision_timeout_seconds", tool_timeout))
        final_answer_timeout = int(self.agent_cfg.get("final_answer_timeout_seconds", tool_timeout))
        web_read_timeout = int(self.agent_cfg.get("web_read_timeout_seconds", tool_timeout))
        max_retries = int(self.agent_cfg.get("llm_max_retries", 0))

        self.llm = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.agent_model,
            temperature=0.0,
            max_tokens=int(self.agent_cfg.get("planner_max_tokens", 700)),
            timeout=planner_timeout,
            max_retries=max_retries,
            extra_body=model_cfg.get("planner_extra_body", {}),
        )
        self.answer_llm = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.agent_model,
            temperature=float(model_cfg.get("temperature", 0.2)),
            max_tokens=int(model_cfg.get("max_tokens", 2000)),
            timeout=final_answer_timeout,
            max_retries=max_retries,
            extra_body=model_cfg.get("answer_extra_body", {}),
        )
        self.image_llm = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            model=config.vision_model,
            temperature=0.0,
            max_tokens=int(self.agent_cfg.get("image_inspection_max_tokens", 800)),
            timeout=vision_timeout,
            max_retries=0,
            extra_body=model_cfg.get("vision_extra_body", {}),
        )
        self.sdk_status = self._inspect_openhands_sdk()
        self.image_tool = ImageInspectExecutor(
            self.image_llm,
            enabled=bool(self.agent_cfg.get("enable_image_inspection_llm", self.agent_cfg.get("send_images_to_llm", True))),
            max_images=int(self.agent_cfg.get("max_llm_images", 4)),
            payload_format=str(self.agent_cfg.get("image_payload_format", "openai_image_url")),
            image_max_side=int(self.agent_cfg.get("image_max_side", 1280)),
            image_jpeg_quality=int(self.agent_cfg.get("image_jpeg_quality", 75)),
        )
        self.search_tool = APIWebSearchExecutor(
            provider_order=self.web_cfg.get("provider_order", ["html"]),
            api_key_envs=self.web_cfg.get("api_key_envs", {}),
            api_keys=self.web_cfg.get("api_keys", {}),
            timeout=tool_timeout,
            html_provider=str(self.agent_cfg.get("search_provider", "duckduckgo")),
        )
        self.domain_tool = DomainSkillExecutor(
            skill_path=self._skill_path(),
            forbidden_roots=[config.image_root, config.dataset_path.parent / "2025"],
        )
        self.read_tool = RobustWebReadExecutor(timeout=web_read_timeout)
        self.rank_tool = EvidenceRankExecutor()
        self.finish_tool = FinishAnswerExecutor(self.answer_llm)

    def run_sample(self, sample: StandardSample) -> InferenceResult:
        start = time.perf_counter()
        trace = TraceLogger()
        image_paths = [str(image.path) for image in sample.images if image.path and image.exists]
        state = AgentState(sample=sample, image_paths=image_paths)
        max_iterations = int(self.agent_cfg.get("max_iterations", 8))
        max_total_seconds = float(self.agent_cfg.get("max_total_seconds", 180))
        max_consecutive_errors = int(self.agent_cfg.get("max_consecutive_tool_errors", 3))

        plan = AgentPlan(
            question_type=self._classify(sample.question_text),
            needs_images=bool(image_paths) and bool(self.agent_cfg.get("use_images", True)),
            needs_local_retrieval=False,
            needs_web_search=bool(self.agent_cfg.get("enable_web_search", True)),
            queries=[],
            steps=[],
            strategy="agentic_tool_loop",
            budgets={
                "max_iterations": max_iterations,
                "max_total_seconds": max_total_seconds,
                "tool_timeout_seconds": self.agent_cfg.get("tool_timeout_seconds"),
                "planner_timeout_seconds": self.agent_cfg.get("planner_timeout_seconds"),
                "vision_timeout_seconds": self.agent_cfg.get("vision_timeout_seconds"),
                "final_answer_timeout_seconds": self.agent_cfg.get("final_answer_timeout_seconds"),
                "web_read_timeout_seconds": self.agent_cfg.get("web_read_timeout_seconds"),
            },
        )

        self._add_bootstrap_events(trace, sample, plan)

        for iteration in range(1, max_iterations + 1):
            state.iteration = iteration
            if time.perf_counter() - start >= max_total_seconds:
                state.final_stop_reason = "total time budget exhausted"
                break
            if state.consecutive_errors >= max_consecutive_errors:
                state.final_stop_reason = "too many consecutive recoverable tool errors"
                break

            with timer() as planner_elapsed:
                forced_action = self._forced_action(state)
                if forced_action:
                    raw_action, action_error = forced_action.to_json(), None
                else:
                    raw_action, action_error = self._next_action(state, start, max_total_seconds)
            action, validation_error = self._validate_action(raw_action)
            if action_error or validation_error:
                error = action_error or validation_error or "invalid planner action"
                state.errors.append(error)
                state.consecutive_errors += 1
                observation = AgentObservation(False, "planner action rejected", error=error, recoverable=True)
                state.observations.append(observation.to_json())
                trace.add(
                    self._planner_event(
                        state,
                        raw_action,
                        action,
                        observation,
                        start,
                        max_total_seconds,
                        success=False,
                        elapsed=planner_elapsed["elapsed"],
                    )
                )
                continue

            state.selected_actions.append(action.to_json())
            plan.selected_actions = state.selected_actions
            trace.add(
                self._planner_event(
                    state,
                    raw_action,
                    action,
                    AgentObservation(True, "planner selected action"),
                    start,
                    max_total_seconds,
                    success=True,
                    elapsed=planner_elapsed["elapsed"],
                )
            )

            guard_error = self._guard_action(state, action, start, max_total_seconds)
            if guard_error:
                observation = AgentObservation(False, "runtime skipped unsafe or exhausted action", error=guard_error, recoverable=True)
                state.observations.append(observation.to_json())
                state.consecutive_errors += 1
                state.errors.append(guard_error)
                trace.add(self._guard_event(state, action, observation, start, max_total_seconds))
                continue

            event, observation = self._execute_action(state, action, start, max_total_seconds)
            trace.add(event)
            state.observations.append(observation.to_json())
            if observation.success:
                state.consecutive_errors = 0
            else:
                state.consecutive_errors += 1
                if observation.error:
                    state.errors.append(observation.error)
            if action.stop or action.tool_name == "finish_answer":
                state.final_stop_reason = state.final_stop_reason or "finish_answer selected"
                break

        if not state.answer:
            action = AgentAction(
                tool_name="finish_answer",
                args={"allow_llm": self._can_start_action("finish_answer", start, max_total_seconds)},
                reason="fallback finalization after loop budget ended",
                stop=True,
            )
            state.selected_actions.append(action.to_json())
            event, observation = self._execute_action(state, action, start, max_total_seconds)
            trace.add(event)
            state.observations.append(observation.to_json())
            state.final_stop_reason = state.final_stop_reason or "fallback finish after loop"

        elapsed_total = time.perf_counter() - start
        plan.queries = state.queries
        plan.selected_actions = state.selected_actions
        plan.final_stop_reason = state.final_stop_reason
        plan.budgets["elapsed_seconds"] = round(elapsed_total, 4)
        return InferenceResult(
            sample_id=sample.sample_id,
            question=sample.question_text,
            answer=state.answer,
            tools_used=trace.tool_names(),
            web_searched=any(event.tool_name == "web_search" for event in trace.events),
            tool_trace=trace.events,
            reasoning_summary=(
                "Agentic action/observation loop; "
                f"iterations={state.iteration}; evidence={len(state.evidence)}; stop={state.final_stop_reason}"
            ),
            elapsed_seconds=round(elapsed_total, 4),
            token_usage={},
            errors=list(dict.fromkeys(error for error in state.errors if error and "LLM unavailable" not in error)),
            plan=plan,
        )

    def _next_action(
        self,
        state: AgentState,
        start: float,
        max_total_seconds: float,
    ) -> tuple[dict[str, Any], str | None]:
        if not self.llm.available:
            return self._fallback_action(state).to_json(), None
        max_planner_failures = int(self.agent_cfg.get("max_planner_failures", 2))
        if state.planner_failures >= max_planner_failures:
            fallback = self._fallback_action(state)
            fallback.args["_planner_fallback"] = True
            fallback.args["_planner_error_type"] = "planner_circuit_open"
            return fallback.to_json(), None

        payload, error = self.llm.json_chat(
            [
                {
                    "role": "system",
                    "content": PLANNER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        self._planner_state_payload(state, start, max_total_seconds),
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        if error:
            state.planner_failures += 1
            fallback = self._fallback_action(state)
            fallback.args["_planner_fallback"] = True
            fallback.args["_planner_error_type"] = self._planner_error_type(error)
            return fallback.to_json(), None
        state.planner_failures = 0
        return payload, None

    def _validate_action(self, payload: dict[str, Any]) -> tuple[AgentAction | None, str | None]:
        if not isinstance(payload, dict):
            return None, "planner action must be a JSON object"
        tool_name = str(payload.get("tool_name") or payload.get("tool") or payload.get("action") or "").strip()
        if tool_name not in ALLOWED_TOOLS:
            return None, f"unknown planner tool: {tool_name or '<empty>'}"
        args = payload.get("args") or {}
        if not isinstance(args, dict):
            return None, "planner action args must be a JSON object"
        reason = compact_text(str(payload.get("reason") or ""), 500)
        return AgentAction(tool_name=tool_name, args=args, reason=reason, stop=bool(payload.get("stop"))), None

    def _execute_action(
        self,
        state: AgentState,
        action: AgentAction,
        start: float,
        max_total_seconds: float,
    ) -> tuple[ToolEvent, AgentObservation]:
        before_evidence = len(state.evidence)
        if action.tool_name == "finish_answer" and not self._can_start_action("finish_answer", start, max_total_seconds):
            action.args["allow_llm"] = False
        with timer() as elapsed:
            run = self._run_tool(state, action)
        if action.tool_name == "rank_evidence" and run.evidence:
            state.evidence = run.evidence
        elif action.tool_name == "finish_answer":
            state.answer = run.text
        else:
            state.evidence.extend(run.evidence)
            state.evidence = self._dedupe_evidence(state.evidence)
        state.tool_counts[action.tool_name] = state.tool_counts.get(action.tool_name, 0) + 1

        observation = AgentObservation(
            success=run.success,
            summary=run.summary,
            evidence=[item.to_json() for item in run.evidence],
            error="; ".join(run.errors) if run.errors else None,
            recoverable=action.tool_name != "finish_answer",
        )
        event = self._event_for_action(
            state,
            action,
            run,
            elapsed["elapsed"],
            before_evidence,
            observation,
            start,
            max_total_seconds,
        )
        return event, observation

    def _run_tool(self, state: AgentState, action: AgentAction) -> ToolRun:
        question = state.sample.question_text
        if action.tool_name == "inspect_image":
            if not bool(self.agent_cfg.get("use_images", True)):
                return ToolRun(summary="image use disabled by config", success=False, errors=["image use disabled"])
            return self.image_tool.run(
                ImageInspectAction(
                    sample_id=state.sample.sample_id,
                    question=question,
                    image_paths=state.image_paths,
                )
            )
        if action.tool_name == "match_domain_skill":
            return self.domain_tool.run(self._domain_skill_question(question, action))
        if action.tool_name == "web_search":
            if not bool(self.agent_cfg.get("enable_web_search", True)):
                return ToolRun(summary="web search disabled by config", success=False, errors=["web search disabled"])
            query = str(action.args.get("query") or self._next_seed_query(state)).strip()
            if not query:
                return ToolRun(summary="web search skipped: no query", success=False, errors=["no search query"])
            state.queries.append(query)
            return self.search_tool.run(query, limit=int(self.web_cfg.get("max_results_per_query", self.agent_cfg.get("max_web_results", 6))))
        if action.tool_name == "web_read":
            item = self._select_read_target(state, str(action.args.get("url") or ""))
            if item is None:
                return ToolRun(summary="web read skipped: no unread URL", success=False, errors=["no unread URL"])
            state.read_urls.add(item.source)
            return self.read_tool.run(item.source, item.title, item.content[:500])
        if action.tool_name == "rank_evidence":
            return self.rank_tool.run(question, state.evidence, max_items=int(action.args.get("max_items") or 12))
        if action.tool_name == "review_evidence":
            enough, summary, missing = review_evidence(question, state.evidence)
            return ToolRun(
                evidence=[],
                summary=summary,
                success=enough,
                errors=[] if enough else [summary],
                metadata={"missing": missing, "evidence_count": len(state.evidence)},
            )
        if action.tool_name == "finish_answer":
            planner_answer = compact_text(str(action.args.get("answer") or "").strip(), 12000)
            if planner_answer:
                return ToolRun(
                    text=planner_answer,
                    summary="final answer accepted from planner finish_answer action",
                    metadata={"answer_source": "planner_finish_answer"},
                )
            evidence = state.evidence
            if state.tool_counts.get("rank_evidence", 0) == 0:
                evidence = self.rank_tool.run(question, evidence, max_items=12).evidence
                state.evidence = evidence
            return self.finish_tool.run(question, evidence, allow_llm=bool(action.args.get("allow_llm", True)))
        return ToolRun(summary=f"unsupported tool: {action.tool_name}", success=False, errors=[f"unsupported tool: {action.tool_name}"])

    def _guard_action(
        self,
        state: AgentState,
        action: AgentAction,
        start: float,
        max_total_seconds: float,
    ) -> str | None:
        if action.tool_name == "inspect_image":
            max_attempts = int(self.agent_cfg.get("max_image_inspect_attempts", 1))
            if state.tool_counts.get("inspect_image", 0) >= max_attempts:
                return "inspect_image retry budget exhausted; keep existing local image evidence and choose another action"
        if action.tool_name == "match_domain_skill":
            max_attempts = int(self.agent_cfg.get("max_domain_skill_calls", DEFAULT_MAX_DOMAIN_SKILL_CALLS))
            if state.tool_counts.get("match_domain_skill", 0) >= max_attempts:
                return "match_domain_skill already collected domain evidence; choose rank_evidence or finish_answer"
        if action.tool_name == "web_search":
            max_queries = int(self.agent_cfg.get("max_web_queries", 3))
            if state.tool_counts.get("web_search", 0) >= max_queries:
                return "web_search query budget exhausted"
        if action.tool_name == "web_read":
            max_pages = int(self.web_cfg.get("max_pages_to_read", self.agent_cfg.get("max_web_pages_to_read", 1)))
            if state.tool_counts.get("web_read", 0) >= max_pages:
                return "web_read page budget exhausted"
            requested_url = str(action.args.get("url") or "").strip()
            if requested_url and requested_url in state.read_urls:
                return "web_read URL already read; choose another action"
        if action.tool_name != "finish_answer" and not self._can_start_action(action.tool_name, start, max_total_seconds):
            return f"not enough time budget remaining to start {action.tool_name}"
        return None

    def _can_start_action(self, tool_name: str, start: float, max_total_seconds: float) -> bool:
        remaining = max_total_seconds - (time.perf_counter() - start)
        return remaining >= self._tool_timeout_seconds(tool_name) + 1.0

    def _tool_timeout_seconds(self, tool_name: str) -> float:
        default_timeout = float(self.agent_cfg.get("tool_timeout_seconds", self.config.raw.get("runtime", {}).get("request_timeout_seconds", 20)))
        if tool_name == "inspect_image":
            return float(self.agent_cfg.get("vision_timeout_seconds", default_timeout))
        if tool_name == "web_read":
            return float(self.agent_cfg.get("web_read_timeout_seconds", default_timeout))
        if tool_name == "finish_answer":
            return float(self.agent_cfg.get("final_answer_timeout_seconds", default_timeout))
        if tool_name == "agent_planner":
            return float(self.agent_cfg.get("planner_timeout_seconds", min(default_timeout, 15)))
        return default_timeout

    def _fallback_action(self, state: AgentState) -> AgentAction:
        if state.image_paths and state.tool_counts.get("inspect_image", 0) == 0 and bool(self.agent_cfg.get("use_images", True)):
            return AgentAction("inspect_image", reason="fallback: inspect available input images")
        if state.tool_counts.get("match_domain_skill", 0) == 0:
            return AgentAction("match_domain_skill", reason="fallback: collect domain skill evidence")
        max_queries = int(self.agent_cfg.get("max_web_queries", 3))
        if (
            bool(self.agent_cfg.get("enable_web_search", True))
            and state.tool_counts.get("web_search", 0) < max_queries
            and self._next_seed_query(state)
        ):
            return AgentAction(
                "web_search",
                args={"query": self._next_seed_query(state)},
                reason="fallback: search public evidence",
            )
        max_pages = int(self.web_cfg.get("max_pages_to_read", self.agent_cfg.get("max_web_pages_to_read", 1)))
        if state.tool_counts.get("web_read", 0) < max_pages and self._select_read_target(state, "") is not None:
            return AgentAction("web_read", reason="fallback: read strongest unread public page")
        if state.tool_counts.get("rank_evidence", 0) == 0 and state.evidence:
            return AgentAction("rank_evidence", reason="fallback: rank collected evidence")
        if state.tool_counts.get("review_evidence", 0) == 0:
            return AgentAction("review_evidence", reason="fallback: check evidence coverage")
        return AgentAction("finish_answer", reason="fallback: finalize answer from available evidence", stop=True)

    def _forced_action(self, state: AgentState) -> AgentAction | None:
        if not state.evidence or state.answer:
            return None
        if state.tool_counts.get("inspect_image", 0) == 0 and state.image_paths and bool(self.agent_cfg.get("use_images", True)):
            return None
        has_domain = any(item.metadata.get("kind") == "domain_skill" for item in state.evidence)
        has_web = any(item.metadata.get("kind") in {"web_search_result", "web_page"} for item in state.evidence)
        has_image = any(item.metadata.get("kind") in {"image_context", "image_inspection"} for item in state.evidence)
        enough_for_answer = (has_image and has_domain) or has_web or len(state.evidence) >= 4
        if not enough_for_answer:
            return None
        if state.tool_counts.get("rank_evidence", 0) == 0:
            return AgentAction("rank_evidence", reason="runtime: evidence is sufficient; rank before final answer")
        return AgentAction(
            "finish_answer",
            reason="runtime: ranked evidence is sufficient; synthesize final answer",
            stop=True,
        )

    def _domain_skill_question(self, question: str, action: AgentAction) -> str:
        extra_parts = []
        for key in ("query", "topic", "details"):
            value = action.args.get(key)
            if value:
                extra_parts.append(str(value))
        if action.reason:
            extra_parts.append(action.reason)
        if not extra_parts:
            return question
        return question + "\n" + "\n".join(extra_parts)

    def _dedupe_evidence(self, evidence: list[Evidence]) -> list[Evidence]:
        seen: set[str] = set()
        kept: list[Evidence] = []
        for item in evidence:
            kind = str(item.metadata.get("kind") or "")
            key = "\n".join(
                [
                    str(item.source or "").strip().lower(),
                    kind.strip().lower(),
                    compact_text(str(item.content or ""), 240).strip().lower(),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            kept.append(item)
        return kept

    def _next_seed_query(self, state: AgentState) -> str:
        max_queries = int(self.agent_cfg.get("max_web_queries", 3))
        seed_queries = build_seed_queries(state.sample.question_text, state.evidence, max_queries=max_queries)
        used = set(state.queries)
        for query in seed_queries:
            if query not in used:
                return query
        return ""

    def _select_read_target(self, state: AgentState, requested_url: str) -> Evidence | None:
        candidates = [
            item
            for item in state.evidence
            if item.source.startswith(("http://", "https://")) and item.source not in state.read_urls
        ]
        if requested_url:
            for item in candidates:
                if item.source == requested_url:
                    return item
            return Evidence(source=requested_url, title=requested_url, content="", metadata={"kind": "web_search_result"})
        ranked = self.rank_tool.run(state.sample.question_text, candidates, max_items=1).evidence
        return ranked[0] if ranked else None

    def _event_for_action(
        self,
        state: AgentState,
        action: AgentAction,
        run: ToolRun,
        elapsed: float,
        before_evidence: int,
        observation: AgentObservation,
        start: float,
        max_total_seconds: float,
    ) -> ToolEvent:
        tool_name, action_name = self._tool_event_names(action.tool_name)
        evidence_delta = len(state.evidence) - before_evidence
        outputs = {
            "sources": [item.source for item in run.evidence],
            "metadata": run.metadata,
            "iteration": state.iteration,
            "action": action.tool_name,
            "validated_action": action.to_json(),
            "observation": observation.to_json(),
            "evidence_delta": evidence_delta,
            "recoverable_error": observation.recoverable and not observation.success,
            "budget_remaining": self._budget_remaining(start, max_total_seconds),
        }
        if action.tool_name == "review_evidence":
            outputs.update(run.metadata)
        return ToolEvent(
            tool_name=tool_name,
            action=action_name,
            success=run.success,
            elapsed_seconds=elapsed,
            summary=run.summary,
            inputs=self._inputs_for_action(state, action),
            outputs=outputs,
            error="; ".join(run.errors) if run.errors else None,
        )

    def _planner_event(
        self,
        state: AgentState,
        raw_action: dict[str, Any],
        action: AgentAction | None,
        observation: AgentObservation,
        start: float,
        max_total_seconds: float,
        success: bool,
        elapsed: float = 0.0,
    ) -> ToolEvent:
        return ToolEvent(
            tool_name="agent_planner",
            action="select_action",
            success=success,
            elapsed_seconds=elapsed,
            summary=observation.summary if not success else f"selected {action.tool_name if action else '<none>'}",
            inputs={
                "sample_id": state.sample.sample_id,
                "iteration": state.iteration,
                "allowed_tools": sorted(ALLOWED_TOOLS),
            },
            outputs={
                "iteration": state.iteration,
                "action": action.tool_name if action else None,
                "raw_action": raw_action,
                "validated_action": action.to_json() if action else None,
                "observation": observation.to_json(),
                "evidence_count": len(state.evidence),
                "budget_remaining": self._budget_remaining(start, max_total_seconds),
            },
            error=observation.error,
        )

    def _guard_event(
        self,
        state: AgentState,
        action: AgentAction,
        observation: AgentObservation,
        start: float,
        max_total_seconds: float,
    ) -> ToolEvent:
        return ToolEvent(
            tool_name="agent_runtime",
            action="skip_action",
            success=False,
            summary=observation.summary,
            inputs={
                "sample_id": state.sample.sample_id,
                "iteration": state.iteration,
                "action": action.to_json(),
            },
            outputs={
                "iteration": state.iteration,
                "validated_action": action.to_json(),
                "observation": observation.to_json(),
                "recoverable_error": True,
                "budget_remaining": self._budget_remaining(start, max_total_seconds),
            },
            error=observation.error,
        )

    def _inputs_for_action(self, state: AgentState, action: AgentAction) -> dict[str, Any]:
        if action.tool_name == "inspect_image":
            return {"sample_id": state.sample.sample_id, "images": state.image_paths, "reason": action.reason}
        if action.tool_name == "match_domain_skill":
            return {"sample_id": state.sample.sample_id, "reason": action.reason}
        if action.tool_name == "web_search":
            return {
                "query": action.args.get("query") or (state.queries[-1] if state.queries else ""),
                "limit": int(self.web_cfg.get("max_results_per_query", self.agent_cfg.get("max_web_results", 6))),
                "reason": action.reason,
            }
        if action.tool_name == "web_read":
            return {"url": action.args.get("url") or "", "reason": action.reason}
        return {"sample_id": state.sample.sample_id, "reason": action.reason}

    def _tool_event_names(self, tool_name: str) -> tuple[str, str]:
        mapping = {
            "inspect_image": ("image_inspect", "multimodal_component_extract"),
            "match_domain_skill": ("domain_skill", "match_electronics_skills"),
            "web_search": ("web_search", "api_or_html_search"),
            "web_read": ("web_reader", "read_or_keep_snippet"),
            "rank_evidence": ("evidence_rank", "rank_and_dedupe"),
            "review_evidence": ("circuit_reviewer", "coverage_check"),
            "finish_answer": ("finish_answer", "synthesize_final_answer"),
        }
        return mapping[tool_name]

    def _planner_state_payload(self, state: AgentState, start: float, max_total_seconds: float) -> dict[str, Any]:
        return {
            "question": state.sample.question_text,
            "sample_id": state.sample.sample_id,
            "has_images": bool(state.image_paths),
            "image_count": len(state.image_paths),
            "allowed_tools": sorted(ALLOWED_TOOLS),
            "tool_counts": state.tool_counts,
            "evidence": [
                {
                    "idx": idx,
                    "source": item.source,
                    "title": item.title,
                    "content": compact_text(item.content, 500),
                    "score": item.score,
                    "kind": item.metadata.get("kind"),
                }
                for idx, item in enumerate(state.evidence[-12:], 1)
            ],
            "recent_observations": [self._compact_observation_for_planner(item) for item in state.observations[-5:]],
            "errors": state.errors[-5:],
            "read_urls": sorted(state.read_urls),
            "budget_remaining": self._budget_remaining(start, max_total_seconds),
            "guidance": planner_guidance(),
        }

    def _compact_observation_for_planner(self, observation: dict[str, Any]) -> dict[str, Any]:
        evidence = observation.get("evidence") or []
        compact_evidence = []
        if isinstance(evidence, list):
            for item in evidence[:3]:
                if not isinstance(item, dict):
                    continue
                compact_evidence.append(
                    {
                        "source": item.get("source"),
                        "title": compact_text(str(item.get("title") or ""), 120),
                        "kind": (item.get("metadata") or {}).get("kind") if isinstance(item.get("metadata"), dict) else None,
                        "content": compact_text(str(item.get("content") or ""), 240),
                    }
                )
        return {
            "success": bool(observation.get("success")),
            "summary": compact_text(str(observation.get("summary") or ""), 240),
            "error": compact_text(str(observation.get("error") or ""), 240) if observation.get("error") else None,
            "recoverable": bool(observation.get("recoverable", True)),
            "evidence": compact_evidence,
        }

    def _budget_remaining(self, start: float, max_total_seconds: float) -> dict[str, Any]:
        elapsed = time.perf_counter() - start
        return {
            "seconds": round(max(0.0, max_total_seconds - elapsed), 4),
            "max_total_seconds": max_total_seconds,
        }

    def _add_bootstrap_events(self, trace: TraceLogger, sample: StandardSample, plan: AgentPlan) -> None:
        trace.add(
            ToolEvent(
                tool_name="agent_planner",
                action="initialize_loop",
                success=True,
                summary=f"initialized agentic loop for {plan.question_type}",
                inputs={"sample_id": sample.sample_id},
                outputs={
                    "strategy": plan.strategy,
                    "question_type": plan.question_type,
                    "needs_images": plan.needs_images,
                    "needs_web_search": plan.needs_web_search,
                    "budgets": plan.budgets,
                    "models": {
                        "planner": self.config.agent_model,
                        "vision": self.config.vision_model,
                        "answer": self.config.agent_model,
                    },
                    "tool_contract": "OpenHands SDK ToolDefinition compatibility layer",
                },
            )
        )
        if self.sdk_status:
            trace.add(
                ToolEvent(
                    tool_name="openhands_sdk",
                    action="load_contracts",
                    success=True,
                    summary=self.sdk_status,
                )
            )

    def _skill_path(self) -> Path:
        raw_path = self.agent_cfg.get("domain_skills_path") or self.config.local_corpus_root / "domain_skills.yaml"
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    def _classify(self, question: str) -> str:
        upper = question.upper()
        if any(term in upper for term in ["LLC", "DCDC", "TL431", "MOS", "PFC"]) or "鐢垫簮" in question:
            return "power_supply"
        if any(term in question for term in ["璋冨厜", "杩愭斁", "LM358"]):
            return "analog_driver_debugging"
        if any(term in question for term in ["璐熸帶", "姝ｆ帶", "淇濇姢鏉?"]):
            return "battery_protection_switching"
        return "general_component_anomaly"

    def _inspect_openhands_sdk(self) -> str:
        try:
            os.environ.setdefault("OPENHANDS_SUPPRESS_BANNER", "1")
            import openhands.sdk as sdk

            return f"OpenHands SDK contracts loaded: {getattr(sdk, '__version__', 'unknown')}"
        except Exception as exc:  # noqa: BLE001
            return f"OpenHands SDK unavailable, using local compatible contracts: {exc}"

    def _planner_error_type(self, error: str) -> str:
        lowered = error.lower()
        if "invalid_request" in lowered or "invalidparameter" in lowered or "400" in lowered:
            return "invalid_request"
        if "timeout" in lowered or "timed out" in lowered:
            return "timeout"
        if "connection" in lowered:
            return "connection_error"
        return "planner_error"

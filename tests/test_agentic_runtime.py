from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import unittest
from uuid import uuid4

from agent.openhands_runtime import AgentState, OpenHandsEvidenceRuntime
from agent.prompts import PLANNER_SYSTEM_PROMPT
from agent.tool_registry import DEFAULT_TOOL_REGISTRY
from configs.config import RuntimeConfig
from schemas import ImageRef, StandardSample
from tools.evidence_tools import ToolRun
from schemas import Evidence
from tools.circuit_kb import build_circuit_md_kb


class ScriptedPlanner:
    available = True

    def __init__(self, actions: list[dict]) -> None:
        self.actions = actions
        self.calls = 0

    def json_chat(self, messages, temperature=0.0, response_format=None):
        if self.calls >= len(self.actions):
            action = {"tool_name": "finish_answer", "args": {}, "reason": "script fallback", "stop": True}
        else:
            action = self.actions[self.calls]
        self.calls += 1
        return action, None


class FakeImageTool:
    def run(self, action):
        return ToolRun(
            evidence=[
                Evidence(
                    source="local_images",
                    title="image context",
                    content="image paths retained after timeout",
                    metadata={"kind": "image_context"},
                )
            ],
            summary="image paths recorded; vision inspection failed",
            success=False,
            errors=["Request timed out."],
        )


class FakeSearchTool:
    def __init__(self, evidence: list[Evidence] | None = None) -> None:
        self.calls: list[dict] = []
        self.evidence = evidence or [
            Evidence(
                source="https://example.com/tl431",
                title="TL431 feedback note",
                content="TL431 optocoupler ripple troubleshooting",
                metadata={"kind": "web_search_result"},
            )
        ]

    def run(self, query, limit=6):
        self.calls.append({"query": query, "limit": limit})
        return ToolRun(evidence=list(self.evidence), summary="fake search ok")


class FakeReadTool:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, url, title="", snippet=""):
        self.calls.append({"url": url, "title": title, "snippet": snippet})
        return ToolRun(
            evidence=[
                Evidence(
                    source=url,
                    title=title or url,
                    content=snippet or "page content",
                    metadata={"kind": "web_page"},
                )
            ],
            summary="fake read ok",
        )


class FakeFinishTool:
    def __init__(self, answer: str = "fallback synthesized answer") -> None:
        self.calls: list[dict] = []
        self.answer = answer

    def run(self, question, evidence, allow_llm=True):
        self.calls.append({"question": question, "allow_llm": allow_llm, "evidence_count": len(evidence)})
        return ToolRun(text=self.answer, summary="fake finish ok")


class AgenticRuntimeTests(unittest.TestCase):
    def test_prompt_and_registry_tool_names_stay_in_sync(self) -> None:
        for tool_name in DEFAULT_TOOL_REGISTRY.planner_tool_names():
            self.assertIn(tool_name, PLANNER_SYSTEM_PROMPT)

    def test_planner_guidance_limits_local_retrieval_to_strong_gaps(self) -> None:
        from agent.prompts import planner_guidance

        guidance = planner_guidance(["inspect_image", "local_retrieve", "finish_answer"])

        self.assertIn("诊断信息", guidance)
        self.assertIn("KB 证据不会进入最终答案", guidance)

    def test_disabled_or_missing_resource_tools_are_not_exposed_to_planner(self) -> None:
        with self._runtime(
            agent_overrides={"enable_domain_skills": False, "enable_local_retrieval": True}
        ) as runtime:
            self.assertNotIn("match_domain_skill", runtime.enabled_tool_names)
            self.assertNotIn("local_retrieve", runtime.enabled_tool_names)
            self.assertNotIn("match_domain_skill", runtime.planner_system_prompt)
            self.assertNotIn("local_retrieve", runtime.planner_system_prompt)

    def test_valid_local_kb_resource_is_exposed_to_planner(self) -> None:
        with self._runtime(agent_overrides={"enable_local_retrieval": True}, create_local_kb=True) as runtime:
            self.assertIn("local_retrieve", runtime.enabled_tool_names)
            self.assertIn("local_retrieve", runtime.planner_system_prompt)
            self.assertTrue(runtime.local_kb_status.usable)

    def test_scripted_valid_actions_drive_loop(self) -> None:
        with self._runtime() as runtime:
            runtime.llm = ScriptedPlanner(
                [
                    {"tool_name": "match_domain_skill", "args": {}, "reason": "need domain prior"},
                    {"tool_name": "rank_evidence", "args": {}, "reason": "rank evidence"},
                    {"tool_name": "finish_answer", "args": {}, "reason": "enough evidence", "stop": True},
                ]
            )

            result = runtime.run_sample(self._sample("LED feedback troubleshooting"))

        self.assertTrue(result.answer)
        self.assertEqual(result.plan.strategy, "agentic_tool_loop")
        self.assertEqual([item["tool_name"] for item in result.plan.selected_actions[:3]], ["match_domain_skill", "rank_evidence", "finish_answer"])

    def test_invalid_planner_action_is_recorded_and_loop_recovers(self) -> None:
        with self._runtime() as runtime:
            runtime.llm = ScriptedPlanner(
                [
                    {"tool_name": "not_a_tool", "args": {}, "reason": "bad"},
                    {"tool_name": "match_domain_skill", "args": {}, "reason": "recover"},
                    {"tool_name": "finish_answer", "args": {}, "reason": "done", "stop": True},
                ]
            )

            result = runtime.run_sample(self._sample("LED feedback troubleshooting"))

        planner_failures = [
            event
            for event in result.tool_trace
            if event.tool_name == "agent_planner" and event.action == "select_action" and not event.success
        ]
        self.assertEqual(len(planner_failures), 1)
        self.assertIn("unknown planner tool", planner_failures[0].error)
        self.assertTrue(result.answer)

    def test_vision_timeout_is_recoverable_and_does_not_block_finish(self) -> None:
        with _workspace_tempdir() as tmp:
            image_path = Path(tmp) / "board.png"
            image_path.write_bytes(b"fake image")
            with self._runtime(image_root=Path(tmp)) as runtime:
                runtime.llm = ScriptedPlanner(
                    [
                        {"tool_name": "inspect_image", "args": {}, "reason": "need image"},
                        {"tool_name": "finish_answer", "args": {}, "reason": "degrade after vision error", "stop": True},
                    ]
                )
                runtime.image_tool = FakeImageTool()
                sample = self._sample(
                    "LED board issue",
                    images=[ImageRef(original_url="board.png", path=image_path, exists=True)],
                )

                result = runtime.run_sample(sample)

        image_events = [event for event in result.tool_trace if event.tool_name == "image_inspect"]
        self.assertEqual(len(image_events), 1)
        self.assertFalse(image_events[0].success)
        self.assertTrue(image_events[0].outputs["recoverable_error"])
        self.assertTrue(result.answer)

    def test_repeated_failed_image_inspection_is_skipped_by_runtime_guard(self) -> None:
        with _workspace_tempdir() as tmp:
            image_path = Path(tmp) / "board.png"
            image_path.write_bytes(b"fake image")
            with self._runtime(image_root=Path(tmp)) as runtime:
                runtime.llm = ScriptedPlanner(
                    [
                        {"tool_name": "inspect_image", "args": {}, "reason": "need image"},
                        {"tool_name": "inspect_image", "args": {}, "reason": "try again"},
                        {"tool_name": "finish_answer", "args": {}, "reason": "done", "stop": True},
                    ]
                )
                runtime.image_tool = FakeImageTool()
                sample = self._sample(
                    "LED board issue",
                    images=[ImageRef(original_url="board.png", path=image_path, exists=True)],
                )

                result = runtime.run_sample(sample)

        image_events = [event for event in result.tool_trace if event.tool_name == "image_inspect"]
        guard_events = [event for event in result.tool_trace if event.tool_name == "agent_runtime" and event.action == "skip_action"]
        self.assertEqual(len(image_events), 1)
        self.assertEqual(len(guard_events), 1)
        self.assertIn("retry budget exhausted", guard_events[0].error)

    def test_repeated_domain_skill_action_is_not_executed_twice(self) -> None:
        with self._runtime() as runtime:
            runtime.llm = ScriptedPlanner(
                [
                    {"tool_name": "match_domain_skill", "args": {"topic": "LED feedback"}, "reason": "need domain"},
                    {"tool_name": "match_domain_skill", "args": {"topic": "LED feedback again"}, "reason": "repeat"},
                    {"tool_name": "finish_answer", "args": {}, "reason": "done", "stop": True},
                ]
            )

            result = runtime.run_sample(self._sample("LED feedback troubleshooting"))

        domain_events = [event for event in result.tool_trace if event.tool_name == "domain_skill"]
        self.assertEqual(len(domain_events), 1)
        self.assertTrue(result.answer)

    def test_fallback_does_not_use_local_retrieval_for_image_only_board_questions(self) -> None:
        with self._runtime(
            agent_overrides={"enable_domain_skills": False, "enable_web_search": False, "enable_local_retrieval": True},
            create_local_kb=True,
        ) as runtime:
            sample = self._sample(
                "请根据图片判断这个板子的供电连接和拓扑是否正确",
                images=[ImageRef(original_url="board.png", path=Path("board.png"), exists=True)],
            )
            state = AgentState(sample=sample, image_paths=["board.png"])
            state.tool_counts["inspect_image"] = 1
            state.evidence.append(
                Evidence(
                    source="local_images",
                    title="image inspection",
                    content="图片显示板级供电连接和若干电感。",
                    metadata={"kind": "image_inspection"},
                )
            )

            action = runtime._fallback_action(state)

        self.assertEqual(action.tool_name, "rank_evidence")

    def test_fallback_ranks_existing_evidence_before_local_retrieval_even_with_strong_terms(self) -> None:
        with self._runtime(
            agent_overrides={"enable_domain_skills": False, "enable_web_search": False, "enable_local_retrieval": True},
            create_local_kb=True,
        ) as runtime:
            sample = self._sample("MC34063 buck LED 电流采样纹波怎么处理")
            state = AgentState(sample=sample, image_paths=[])
            state.evidence.append(
                Evidence(
                    source="domain",
                    title="current sense prior",
                    content="已有电流采样和纹波处理证据。",
                    metadata={"kind": "domain_skill"},
                )
            )

            action = runtime._fallback_action(state)

        self.assertEqual(action.tool_name, "rank_evidence")

    def test_planner_state_hides_local_retrieval_when_current_sample_is_not_eligible(self) -> None:
        with self._runtime(
            agent_overrides={"enable_domain_skills": False, "enable_web_search": False, "enable_local_retrieval": True},
            create_local_kb=True,
        ) as runtime:
            sample = self._sample(
                "请根据图片判断这个板子的供电连接和拓扑是否正确",
                images=[ImageRef(original_url="board.png", path=Path("board.png"), exists=True)],
            )
            state = AgentState(sample=sample, image_paths=["board.png"])
            state.tool_counts["inspect_image"] = 1
            state.evidence.append(
                Evidence(
                    source="local_images",
                    title="image inspection",
                    content="图片显示板级供电连接。",
                    metadata={"kind": "image_inspection"},
                )
            )

            allowed = runtime._effective_tool_names(state)
            payload = runtime._planner_state_payload(state, start=0.0, max_total_seconds=20.0, effective_tool_names=allowed)

        self.assertNotIn("local_retrieve", allowed)
        self.assertNotIn("local_retrieve", payload["allowed_tools"])

    def test_web_search_query_is_repaired_and_limit_is_clipped(self) -> None:
        with self._runtime(
            agent_overrides={"enable_web_search": True},
            web_overrides={"max_results_per_query": 2},
        ) as runtime:
            runtime.llm = ScriptedPlanner(
                [
                    {"tool_name": "web_search", "args": {"query": "原因 处理", "limit": 99}, "reason": "need public evidence"},
                    {"tool_name": "finish_answer", "args": {}, "reason": "done", "stop": True},
                ]
            )
            fake_search = FakeSearchTool()
            runtime.search_tool = fake_search

            result = runtime.run_sample(self._sample("TL431 光耦反馈纹波异常怎么处理"))

        self.assertEqual(fake_search.calls[0]["limit"], 2)
        self.assertIn("TL431", fake_search.calls[0]["query"])
        planner_event = next(event for event in result.tool_trace if event.tool_name == "agent_planner" and event.action == "select_action")
        self.assertTrue(planner_event.outputs["repair_notes"]["repair_applied"])
        self.assertEqual(result.plan.selected_actions[0]["repaired"], True)

    def test_web_read_missing_url_selects_best_candidate(self) -> None:
        with self._runtime(
            agent_overrides={"enable_web_search": True},
            web_overrides={"max_results_per_query": 2},
        ) as runtime:
            runtime.llm = ScriptedPlanner(
                [
                    {"tool_name": "web_search", "args": {"query": ""}, "reason": "search"},
                    {"tool_name": "web_read", "args": {}, "reason": "read strongest page"},
                    {"tool_name": "finish_answer", "args": {}, "reason": "done", "stop": True},
                ]
            )
            runtime.search_tool = FakeSearchTool(
                [
                    Evidence(
                        source="https://example.com/appnote",
                        title="Application note",
                        content="current sense spike filtering details",
                        metadata={"kind": "web_search_result"},
                    )
                ]
            )
            fake_read = FakeReadTool()
            runtime.read_tool = fake_read
            runtime._forced_action = lambda state: None

            result = runtime.run_sample(self._sample("电流采样尖峰和滤波怎么处理"))

        self.assertEqual(fake_read.calls[0]["url"], "https://example.com/appnote")
        self.assertEqual(result.plan.selected_actions[1]["repaired"], True)
        web_read_event = next(event for event in result.tool_trace if event.tool_name == "web_reader")
        self.assertEqual(web_read_event.outputs["effective_action"]["args"]["url"], "https://example.com/appnote")

    def test_blank_finish_answer_falls_back_to_finish_tool_and_trace_keeps_all_action_forms(self) -> None:
        with self._runtime() as runtime:
            runtime.llm = ScriptedPlanner(
                [
                    {"tool_name": "match_domain_skill", "args": {}, "reason": "get prior"},
                    {"tool_name": "finish_answer", "args": {"answer": ""}, "reason": "done", "stop": True},
                ]
            )
            fake_finish = FakeFinishTool(answer="synthesized from evidence")
            runtime.finish_tool = fake_finish

            result = runtime.run_sample(self._sample("LED feedback troubleshooting"))

        self.assertEqual(result.answer, "synthesized from evidence")
        self.assertEqual(fake_finish.calls[0]["evidence_count"], 1)
        finish_planner_event = [
            event
            for event in result.tool_trace
            if event.tool_name == "agent_planner" and event.outputs.get("action") == "finish_answer"
        ][0]
        self.assertIn("raw_action", finish_planner_event.outputs)
        self.assertIn("validated_action", finish_planner_event.outputs)
        self.assertIn("effective_action", finish_planner_event.outputs)
        self.assertTrue(result.plan.selected_actions[-1]["repaired"])

    def test_trace_payload_written_as_parseable_json(self) -> None:
        from scripts.run_infer import _write_trace

        with _workspace_tempdir() as tmp:
            trace_dir = Path(tmp) / "traces"
            row = {
                "sample_id": "sample",
                "question": "bad\x00question",
                "answer": "answer",
                "tools_used": [],
                "web_searched": False,
                "elapsed_seconds": 1.0,
                "token_usage": {},
                "errors": [],
                "tool_trace": [],
                "plan": {"strategy": "agentic_tool_loop"},
            }
            _write_trace(trace_dir, row)
            payload = json.loads((trace_dir / "sample.trace.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["sample_id"], "sample")
        self.assertIn("\ufffd", payload["question"])

    def _runtime(
        self,
        image_root: Path | None = None,
        agent_overrides: dict | None = None,
        web_overrides: dict | None = None,
        create_local_kb: bool = False,
    ):
        return _RuntimeContext(
            image_root=image_root,
            agent_overrides=agent_overrides,
            web_overrides=web_overrides,
            create_local_kb=create_local_kb,
        )

    def _sample(self, question: str, images: list[ImageRef] | None = None) -> StandardSample:
        return StandardSample(
            sample_id="sample",
            post_id="sample",
            question_text=question,
            images=images or [],
            reference_answer="",
            raw_messages=[],
        )


@contextmanager
def _workspace_tempdir():
    base = Path(__file__).resolve().parents[1] / "outputs" / "test_agentic_runtime"
    base.mkdir(parents=True, exist_ok=True)
    path = base / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)


class _RuntimeContext:
    def __init__(
        self,
        image_root: Path | None = None,
        agent_overrides: dict | None = None,
        web_overrides: dict | None = None,
        create_local_kb: bool = False,
    ) -> None:
        self.temp_dir = _workspace_tempdir()
        self.root: Path | None = None
        self.image_root_override = image_root
        self.image_root: Path | None = None
        self.runtime: OpenHandsEvidenceRuntime | None = None
        self.agent_overrides = agent_overrides or {}
        self.web_overrides = web_overrides or {}
        self.create_local_kb = create_local_kb

    def __enter__(self) -> OpenHandsEvidenceRuntime:
        self.root = Path(self.temp_dir.__enter__())
        self.image_root = self.image_root_override or self.root / "images"
        local_corpus = self.root / "knowledge_base"
        local_corpus.mkdir(parents=True)
        (local_corpus / "domain_skills.yaml").write_text(
            """
skills:
  - id: led_feedback
    title: LED feedback skill
    triggers: ["LED", "feedback"]
    query_terms: ["LED feedback circuit troubleshooting"]
    content: "Check feedback, supply, drive margin and load path."
  - id: current_sense
    title: Current sense skill
    triggers: ["current sense", "SENSE", "采样"]
    query_terms: ["current sense filter"]
    content: "Check current sense filtering and leading edge noise."
""",
            encoding="utf-8",
        )
        local_kb_index = local_corpus / "circuit_md_fts"
        if self.create_local_kb:
            source = self.root / "circuit_pages"
            source.mkdir()
            (source / "page_1.md").write_text(
                (
                    """
**标题**: MC34063 buck LED constant current driver

**链接**: https://example.com/mc34063

**发布时间**: 未知

MC34063 buck LED constant current driver uses current sense feedback and inductor switching.
""".strip()
                    + "\n\n"
                    + "MC34063 buck LED constant current driver uses current sense feedback and inductor switching. " * 8
                ),
                encoding="utf-8",
            )
            build_circuit_md_kb(source, local_kb_index, max_docs=10, min_chunk_chars=120)
        dataset_path = self.root / "dataset.jsonl"
        dataset_path.write_text("", encoding="utf-8")
        self.image_root.mkdir(parents=True, exist_ok=True)
        raw = {
            "paths": {
                "dataset": str(dataset_path),
                "image_root": str(self.image_root),
                "local_corpus_root": str(local_corpus),
                "local_kb_index": str(local_kb_index),
                "outputs_dir": str(self.root / "outputs"),
                "logs_dir": str(self.root / "logs"),
            },
            "model": {
                "api_key_env": "__NO_SUCH_AGENTIC_RUNTIME_KEY__",
                "base_url_env": "__NO_SUCH_AGENTIC_RUNTIME_BASE__",
                "default_base_url": "https://example.invalid/v1",
                "agent_model_env": "__NO_SUCH_AGENTIC_RUNTIME_MODEL__",
                "default_agent_model": "test-model",
                "judge_model_env": "__NO_SUCH_AGENTIC_RUNTIME_JUDGE__",
                "default_judge_model": "test-judge",
                "temperature": 0.0,
                "max_tokens": 200,
            },
            "agent": {
                "strategy": "agentic_tool_loop",
                "max_iterations": 5,
                "max_total_seconds": 20,
                "tool_timeout_seconds": 1,
                "vision_timeout_seconds": 1,
                "web_read_timeout_seconds": 1,
                "max_consecutive_tool_errors": 3,
                "enable_web_search": False,
                "use_images": True,
                "send_images_to_llm": True,
                "enable_image_inspection_llm": True,
                "max_llm_images": 1,
            },
            "web": {
                "provider_order": ["html"],
                "api_key_envs": {},
                "api_keys": {},
                "max_results_per_query": 2,
                "max_pages_to_read": 1,
            },
            "runtime": {"request_timeout_seconds": 1},
        }
        raw["agent"].update(self.agent_overrides)
        raw["web"].update(self.web_overrides)
        self.runtime = OpenHandsEvidenceRuntime(RuntimeConfig(raw))
        return self.runtime

    def __exit__(self, exc_type, exc, tb) -> None:
        self.temp_dir.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    unittest.main()

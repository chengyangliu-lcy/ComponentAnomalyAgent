from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
import unittest
from uuid import uuid4

from agent.openhands_runtime import OpenHandsEvidenceRuntime
from configs.config import RuntimeConfig
from schemas import ImageRef, StandardSample
from tools.evidence_tools import ToolRun
from schemas import Evidence


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


class AgenticRuntimeTests(unittest.TestCase):
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

    def _runtime(self, image_root: Path | None = None):
        return _RuntimeContext(image_root=image_root)

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
    def __init__(self, image_root: Path | None = None) -> None:
        self.temp_dir = _workspace_tempdir()
        self.root: Path | None = None
        self.image_root_override = image_root
        self.image_root: Path | None = None
        self.runtime: OpenHandsEvidenceRuntime | None = None

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
        dataset_path = self.root / "dataset.jsonl"
        dataset_path.write_text("", encoding="utf-8")
        self.image_root.mkdir(parents=True, exist_ok=True)
        raw = {
            "paths": {
                "dataset": str(dataset_path),
                "image_root": str(self.image_root),
                "local_corpus_root": str(local_corpus),
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
        self.runtime = OpenHandsEvidenceRuntime(RuntimeConfig(raw))
        return self.runtime

    def __exit__(self, exc_type, exc, tb) -> None:
        self.temp_dir.__exit__(exc_type, exc, tb)


if __name__ == "__main__":
    unittest.main()

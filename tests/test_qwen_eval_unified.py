from __future__ import annotations

import unittest

from qwen_eval import (
    _dedupe_predictions,
    _filter_retry_failed_samples,
    _prediction_payload,
    _qwen_compatible_evaluation,
    is_hard_failed_prediction,
    is_successful_prediction,
)


class QwenEvalUnifiedTests(unittest.TestCase):
    def test_qwen_compatible_evaluation_uses_unified_judge(self) -> None:
        row = {
            "llm_judge": {
                "accuracy": 4,
                "completeness": 3,
                "clarity": 5,
                "usefulness": 4,
                "average_score": 4.0,
                "score": 0.75,
                "factual_consistency": 0.8,
            }
        }

        compat = _qwen_compatible_evaluation(row)

        self.assertEqual(compat["accuracy"], 4)
        self.assertEqual(compat["average_score"], 4.0)
        self.assertEqual(compat["unified_score"], 0.75)
        self.assertEqual(compat["factual_consistency"], 0.8)

    def test_error_answer_is_not_successful_prediction(self) -> None:
        row = {
            "sample_id": "1",
            "question": "q",
            "answer": "[ERROR] 生成失败",
            "errors": ["baseline generation failed"],
        }

        self.assertTrue(is_hard_failed_prediction(row))
        self.assertFalse(is_successful_prediction(row))

    def test_prediction_payload_matches_agent_shape(self) -> None:
        payload = _prediction_payload(
            {
                "sample_id": "1",
                "question": "问题",
                "answer": "答案",
                "errors": [],
            }
        )

        for key in [
            "sample_id",
            "question",
            "answer",
            "tools_used",
            "web_searched",
            "tool_trace",
            "reasoning_summary",
            "elapsed_seconds",
            "token_usage",
            "errors",
            "plan",
        ]:
            self.assertIn(key, payload)
        self.assertEqual(payload["answer"], "答案")

    def test_dedupe_predictions_prefers_latest_success_over_failure(self) -> None:
        rows = [
            {"sample_id": "1", "answer": "[ERROR] 生成失败", "errors": ["failed"]},
            {"sample_id": "2", "answer": "ok", "errors": []},
            {"sample_id": "1", "answer": "new", "errors": []},
        ]

        latest = _dedupe_predictions(rows)

        self.assertEqual(len(latest), 2)
        self.assertEqual(latest[0]["answer"], "new")
        self.assertEqual(latest[1]["answer"], "ok")

    def test_retry_failed_only_with_no_failed_ids_selects_no_samples(self) -> None:
        samples = [{"post_id": "1"}, {"post_id": "2"}]

        selected = _filter_retry_failed_samples(samples, [])

        self.assertEqual(selected, [])

    def test_retry_failed_only_selects_failed_ids(self) -> None:
        samples = [{"post_id": "1"}, {"post_id": "2"}]

        selected = _filter_retry_failed_samples(samples, ["2"])

        self.assertEqual(selected, [{"post_id": "2"}])


if __name__ == "__main__":
    unittest.main()

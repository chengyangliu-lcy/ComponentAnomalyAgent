from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

from evaluator.baseline_compare import compare_runs
from scripts.compare_runs import _check_sample_sets, _kb_bucket_report, _sample_set_report, _read_jsonl


class CompareRunsTests(unittest.TestCase):
    def test_duplicate_sample_ids_error_by_default(self) -> None:
        path = Path("outputs") / "test_compare" / f"{uuid4().hex}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sample_id":"a"}\n{"sample_id":"a"}\n', encoding="utf-8")

        with self.assertRaises(SystemExit):
            _read_jsonl(path, duplicates="error", label="test")

    def test_duplicate_sample_ids_can_keep_last(self) -> None:
        path = Path("outputs") / "test_compare" / f"{uuid4().hex}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"sample_id":"a","score":1}\n{"sample_id":"a","score":2}\n', encoding="utf-8")

        rows = _read_jsonl(path, duplicates="keep-last", label="test")

        self.assertEqual(rows, [{"sample_id": "a", "score": 2}])

    def test_sample_set_mismatch_errors_by_default(self) -> None:
        with self.assertRaises(SystemExit):
            _check_sample_sets([{"sample_id": "a"}], [{"sample_id": "b"}], mode="error")

    def test_sample_set_report_lists_missing_ids(self) -> None:
        report = _sample_set_report([{"sample_id": "a"}], [{"sample_id": "a"}, {"sample_id": "b"}])

        self.assertEqual(report["agent_samples"], 1)
        self.assertEqual(report["baseline_samples"], 2)
        self.assertEqual(report["shared_samples"], 1)
        self.assertEqual(report["missing_in_agent"], ["b"])
        self.assertEqual(report["missing_in_baseline"], [])

    def test_compare_runs_includes_metric_deltas(self) -> None:
        baseline = [
            {
                "sample_id": "a",
                "final_score": 0.5,
                "llm_judge": {"score": 0.4},
                "semantic_similarity": {"score": 0.7},
                "claim_rouge_l": {"score": 0.2},
                "technical_entity_match": {"score": 0.3},
            }
        ]
        agent = [
            {
                "sample_id": "a",
                "final_score": 0.6,
                "llm_judge": {"score": 0.7},
                "semantic_similarity": {"score": 0.69},
                "claim_rouge_l": {"score": 0.25},
                "technical_entity_match": {"score": 0.35},
            }
        ]

        report = compare_runs(baseline, agent)

        self.assertEqual(report["metric_deltas"]["llm_judge"], 0.3)
        self.assertEqual(report["metric_deltas"]["semantic_similarity"], -0.01)
        self.assertEqual(report["all_sample_deltas"][0]["llm_judge_delta"], 0.3)
        self.assertEqual(report["all_sample_deltas"][0]["semantic_similarity_delta"], -0.01)

    def test_kb_bucket_report_separates_not_called_no_high_discarded_and_entered_answer(self) -> None:
        report = {
            "all_sample_deltas": [
                {"sample_id": "a", "delta": 0.1},
                {"sample_id": "b", "delta": -0.2},
                {"sample_id": "c", "delta": 0.3},
                {"sample_id": "d", "delta": -0.4},
            ]
        }
        predictions = [
            {"sample_id": "a", "tool_trace": []},
            {
                "sample_id": "b",
                "tool_trace": [
                    {
                        "tool_name": "local_retrieve",
                        "outputs": {"metadata": {"kb_candidate_count": 0, "kb_discarded_count": 0}},
                    }
                ],
            },
            {
                "sample_id": "c",
                "tool_trace": [
                    {
                        "tool_name": "local_retrieve",
                        "outputs": {"metadata": {"kb_candidate_count": 5, "kb_discarded_count": 5}},
                    }
                ],
            },
            {
                "sample_id": "d",
                "tool_trace": [
                    {
                        "tool_name": "local_retrieve",
                        "outputs": {"metadata": {"kb_candidate_count": 5, "kb_discarded_count": 3}},
                    },
                    {
                        "tool_name": "finish_answer",
                        "inputs": {
                            "evidence": [
                                {"metadata": {"kind": "local_kb_chunk", "discarded_kb": True}},
                                {"metadata": {"kind": "local_kb_chunk"}},
                            ],
                        },
                    },
                ],
            },
        ]

        buckets = _kb_bucket_report(report, predictions)

        self.assertEqual(buckets["KB未调用"]["samples"], 1)
        self.assertEqual(buckets["KB无高相关结果"]["samples"], 1)
        self.assertEqual(buckets["KB候选被丢弃"]["samples"], 1)
        self.assertEqual(buckets["KB进入答案"]["samples"], 1)
        self.assertEqual(buckets["KB进入答案"]["average_delta"], -0.4)


if __name__ == "__main__":
    unittest.main()

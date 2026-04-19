from __future__ import annotations

from pathlib import Path
import unittest
from uuid import uuid4

from scripts.compare_runs import _check_sample_sets, _sample_set_report, _read_jsonl


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


if __name__ == "__main__":
    unittest.main()

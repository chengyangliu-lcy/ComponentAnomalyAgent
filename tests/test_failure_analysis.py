"""Tests for evaluator.failure_analysis module."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.failure_analysis import FailureAnalyzer, FailureReport, FailureCategory


def _make_eval_result(
    sample_id: str = "001",
    final_score: float = 0.5,
    completeness: int = 3,
    accuracy: int = 3,
    factual_consistency: float = 0.7,
    critical_errors: list | None = None,
    scoring_points: list | None = None,
    errors: list | None = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "final_score": final_score,
        "llm_judge": {
            "score": final_score,
            "accuracy": accuracy,
            "completeness": completeness,
            "factual_consistency": factual_consistency,
            "critical_errors": critical_errors or [],
            "scoring_point_matches": scoring_points or [],
        },
        "scoring_point_coverage": {"coverage": 0.5},
        "errors": errors or [],
    }


def _make_experiment_dir(tmpdir: str, name: str, results: list) -> str:
    exp_dir = os.path.join(tmpdir, name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "eval_results.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return exp_dir


class TestFailureAnalyzer(unittest.TestCase):
    """Tests for FailureAnalyzer."""

    def setUp(self):
        self.config = {
            "enabled": True,
            "score_threshold": 0.4,
            "critical_threshold": 0.2,
            "min_category_size": 2,
        }
        self.analyzer = FailureAnalyzer(self.config)
        self.tmpdir = tempfile.mkdtemp()

    def test_basic_failure_analysis(self):
        results = [
            _make_eval_result("001", final_score=0.8),
            _make_eval_result("002", final_score=0.6),
            _make_eval_result("003", final_score=0.3),  # failed
            _make_eval_result("004", final_score=0.1),  # critical
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_basic", results)

        report = self.analyzer.analyze(exp_dir)

        self.assertIsInstance(report, FailureReport)
        self.assertEqual(report.total_samples, 4)
        self.assertEqual(report.failed_samples, 2)  # 0.3 and 0.1
        self.assertAlmostEqual(report.failure_rate, 0.5)

    def test_critical_failure_rate(self):
        results = [
            _make_eval_result("001", final_score=0.8),
            _make_eval_result("002", final_score=0.1),  # critical (< 0.2)
            _make_eval_result("003", final_score=0.1),  # critical
            _make_eval_result("004", final_score=0.1),  # critical
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_critical", results)

        report = self.analyzer.analyze(exp_dir)

        self.assertAlmostEqual(report.critical_failure_rate, 0.75)

    def test_low_completeness_category(self):
        results = [
            _make_eval_result("001", final_score=0.8, completeness=5),
            _make_eval_result("002", final_score=0.3, completeness=1),
            _make_eval_result("003", final_score=0.2, completeness=1),
            _make_eval_result("004", final_score=0.2, completeness=2),
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_completeness", results)

        report = self.analyzer.analyze(exp_dir)

        cat_names = [c.name for c in report.categories]
        self.assertIn("low_completeness", cat_names)

    def test_contradiction_category(self):
        results = [
            _make_eval_result("001", final_score=0.8),
            _make_eval_result("002", final_score=0.3, critical_errors=["core_conclusion_contradicted:p1"]),
            _make_eval_result("003", final_score=0.2, critical_errors=["core_conclusion_contradicted:p2"]),
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_contradictions", results)

        report = self.analyzer.analyze(exp_dir)

        cat_names = [c.name for c in report.categories]
        self.assertIn("contradictions", cat_names)

    def test_low_factual_consistency_category(self):
        results = [
            _make_eval_result("001", final_score=0.8, factual_consistency=0.9),
            _make_eval_result("002", final_score=0.3, factual_consistency=0.2),
            _make_eval_result("003", final_score=0.2, factual_consistency=0.3),
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_fc", results)

        report = self.analyzer.analyze(exp_dir)

        cat_names = [c.name for c in report.categories]
        self.assertIn("low_factual_consistency", cat_names)

    def test_timeout_error_category(self):
        results = [
            _make_eval_result("001", final_score=0.8),
            _make_eval_result("002", final_score=0.3, errors=["Request timed out."]),
            _make_eval_result("003", final_score=0.2, errors=["Request timed out."]),
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_timeout", results)

        report = self.analyzer.analyze(exp_dir)

        cat_names = [c.name for c in report.categories]
        self.assertIn("timeout_errors", cat_names)

    def test_score_distribution(self):
        results = [
            _make_eval_result("001", final_score=0.9),
            _make_eval_result("002", final_score=0.7),
            _make_eval_result("003", final_score=0.5),
            _make_eval_result("004", final_score=0.3),
            _make_eval_result("005", final_score=0.1),
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_dist", results)

        report = self.analyzer.analyze(exp_dir)

        dist = report.score_distribution
        self.assertEqual(dist["0.8-1.0"], 1)
        self.assertEqual(dist["0.6-0.8"], 1)
        self.assertEqual(dist["0.4-0.6"], 1)
        self.assertEqual(dist["0.2-0.4"], 1)
        self.assertEqual(dist["0.0-0.2"], 1)

    def test_empty_experiment(self):
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_empty", [])

        report = self.analyzer.analyze(exp_dir)

        self.assertEqual(report.total_samples, 0)
        self.assertEqual(report.failed_samples, 0)
        self.assertAlmostEqual(report.failure_rate, 0.0)

    def test_failure_report_to_dict(self):
        results = [
            _make_eval_result("001", final_score=0.8),
            _make_eval_result("002", final_score=0.3),
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_dict", results)

        report = self.analyzer.analyze(exp_dir)
        d = report.to_dict()

        self.assertIn("experiment", d)
        self.assertIn("total_samples", d)
        self.assertIn("categories", d)
        self.assertIn("score_distribution", d)

    def test_failure_category_to_dict(self):
        cat = FailureCategory(
            name="test_cat",
            description="Test category",
            count=5,
            percentage=50.0,
            avg_score=0.3,
            sample_ids=["001", "002", "003", "004", "005"],
            common_patterns=["pattern1", "pattern2"],
        )
        d = cat.to_dict()

        self.assertEqual(d["name"], "test_cat")
        self.assertEqual(d["count"], 5)
        # sample_ids should be limited to 10
        self.assertLessEqual(len(d["sample_ids"]), 10)

    def test_missing_eval_results(self):
        empty_dir = os.path.join(self.tmpdir, "empty_failure")
        os.makedirs(empty_dir, exist_ok=True)

        report = self.analyzer.analyze(empty_dir)

        self.assertEqual(report.total_samples, 0)

    def test_min_category_size_filter(self):
        """Categories with fewer samples than min_category_size should be excluded."""
        results = [
            _make_eval_result("001", final_score=0.8, completeness=5),
            _make_eval_result("002", final_score=0.3, completeness=1),  # Only 1 sample
        ]
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_min_cat", results)

        # Set min_category_size to 2
        self.config["min_category_size"] = 2
        analyzer = FailureAnalyzer(self.config)

        report = analyzer.analyze(exp_dir)

        # With only 1 low-completeness sample, it shouldn't form a category
        cat_names = [c.name for c in report.categories]
        self.assertNotIn("low_completeness", cat_names)


if __name__ == "__main__":
    unittest.main()

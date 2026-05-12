"""Tests for evaluator.compare module."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.compare import ExperimentComparator, MetricComparison, ComparisonResult


def _make_summary(
    final_score: float = 0.5,
    llm_judge: float = 0.6,
    semantic: float = 0.7,
    coverage: float = 0.4,
    rouge: float = 0.25,
    entity: float = 0.3,
    accuracy: float = 3.5,
    completeness: float = 3.5,
    critical_error_rate: float = 0.4,
) -> dict:
    """Create a minimal evaluation_summary.json."""
    return {
        "final_score": final_score,
        "total": 10,
        "averages": {
            "semantic_similarity": semantic,
            "scoring_point_coverage": coverage,
            "claim_rouge_l": rouge,
            "technical_entity_match": entity,
        },
        "llm_judge": {
            "score": llm_judge,
            "accuracy": accuracy,
            "completeness": completeness,
        },
        "quality_rates": {
            "critical_error_rate": critical_error_rate,
        },
    }


def _make_experiment_dir(tmpdir: str, name: str, summary: dict) -> str:
    """Create a temporary experiment directory with summary."""
    exp_dir = os.path.join(tmpdir, name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f)
    return exp_dir


class TestExperimentComparator(unittest.TestCase):
    """Tests for ExperimentComparator."""

    def setUp(self):
        self.config = {
            "improvement_threshold": 0.01,
            "regression_threshold": 0.02,
            "fail_on_regression": True,
        }
        self.comparator = ExperimentComparator(self.config)
        self.tmpdir = tempfile.mkdtemp()

    def test_identical_experiments_no_improvements_or_regressions(self):
        summary = _make_summary()
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_a", summary)
        base_dir = _make_experiment_dir(self.tmpdir, "exp_b", summary)

        result = self.comparator.compare(exp_dir, base_dir)

        self.assertIsInstance(result, ComparisonResult)
        self.assertEqual(len(result.improvements), 0)
        self.assertEqual(len(result.regressions), 0)
        self.assertTrue(result.passed)

    def test_improvement_detected(self):
        base_summary = _make_summary(final_score=0.5, llm_judge=0.5)
        exp_summary = _make_summary(final_score=0.55, llm_judge=0.6)  # 20% improvement in llm_judge

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_improve", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_improve", base_summary)

        result = self.comparator.compare(exp_dir, base_dir)

        # llm_judge improved by 20%, should be detected
        improved_names = [m.name for m in result.improvements]
        self.assertIn("llm_judge", improved_names)
        self.assertTrue(result.passed)

    def test_regression_causes_failure(self):
        base_summary = _make_summary(final_score=0.6, llm_judge=0.7)
        exp_summary = _make_summary(final_score=0.5, llm_judge=0.5)  # ~29% regression in llm_judge

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_regress", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_regress", base_summary)

        result = self.comparator.compare(exp_dir, base_dir)

        regressed_names = [m.name for m in result.regressions]
        self.assertIn("llm_judge", regressed_names)
        # Significant regression should cause failure
        self.assertFalse(result.passed)

    def test_score_delta_computation(self):
        base_summary = _make_summary(final_score=0.5)
        exp_summary = _make_summary(final_score=0.6)

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_delta", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_delta", base_summary)

        result = self.comparator.compare(exp_dir, base_dir)

        self.assertAlmostEqual(result.score_delta, 0.1, places=4)
        self.assertAlmostEqual(result.experiment_score, 0.6, places=4)
        self.assertAlmostEqual(result.baseline_score, 0.5, places=4)

    def test_lower_is_better_metrics(self):
        """Critical error rate: lower is better."""
        base_summary = _make_summary(critical_error_rate=0.5)
        exp_summary = _make_summary(critical_error_rate=0.3)  # Improved (lower)

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_lower", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_lower", base_summary)

        # Add critical_error_rate to metrics to compare
        self.config["metrics"] = ["critical_error_rate"]
        comparator = ExperimentComparator(self.config)

        result = comparator.compare(exp_dir, base_dir)

        improved_names = [m.name for m in result.improvements]
        self.assertIn("critical_error_rate", improved_names)

    def test_marginal_change_not_flagged(self):
        """Changes within threshold should not be flagged as improvement/regression."""
        base_summary = _make_summary(llm_judge=0.6)
        exp_summary = _make_summary(llm_judge=0.605)  # 0.8% change, below 1% threshold

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_marginal", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_marginal", base_summary)

        result = self.comparator.compare(exp_dir, base_dir)

        # Should not be flagged as improvement or regression
        improved_names = [m.name for m in result.improvements]
        regressed_names = [m.name for m in result.regressions]
        self.assertNotIn("llm_judge", improved_names)
        self.assertNotIn("llm_judge", regressed_names)

    def test_metric_comparison_to_dict(self):
        mc = MetricComparison(
            name="test_metric",
            experiment_value=0.6,
            baseline_value=0.5,
            delta=0.1,
            delta_pct=20.0,
            improved=True,
            regressed=False,
            significance="significant",
        )
        d = mc.to_dict()
        self.assertEqual(d["name"], "test_metric")
        self.assertTrue(d["improved"])
        self.assertEqual(d["significance"], "significant")

    def test_comparison_result_summary(self):
        base_summary = _make_summary(final_score=0.5)
        exp_summary = _make_summary(final_score=0.6)

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_summary", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_summary", base_summary)

        result = self.comparator.compare(exp_dir, base_dir)
        summary = result.summary

        self.assertIn("PASSED", summary)
        self.assertIn("0.6", summary)

    def test_missing_summary_file_raises(self):
        empty_dir = os.path.join(self.tmpdir, "empty")
        os.makedirs(empty_dir, exist_ok=True)

        with self.assertRaises(FileNotFoundError):
            self.comparator.compare(empty_dir, empty_dir)


if __name__ == "__main__":
    unittest.main()

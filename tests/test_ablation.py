"""Tests for evaluator.ablation module."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator.ablation import AblationAnalyzer, AblationResult, ComponentContribution


def _make_summary(
    final_score: float = 0.5,
    llm_judge: float = 0.6,
    semantic: float = 0.7,
    coverage: float = 0.4,
    rouge: float = 0.25,
    entity: float = 0.3,
) -> dict:
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
        },
    }


def _make_experiment_dir(tmpdir: str, name: str, summary: dict) -> str:
    exp_dir = os.path.join(tmpdir, name)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f)
    return exp_dir


class TestAblationAnalyzer(unittest.TestCase):
    """Tests for AblationAnalyzer."""

    def setUp(self):
        self.config = {
            "enabled": True,
            "weights": {
                "llm_judge": 0.45,
                "structured_point_coverage": 0.25,
                "semantic_similarity": 0.20,
                "claim_rouge_l": 0.05,
                "technical_entity_match": 0.05,
            },
        }
        self.analyzer = AblationAnalyzer(self.config)
        self.tmpdir = tempfile.mkdtemp()

    def test_component_contributions_computed(self):
        summary = _make_summary(
            final_score=0.5,
            llm_judge=0.6,
            semantic=0.7,
            coverage=0.4,
            rouge=0.25,
            entity=0.3,
        )
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_ablation", summary)

        result = self.analyzer.analyze(exp_dir)

        self.assertIsInstance(result, AblationResult)
        self.assertEqual(len(result.components), 5)

        # Check llm_judge contribution
        llm_comp = next(c for c in result.components if c.name == "llm_judge")
        self.assertAlmostEqual(llm_comp.weight, 0.45)
        self.assertAlmostEqual(llm_comp.score, 0.6)
        self.assertAlmostEqual(llm_comp.weighted_score, 0.27)  # 0.6 * 0.45

    def test_contribution_percentages_sum_to_100(self):
        # Use component scores that are consistent with final_score
        # llm_judge=0.6*0.45=0.27, coverage=0.4*0.25=0.10, semantic=0.7*0.20=0.14,
        # rouge=0.25*0.05=0.0125, entity=0.3*0.05=0.015 -> total=0.5375
        summary = _make_summary(final_score=0.5375)
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_pct", summary)

        result = self.analyzer.analyze(exp_dir)

        total_pct = sum(c.contribution_pct for c in result.components)
        self.assertAlmostEqual(total_pct, 100.0, places=0)

    def test_top_contributors_sorted(self):
        summary = _make_summary()
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_sorted", summary)

        result = self.analyzer.analyze(exp_dir)

        top = result.top_contributors
        # Should be sorted by absolute weighted score descending
        for i in range(len(top) - 1):
            self.assertGreaterEqual(abs(top[i].weighted_score), abs(top[i + 1].weighted_score))

    def test_ablation_with_baseline(self):
        base_summary = _make_summary(final_score=0.4, llm_judge=0.5)
        exp_summary = _make_summary(final_score=0.5, llm_judge=0.6)

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_with_base", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_for_ablation", base_summary)

        result = self.analyzer.analyze(exp_dir, baseline_dir=base_dir)

        self.assertEqual(result.baseline_name, "base_for_ablation")
        self.assertAlmostEqual(result.baseline_score, 0.4)

        # Check deltas exist
        llm_comp = next(c for c in result.components if c.name == "llm_judge")
        self.assertIsNotNone(llm_comp.delta)
        self.assertIsNotNone(llm_comp.baseline_weighted_score)

    def test_top_deltas_with_baseline(self):
        base_summary = _make_summary(llm_judge=0.4, coverage=0.3)
        exp_summary = _make_summary(llm_judge=0.7, coverage=0.5)

        exp_dir = _make_experiment_dir(self.tmpdir, "exp_deltas", exp_summary)
        base_dir = _make_experiment_dir(self.tmpdir, "base_deltas", base_summary)

        result = self.analyzer.analyze(exp_dir, baseline_dir=base_dir)

        top_deltas = result.top_deltas
        self.assertGreater(len(top_deltas), 0)
        # Largest delta should be first
        for i in range(len(top_deltas) - 1):
            self.assertGreaterEqual(abs(top_deltas[i].delta), abs(top_deltas[i + 1].delta))

    def test_ablation_without_baseline_no_deltas(self):
        summary = _make_summary()
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_no_base", summary)

        result = self.analyzer.analyze(exp_dir)

        self.assertIsNone(result.baseline_name)
        self.assertEqual(len(result.top_deltas), 0)
        for comp in result.components:
            self.assertIsNone(comp.delta)
            self.assertIsNone(comp.baseline_weighted_score)

    def test_component_contribution_to_dict(self):
        cc = ComponentContribution(
            name="test",
            weight=0.5,
            score=0.8,
            weighted_score=0.4,
            baseline_weighted_score=0.3,
            delta=0.1,
            contribution_pct=80.0,
        )
        d = cc.to_dict()
        self.assertEqual(d["name"], "test")
        self.assertAlmostEqual(d["delta"], 0.1)

    def test_ablation_result_to_dict(self):
        summary = _make_summary()
        exp_dir = _make_experiment_dir(self.tmpdir, "exp_dict", summary)

        result = self.analyzer.analyze(exp_dir)
        d = result.to_dict()

        self.assertIn("experiment", d)
        self.assertIn("final_score", d)
        self.assertIn("components", d)
        self.assertEqual(len(d["components"]), 5)

    def test_missing_summary_raises(self):
        empty_dir = os.path.join(self.tmpdir, "empty_ablation")
        os.makedirs(empty_dir, exist_ok=True)

        with self.assertRaises(FileNotFoundError):
            self.analyzer.analyze(empty_dir)

    def test_default_weights(self):
        analyzer = AblationAnalyzer({})
        self.assertEqual(analyzer.weights, AblationAnalyzer.DEFAULT_WEIGHTS)


if __name__ == "__main__":
    unittest.main()

"""Experiment comparison engine.

Compares two experiments across all metrics, identifying significant
improvements and regressions with configurable thresholds.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MetricComparison:
    """Comparison result for a single metric."""

    name: str
    experiment_value: float
    baseline_value: float
    delta: float
    delta_pct: float
    improved: bool
    regressed: bool
    significance: str  # "significant", "marginal", "negligible"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "experiment": self.experiment_value,
            "baseline": self.baseline_value,
            "delta": self.delta,
            "delta_pct": self.delta_pct,
            "improved": self.improved,
            "regressed": self.regressed,
            "significance": self.significance,
        }


@dataclass
class ComparisonResult:
    """Full comparison between two experiments."""

    experiment_name: str
    baseline_name: str
    experiment_score: float
    baseline_score: float
    score_delta: float
    metrics: List[MetricComparison]
    improvements: List[MetricComparison]
    regressions: List[MetricComparison]
    passed: bool

    @property
    def summary(self) -> str:
        """Human-readable summary of the comparison."""
        lines = [
            f"Comparison: {self.experiment_name} vs {self.baseline_name}",
            f"Score: {self.experiment_score:.4f} vs {self.baseline_score:.4f} ({self.score_delta:+.4f})",
            f"Improvements: {len(self.improvements)}, Regressions: {len(self.regressions)}",
            f"Verification: {'PASSED' if self.passed else 'FAILED'}",
        ]
        if self.improvements:
            lines.append("Top improvements:")
            for m in sorted(self.improvements, key=lambda x: abs(x.delta), reverse=True)[:3]:
                lines.append(f"  {m.name}: {m.baseline_value:.4f} -> {m.experiment_value:.4f} ({m.delta_pct:+.1f}%)")
        if self.regressions:
            lines.append("Top regressions:")
            for m in sorted(self.regressions, key=lambda x: abs(x.delta), reverse=True)[:3]:
                lines.append(f"  {m.name}: {m.baseline_value:.4f} -> {m.experiment_value:.4f} ({m.delta_pct:+.1f}%)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "baseline": self.baseline_name,
            "experiment_score": self.experiment_score,
            "baseline_score": self.baseline_score,
            "score_delta": self.score_delta,
            "passed": self.passed,
            "improvements": [m.to_dict() for m in self.improvements],
            "regressions": [m.to_dict() for m in self.regressions],
            "all_metrics": [m.to_dict() for m in self.metrics],
        }


class ExperimentComparator:
    """Compares two experiments across all metrics.

    Args:
        config: Comparison configuration with thresholds.
    """

    # Metrics where higher is better
    HIGHER_IS_BETTER = {
        "final_score", "llm_judge", "semantic_similarity",
        "scoring_point_coverage", "claim_rouge_l", "technical_entity_match",
        "accuracy", "completeness", "clarity", "usefulness",
        "factual_consistency", "fully_correct_rate", "core_conclusion_hit_rate",
    }

    # Metrics where lower is better
    LOWER_IS_BETTER = {
        "critical_error_rate", "unsupported_entity_rate",
    }

    def __init__(self, config: Dict[str, Any]):
        self.improvement_threshold = config.get("improvement_threshold", 0.01)
        self.regression_threshold = config.get("regression_threshold", 0.02)
        self.fail_on_regression = config.get("fail_on_regression", True)
        self.metrics_to_compare = config.get("metrics", [
            "final_score", "llm_judge", "semantic_similarity",
            "scoring_point_coverage", "claim_rouge_l", "technical_entity_match",
        ])

    def compare(self, experiment_dir: str, baseline_dir: str) -> ComparisonResult:
        """Compare two experiments.

        Args:
            experiment_dir: Path to experiment output directory.
            baseline_dir: Path to baseline experiment directory.

        Returns:
            ComparisonResult with detailed metric comparisons.
        """
        exp_summary = self._load_summary(experiment_dir)
        base_summary = self._load_summary(baseline_dir)

        exp_name = Path(experiment_dir).name
        base_name = Path(baseline_dir).name

        exp_score = exp_summary.get("final_score", 0.0)
        base_score = base_summary.get("final_score", 0.0)
        score_delta = exp_score - base_score

        metrics = []
        for metric_name in self.metrics_to_compare:
            exp_val = self._extract_metric(exp_summary, metric_name)
            base_val = self._extract_metric(base_summary, metric_name)

            if exp_val is None or base_val is None:
                continue

            delta = exp_val - base_val
            delta_pct = (delta / base_val * 100) if base_val != 0 else 0.0

            # Determine if improved or regressed
            if metric_name in self.LOWER_IS_BETTER:
                improved = delta < -self.improvement_threshold * abs(base_val)
                regressed = delta > self.regression_threshold * abs(base_val)
            else:
                improved = delta > self.improvement_threshold * abs(base_val) if base_val != 0 else delta > 0
                regressed = delta < -self.regression_threshold * abs(base_val) if base_val != 0 else delta < 0

            # Determine significance
            abs_delta_pct = abs(delta_pct)
            if abs_delta_pct >= 5.0:
                significance = "significant"
            elif abs_delta_pct >= 1.0:
                significance = "marginal"
            else:
                significance = "negligible"

            metrics.append(MetricComparison(
                name=metric_name,
                experiment_value=exp_val,
                baseline_value=base_val,
                delta=delta,
                delta_pct=delta_pct,
                improved=improved,
                regressed=regressed,
                significance=significance,
            ))

        improvements = [m for m in metrics if m.improved]
        regressions = [m for m in metrics if m.regressed]

        # Determine if verification passed
        passed = True
        if self.fail_on_regression and regressions:
            # Only fail on significant regressions
            significant_regressions = [r for r in regressions if r.significance == "significant"]
            if significant_regressions:
                passed = False

        return ComparisonResult(
            experiment_name=exp_name,
            baseline_name=base_name,
            experiment_score=exp_score,
            baseline_score=base_score,
            score_delta=score_delta,
            metrics=metrics,
            improvements=improvements,
            regressions=regressions,
            passed=passed,
        )

    def _load_summary(self, dir_path: str) -> Dict[str, Any]:
        """Load evaluation summary from directory."""
        summary_path = Path(dir_path) / "evaluation_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No evaluation_summary.json in {dir_path}")
        with open(summary_path) as f:
            return json.load(f)

    def _extract_metric(self, summary: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract a metric value from summary, checking multiple locations."""
        # Check top-level averages
        averages = summary.get("averages", {})
        if metric_name in averages:
            return float(averages[metric_name])

        # Check llm_judge sub-metrics
        llm_judge = summary.get("llm_judge", {})
        if metric_name in llm_judge:
            return float(llm_judge[metric_name])

        # Special case: "llm_judge" metric maps to llm_judge.score
        if metric_name == "llm_judge" and "score" in llm_judge:
            return float(llm_judge["score"])

        # Check quality_rates
        quality_rates = summary.get("quality_rates", {})
        if metric_name in quality_rates:
            return float(quality_rates[metric_name])

        # Check top-level (only if it's a numeric value, not a dict/list)
        if metric_name in summary:
            val = summary[metric_name]
            if isinstance(val, (int, float)):
                return float(val)

        return None

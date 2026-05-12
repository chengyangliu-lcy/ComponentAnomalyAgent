"""Experiment verification orchestrator.

Coordinates comparison, ablation, and failure analysis to produce
a unified verification report for an experiment.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compare import ExperimentComparator, ComparisonResult
from .ablation import AblationAnalyzer, AblationResult
from .failure_analysis import FailureAnalyzer, FailureReport

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    """Complete verification report for an experiment."""

    experiment_name: str
    experiment_dir: str
    baseline_name: Optional[str]

    # Summary
    final_score: float
    baseline_score: Optional[float]
    score_delta: Optional[float]

    # Component results
    comparison: Optional[ComparisonResult]
    ablation: Optional[AblationResult]
    failure_report: Optional[FailureReport]

    # Metadata
    sample_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Check if experiment passes verification thresholds."""
        if self.comparison and not self.comparison.passed:
            return False
        if self.failure_report and self.failure_report.critical_failure_rate > 0.5:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "experiment": self.experiment_name,
            "experiment_dir": self.experiment_dir,
            "baseline": self.baseline_name,
            "final_score": self.final_score,
            "baseline_score": self.baseline_score,
            "score_delta": self.score_delta,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "metadata": self.metadata,
        }
        if self.comparison:
            result["comparison"] = self.comparison.to_dict()
        if self.ablation:
            result["ablation"] = self.ablation.to_dict()
        if self.failure_report:
            result["failure_analysis"] = self.failure_report.to_dict()
        return result


class ExperimentVerifier:
    """Orchestrates experiment verification pipeline.

    Args:
        config: Verification configuration dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compare_config = config.get("comparison", {})
        self.ablation_config = config.get("ablation", {})
        self.failure_config = config.get("failure_analysis", {})
        self.output_config = config.get("output", {})

    def verify(
        self,
        experiment_dir: str,
        baseline_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> VerificationReport:
        """Run full verification pipeline on an experiment.

        Args:
            experiment_dir: Path to experiment output directory.
            baseline_dir: Path to baseline experiment directory (optional).
            experiment_name: Name of the experiment (optional, inferred from dir).

        Returns:
            VerificationReport with all analysis results.
        """
        exp_path = Path(experiment_dir)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

        # Infer experiment name
        if experiment_name is None:
            experiment_name = exp_path.name

        # Load experiment summary
        summary = self._load_summary(exp_path)
        final_score = summary.get("final_score", 0.0)
        sample_count = summary.get("total", 0)

        logger.info("Verifying experiment %s (score=%.4f, samples=%d)", experiment_name, final_score, sample_count)

        # Run comparison
        comparison = None
        baseline_score = None
        score_delta = None
        if baseline_dir:
            baseline_path = Path(baseline_dir)
            if baseline_path.exists():
                comparator = ExperimentComparator(self.compare_config)
                comparison = comparator.compare(str(exp_path), str(baseline_path))
                baseline_score = comparison.baseline_score
                score_delta = comparison.score_delta
                logger.info("Comparison: delta=%+.4f, passed=%s", score_delta, comparison.passed)

        # Run ablation
        ablation = None
        if self.ablation_config.get("enabled", True):
            ablation_analyzer = AblationAnalyzer(self.ablation_config)
            ablation = ablation_analyzer.analyze(
                str(exp_path),
                baseline_dir=baseline_dir,
            )
            logger.info("Ablation: %d components analyzed", len(ablation.components))

        # Run failure analysis
        failure_report = None
        if self.failure_config.get("enabled", True):
            failure_analyzer = FailureAnalyzer(self.failure_config)
            failure_report = failure_analyzer.analyze(str(exp_path))
            logger.info("Failure analysis: %d categories, critical_rate=%.2f",
                       len(failure_report.categories), failure_report.critical_failure_rate)

        report = VerificationReport(
            experiment_name=experiment_name,
            experiment_dir=str(exp_path),
            baseline_name=Path(baseline_dir).name if baseline_dir else None,
            final_score=final_score,
            baseline_score=baseline_score,
            score_delta=score_delta,
            comparison=comparison,
            ablation=ablation,
            failure_report=failure_report,
            sample_count=sample_count,
            metadata=summary,
        )

        # Save report if configured
        output_path = self.output_config.get("report_path")
        if output_path:
            self._save_report(report, output_path)

        return report

    def _load_summary(self, exp_path: Path) -> Dict[str, Any]:
        """Load experiment summary from evaluation_summary.json."""
        summary_path = exp_path / "evaluation_summary.json"
        if not summary_path.exists():
            logger.warning("No evaluation_summary.json found in %s", exp_path)
            return {}
        with open(summary_path) as f:
            return json.load(f)

    def _save_report(self, report: VerificationReport, output_path: str):
        """Save verification report to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Verification report saved to %s", output_path)

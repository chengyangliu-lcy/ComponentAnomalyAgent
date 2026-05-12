"""Failure analysis module.

Categorizes and analyzes failed samples to identify systematic
failure patterns and provide actionable insights.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FailureCategory:
    """A category of failures with representative samples."""

    name: str
    description: str
    count: int
    percentage: float
    avg_score: float
    sample_ids: List[str]
    common_patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "count": self.count,
            "percentage": self.percentage,
            "avg_score": self.avg_score,
            "sample_ids": self.sample_ids[:10],  # Limit to 10 samples
            "common_patterns": self.common_patterns,
        }


@dataclass
class FailureReport:
    """Complete failure analysis report."""

    experiment_name: str
    total_samples: int
    failed_samples: int
    failure_rate: float
    categories: List[FailureCategory]
    critical_failure_rate: float

    # Score distribution
    score_distribution: Dict[str, int]

    # Top failure patterns
    top_patterns: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment": self.experiment_name,
            "total_samples": self.total_samples,
            "failed_samples": self.failed_samples,
            "failure_rate": self.failure_rate,
            "critical_failure_rate": self.critical_failure_rate,
            "categories": [c.to_dict() for c in self.categories],
            "score_distribution": self.score_distribution,
            "top_patterns": self.top_patterns,
        }


class FailureAnalyzer:
    """Analyzes failed samples to identify patterns.

    Args:
        config: Failure analysis configuration.
    """

    def __init__(self, config: Dict[str, Any]):
        self.score_threshold = config.get("score_threshold", 0.4)
        self.critical_threshold = config.get("critical_threshold", 0.2)
        self.min_category_size = config.get("min_category_size", 3)
        self.enabled = config.get("enabled", True)

    def analyze(self, experiment_dir: str) -> FailureReport:
        """Analyze failures in an experiment.

        Args:
            experiment_dir: Path to experiment output directory.

        Returns:
            FailureReport with categorized failures.
        """
        exp_path = Path(experiment_dir)
        eval_results = self._load_eval_results(exp_path)
        exp_name = exp_path.name

        if not eval_results:
            return FailureReport(
                experiment_name=exp_name,
                total_samples=0,
                failed_samples=0,
                failure_rate=0.0,
                categories=[],
                critical_failure_rate=0.0,
                score_distribution={},
                top_patterns=[],
            )

        total = len(eval_results)

        # Identify failed samples
        failed = [r for r in eval_results if r.get("final_score", 0) < self.score_threshold]
        critical = [r for r in eval_results if r.get("final_score", 0) < self.critical_threshold]

        # Categorize failures
        categories = self._categorize_failures(failed)

        # Compute score distribution
        distribution = self._compute_distribution(eval_results)

        # Identify top patterns
        patterns = self._identify_patterns(failed)

        return FailureReport(
            experiment_name=exp_name,
            total_samples=total,
            failed_samples=len(failed),
            failure_rate=len(failed) / total if total > 0 else 0.0,
            categories=categories,
            critical_failure_rate=len(critical) / total if total > 0 else 0.0,
            score_distribution=distribution,
            top_patterns=patterns,
        )

    def _load_eval_results(self, exp_path: Path) -> List[Dict[str, Any]]:
        """Load evaluation results from JSONL file."""
        results_path = exp_path / "eval_results.jsonl"
        if not results_path.exists():
            logger.warning("No eval_results.jsonl found in %s", exp_path)
            return []

        results = []
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results

    def _categorize_failures(self, failed: List[Dict[str, Any]]) -> List[FailureCategory]:
        """Categorize failures by their root cause."""
        categories = []

        # Category 1: Low completeness (judge says incomplete)
        low_completeness = []
        for r in failed:
            judge = r.get("llm_judge", {})
            if judge.get("completeness", 5) <= 2:
                low_completeness.append(r)

        if len(low_completeness) >= self.min_category_size:
            categories.append(FailureCategory(
                name="low_completeness",
                description="Judge rated completeness as very low (<=2/5)",
                count=len(low_completeness),
                percentage=len(low_completeness) / len(failed) * 100 if failed else 0,
                avg_score=sum(r.get("final_score", 0) for r in low_completeness) / len(low_completeness),
                sample_ids=[r.get("sample_id", "") for r in low_completeness],
                common_patterns=self._extract_patterns(low_completeness),
            ))

        # Category 2: Contradictions detected
        contradictions = []
        for r in failed:
            judge = r.get("llm_judge", {})
            errors = judge.get("critical_errors", [])
            if any("contradict" in str(e).lower() for e in errors):
                contradictions.append(r)

        if len(contradictions) >= self.min_category_size:
            categories.append(FailureCategory(
                name="contradictions",
                description="Critical contradictions detected in scoring points",
                count=len(contradictions),
                percentage=len(contradictions) / len(failed) * 100 if failed else 0,
                avg_score=sum(r.get("final_score", 0) for r in contradictions) / len(contradictions),
                sample_ids=[r.get("sample_id", "") for r in contradictions],
                common_patterns=self._extract_patterns(contradictions),
            ))

        # Category 3: Low factual consistency
        low_fc = []
        for r in failed:
            judge = r.get("llm_judge", {})
            if judge.get("factual_consistency", 1.0) < 0.5:
                low_fc.append(r)

        if len(low_fc) >= self.min_category_size:
            categories.append(FailureCategory(
                name="low_factual_consistency",
                description="Factual consistency below 0.5",
                count=len(low_fc),
                percentage=len(low_fc) / len(failed) * 100 if failed else 0,
                avg_score=sum(r.get("final_score", 0) for r in low_fc) / len(low_fc),
                sample_ids=[r.get("sample_id", "") for r in low_fc],
                common_patterns=self._extract_patterns(low_fc),
            ))

        # Category 4: Low scoring point coverage
        low_coverage = []
        for r in failed:
            coverage = r.get("scoring_point_coverage", {})
            if isinstance(coverage, dict):
                cov_val = coverage.get("coverage", 1.0)
            else:
                cov_val = float(coverage) if coverage else 1.0
            if cov_val < 0.3:
                low_coverage.append(r)

        if len(low_coverage) >= self.min_category_size:
            categories.append(FailureCategory(
                name="low_coverage",
                description="Scoring point coverage below 30%",
                count=len(low_coverage),
                percentage=len(low_coverage) / len(failed) * 100 if failed else 0,
                avg_score=sum(r.get("final_score", 0) for r in low_coverage) / len(low_coverage),
                sample_ids=[r.get("sample_id", "") for r in low_coverage],
                common_patterns=self._extract_patterns(low_coverage),
            ))

        # Category 5: Timeout/execution errors
        timeout_errors = []
        for r in failed:
            errors = r.get("errors", [])
            if any(("timeout" in str(e).lower() or "timed out" in str(e).lower()) for e in errors):
                timeout_errors.append(r)

        if len(timeout_errors) >= self.min_category_size:
            categories.append(FailureCategory(
                name="timeout_errors",
                description="Execution timeout errors",
                count=len(timeout_errors),
                percentage=len(timeout_errors) / len(failed) * 100 if failed else 0,
                avg_score=sum(r.get("final_score", 0) for r in timeout_errors) / len(timeout_errors),
                sample_ids=[r.get("sample_id", "") for r in timeout_errors],
                common_patterns=["timeout"],
            ))

        # Category 6: Other failures (uncategorized)
        categorized_ids = set()
        for cat in categories:
            categorized_ids.update(cat.sample_ids)

        other = [r for r in failed if r.get("sample_id", "") not in categorized_ids]
        if len(other) >= self.min_category_size:
            categories.append(FailureCategory(
                name="other",
                description="Uncategorized failures",
                count=len(other),
                percentage=len(other) / len(failed) * 100 if failed else 0,
                avg_score=sum(r.get("final_score", 0) for r in other) / len(other),
                sample_ids=[r.get("sample_id", "") for r in other],
                common_patterns=self._extract_patterns(other),
            ))

        # Sort by count descending
        categories.sort(key=lambda c: c.count, reverse=True)
        return categories

    def _extract_patterns(self, samples: List[Dict[str, Any]]) -> List[str]:
        """Extract common patterns from a list of samples."""
        patterns = []

        # Check for common error types
        error_counter = Counter()
        for r in samples:
            for error in r.get("errors", []):
                error_counter[str(error)[:50]] += 1

        for error, count in error_counter.most_common(3):
            patterns.append(f"error:{error} ({count}x)")

        # Check for common scoring point failures
        sp_failures = Counter()
        for r in samples:
            judge = r.get("llm_judge", {})
            for sp in judge.get("scoring_point_matches", []):
                if sp.get("status") in ("missed", "contradicted"):
                    sp_failures[sp.get("id", "unknown")] += 1

        for sp_id, count in sp_failures.most_common(3):
            patterns.append(f"sp_miss:{sp_id} ({count}x)")

        return patterns

    def _compute_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Compute score distribution buckets."""
        buckets = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        for r in results:
            score = r.get("final_score", 0)
            if score < 0.2:
                buckets["0.0-0.2"] += 1
            elif score < 0.4:
                buckets["0.2-0.4"] += 1
            elif score < 0.6:
                buckets["0.4-0.6"] += 1
            elif score < 0.8:
                buckets["0.6-0.8"] += 1
            else:
                buckets["0.8-1.0"] += 1

        return buckets

    def _identify_patterns(self, failed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify top failure patterns across all categories."""
        patterns = []

        # Pattern 1: Most common scoring point failures
        sp_counter = Counter()
        for r in failed:
            judge = r.get("llm_judge", {})
            for sp in judge.get("scoring_point_matches", []):
                if sp.get("status") in ("missed", "contradicted"):
                    sp_counter[sp.get("id", "unknown")] += 1

        if sp_counter:
            patterns.append({
                "type": "scoring_point_failures",
                "description": "Most frequently missed/contradicted scoring points",
                "items": [{"id": sp_id, "count": count} for sp_id, count in sp_counter.most_common(5)],
            })

        # Pattern 2: Judge dimension weaknesses
        dimension_scores = defaultdict(list)
        for r in failed:
            judge = r.get("llm_judge", {})
            for dim in ("accuracy", "completeness", "clarity", "usefulness"):
                val = judge.get(dim)
                if val is not None:
                    dimension_scores[dim].append(val)

        if dimension_scores:
            weak_dims = []
            for dim, scores in dimension_scores.items():
                avg = sum(scores) / len(scores) if scores else 0
                weak_dims.append({"dimension": dim, "avg_score": round(avg, 2), "count": len(scores)})
            weak_dims.sort(key=lambda d: d["avg_score"])
            patterns.append({
                "type": "weak_dimensions",
                "description": "Judge dimensions with lowest average scores in failures",
                "items": weak_dims,
            })

        return patterns

"""Ablation analyzer for experiment verification.

Disassembles the final score into component contributions and
identifies which components drive improvements or regressions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ComponentContribution:
    """Contribution of a single scoring component."""

    name: str
    weight: float
    score: float
    weighted_score: float
    baseline_weighted_score: Optional[float]
    delta: Optional[float]
    contribution_pct: float  # percentage of final score

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "weight": self.weight,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "contribution_pct": self.contribution_pct,
        }
        if self.baseline_weighted_score is not None:
            result["baseline_weighted_score"] = self.baseline_weighted_score
            result["delta"] = self.delta
        return result


@dataclass
class AblationResult:
    """Complete ablation analysis result."""

    experiment_name: str
    final_score: float
    components: List[ComponentContribution]
    baseline_name: Optional[str] = None
    baseline_score: Optional[float] = None

    @property
    def top_contributors(self) -> List[ComponentContribution]:
        """Components sorted by absolute contribution (descending)."""
        return sorted(self.components, key=lambda c: abs(c.weighted_score), reverse=True)

    @property
    def top_deltas(self) -> List[ComponentContribution]:
        """Components with baseline comparison, sorted by absolute delta."""
        if not self.baseline_name:
            return []
        with_delta = [c for c in self.components if c.delta is not None]
        return sorted(with_delta, key=lambda c: abs(c.delta), reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "experiment": self.experiment_name,
            "final_score": self.final_score,
            "components": [c.to_dict() for c in self.components],
        }
        if self.baseline_name:
            result["baseline"] = self.baseline_name
            result["baseline_score"] = self.baseline_score
            result["top_deltas"] = [c.to_dict() for c in self.top_deltas[:3]]
        return result


class AblationAnalyzer:
    """Analyzes component contributions to the final score.

    Args:
        config: Ablation configuration.
    """

    DEFAULT_WEIGHTS = {
        "llm_judge": 0.45,
        "structured_point_coverage": 0.25,
        "semantic_similarity": 0.20,
        "claim_rouge_l": 0.05,
        "technical_entity_match": 0.05,
    }

    def __init__(self, config: Dict[str, Any]):
        self.weights = config.get("weights", self.DEFAULT_WEIGHTS)
        self.enabled = config.get("enabled", True)

    def analyze(
        self,
        experiment_dir: str,
        baseline_dir: Optional[str] = None,
    ) -> AblationResult:
        """Perform ablation analysis on an experiment.

        Args:
            experiment_dir: Path to experiment output directory.
            baseline_dir: Path to baseline directory for comparison (optional).

        Returns:
            AblationResult with component contributions.
        """
        exp_summary = self._load_summary(experiment_dir)
        exp_name = Path(experiment_dir).name

        # Extract component scores
        component_scores = self._extract_component_scores(exp_summary)
        final_score = exp_summary.get("final_score", 0.0)

        # Load baseline if provided
        base_scores = None
        base_name = None
        base_score = None
        if baseline_dir:
            base_path = Path(baseline_dir)
            if base_path.exists():
                base_summary = self._load_summary(baseline_dir)
                base_scores = self._extract_component_scores(base_summary)
                base_name = base_path.name
                base_score = base_summary.get("final_score", 0.0)

        # Build contributions
        components = []
        for name, weight in self.weights.items():
            score = component_scores.get(name, 0.0)
            weighted_score = score * weight
            contribution_pct = (weighted_score / final_score * 100) if final_score > 0 else 0.0

            baseline_weighted = None
            delta = None
            if base_scores is not None:
                base_val = base_scores.get(name, 0.0)
                baseline_weighted = base_val * weight
                delta = weighted_score - baseline_weighted

            components.append(ComponentContribution(
                name=name,
                weight=weight,
                score=score,
                weighted_score=weighted_score,
                baseline_weighted_score=baseline_weighted,
                delta=delta,
                contribution_pct=contribution_pct,
            ))

        return AblationResult(
            experiment_name=exp_name,
            final_score=final_score,
            components=components,
            baseline_name=base_name,
            baseline_score=base_score,
        )

    def _load_summary(self, dir_path: str) -> Dict[str, Any]:
        """Load evaluation summary."""
        summary_path = Path(dir_path) / "evaluation_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"No evaluation_summary.json in {dir_path}")
        with open(summary_path) as f:
            return json.load(f)

    def _extract_component_scores(self, summary: Dict[str, Any]) -> Dict[str, float]:
        """Extract component scores from summary."""
        scores = {}
        averages = summary.get("averages", {})
        llm_judge = summary.get("llm_judge", {})

        # Map component names to their locations in the summary
        score_map = {
            "llm_judge": llm_judge.get("score"),
            "structured_point_coverage": averages.get("scoring_point_coverage"),
            "semantic_similarity": averages.get("semantic_similarity"),
            "claim_rouge_l": averages.get("claim_rouge_l"),
            "technical_entity_match": averages.get("technical_entity_match"),
        }

        for name, value in score_map.items():
            if value is not None:
                scores[name] = float(value)

        return scores

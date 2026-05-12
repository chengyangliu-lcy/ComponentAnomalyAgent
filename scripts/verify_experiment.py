#!/usr/bin/env python3
"""CLI entry point for experiment verification.

Usage:
    python scripts/verify_experiment.py --experiment outputs/exp21_qwen_forced_search
    python scripts/verify_experiment.py --experiment outputs/exp21_qwen_forced_search --baseline outputs/exp20_qwen_search
    python scripts/verify_experiment.py --experiment outputs/exp21_qwen_forced_search --baseline outputs/exp20_qwen_search --config configs/verification.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from evaluator.verify import ExperimentVerifier


def load_config(config_path: str) -> dict:
    """Load verification configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def print_report(report, verbose: bool = False):
    """Print verification report to stdout."""
    print("\n" + "=" * 60)
    print(f"VERIFICATION REPORT: {report.experiment_name}")
    print("=" * 60)

    # Overall result
    status = "PASSED" if report.passed else "FAILED"
    print(f"\nOverall: {status}")
    print(f"Final Score: {report.final_score:.4f}")
    if report.baseline_score is not None:
        print(f"Baseline Score: {report.baseline_score:.4f}")
        print(f"Delta: {report.score_delta:+.4f}")
    print(f"Samples: {report.sample_count}")

    # Comparison summary
    if report.comparison:
        print("\n--- Comparison ---")
        print(report.comparison.summary)

    # Ablation summary
    if report.ablation:
        print("\n--- Component Ablation ---")
        for comp in report.ablation.top_contributors:
            delta_str = ""
            if comp.delta is not None:
                delta_str = f" (delta: {comp.delta:+.4f})"
            print(f"  {comp.name}: {comp.score:.4f} * {comp.weight:.2f} = {comp.weighted_score:.4f} "
                  f"({comp.contribution_pct:.1f}%){delta_str}")

        if report.ablation.top_deltas:
            print("\n  Top deltas:")
            for comp in report.ablation.top_deltas[:3]:
                print(f"    {comp.name}: {comp.delta:+.4f}")

    # Failure analysis summary
    if report.failure_report:
        fr = report.failure_report
        print("\n--- Failure Analysis ---")
        print(f"  Failed samples: {fr.failed_samples}/{fr.total_samples} ({fr.failure_rate:.1%})")
        print(f"  Critical failures: {fr.critical_failure_rate:.1%}")

        if fr.score_distribution:
            print("  Score distribution:")
            for bucket, count in fr.score_distribution.items():
                bar = "#" * (count // 5)
                print(f"    {bucket}: {count:3d} {bar}")

        if fr.categories:
            print("  Failure categories:")
            for cat in fr.categories:
                print(f"    {cat.name}: {cat.count} samples ({cat.percentage:.1f}%), avg_score={cat.avg_score:.3f}")
                if verbose and cat.common_patterns:
                    for p in cat.common_patterns[:3]:
                        print(f"      - {p}")

        if fr.top_patterns and verbose:
            print("  Top patterns:")
            for pattern in fr.top_patterns:
                print(f"    {pattern['type']}: {pattern['description']}")
                for item in pattern.get("items", [])[:3]:
                    print(f"      {item}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Verify experiment results")
    parser.add_argument("--experiment", "-e", required=True, help="Path to experiment output directory")
    parser.add_argument("--baseline", "-b", help="Path to baseline experiment directory")
    parser.add_argument("--config", "-c", default="configs/verification.yaml", help="Verification config file")
    parser.add_argument("--output", "-o", help="Output report path (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--name", help="Experiment name (default: directory name)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        logging.warning("Config file %s not found, using defaults", config_path)
        config = {}

    # Override output path if specified
    if args.output:
        config.setdefault("output", {})["report_path"] = args.output

    # Create verifier and run
    verifier = ExperimentVerifier(config)
    report = verifier.verify(
        experiment_dir=args.experiment,
        baseline_dir=args.baseline,
        experiment_name=args.name,
    )

    # Print report
    print_report(report, verbose=args.verbose)

    # Save report
    output_path = args.output or config.get("output", {}).get("report_path")
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {output_path}")

    # Exit with appropriate code
    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()

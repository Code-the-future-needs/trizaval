"""
Command-line entrypoint for running trizaval eval suites.

Usage:
    python -m trizaval.cli run suites/example_suite.yaml
    python -m trizaval.cli run suites/example_suite.yaml --format json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from trizaval import BootstrapResult, EffectSizeResult
from trizaval.harness.runner import CandidateReport, RunnerError, SequentialStatResult, SuiteReport, run_suite
from trizaval.suite.loader import SuiteLoadError, load_suite


def _statistic_to_dict(stat) -> Optional[dict]:
    if stat is None:
        return None
    if isinstance(stat, SequentialStatResult):
        return {
            "method": "sequential",
            "rejected": stat.rejected,
            "rejected_at_n": stat.rejected_at_n,
            "final_mean": stat.final_mean,
            "final_n": stat.final_n,
            "total_paired_observations": stat.total_paired_observations,
        }
    if isinstance(stat, EffectSizeResult):
        return {
            "method": "effect_size",
            "cohens_d": stat.cohens_d,
            "hedges_g": stat.hedges_g,
            "magnitude": str(stat.magnitude),
            "n_baseline": stat.n_baseline,
            "n_treatment": stat.n_treatment,
        }
    if isinstance(stat, BootstrapResult):
        return {
            "method": "bootstrap",
            "point_estimate": stat.point_estimate,
            "ci_lower": stat.ci_lower,
            "ci_upper": stat.ci_upper,
            "confidence_level": stat.confidence_level,
            "n_resamples": stat.n_resamples,
        }
    # Defensive fallback -- should be unreachable given the three
    # known statistic types, but surfaces clearly rather than
    # silently dropping an unrecognized type if one is added later
    # without updating this function.
    return {"method": "unknown", "repr": repr(stat)}


def _length_bias_to_dict(lb) -> Optional[dict]:
    if lb is None:
        return None
    return {
        "slope": lb.slope,
        "intercept": lb.intercept,
        "correlation": lb.correlation,
        "adjusted_scores": lb.adjusted_scores,
    }


def _candidate_report_to_dict(cr: CandidateReport) -> dict:
    return {
        "candidate_name": cr.candidate_name,
        "baseline_scores": cr.baseline_scores,
        "candidate_scores": cr.candidate_scores,
        "statistic_result": _statistic_to_dict(cr.statistic_result),
        "length_bias_applied": cr.length_bias_applied,
        "baseline_length_bias": _length_bias_to_dict(cr.baseline_length_bias),
        "candidate_length_bias": _length_bias_to_dict(cr.candidate_length_bias),
        "raw_baseline_scores": cr.raw_baseline_scores,
        "raw_candidate_scores": cr.raw_candidate_scores,
        "errors": cr.errors,
    }


def format_json(report: SuiteReport) -> str:
    data = {
        "suite_name": report.suite_name,
        "candidate_reports": [_candidate_report_to_dict(cr) for cr in report.candidate_reports],
    }
    return json.dumps(data, indent=2)


def format_text(report: SuiteReport) -> str:
    lines = [f"Suite: {report.suite_name}", ""]

    for cr in report.candidate_reports:
        lines.append(f"Candidate: {cr.candidate_name}")
        lines.append(f"  Baseline scores:  {cr.baseline_scores}")
        lines.append(f"  Candidate scores: {cr.candidate_scores}")

        if cr.length_bias_applied:
            lines.append(f"  Length-bias correction applied (raw candidate scores were: {cr.raw_candidate_scores})")

        stat = cr.statistic_result
        if stat is None:
            lines.append("  Statistic: not computed (insufficient data, or method not applicable here)")
        elif isinstance(stat, SequentialStatResult):
            status = f"REJECTED at n={stat.rejected_at_n}" if stat.rejected else "not rejected"
            lines.append(
                f"  Sequential test: {status} "
                f"(final mean={stat.final_mean:.4f}, n={stat.final_n}/{stat.total_paired_observations})"
            )
        elif isinstance(stat, EffectSizeResult):
            lines.append(
                f"  Effect size: Cohen's d={stat.cohens_d:.4f}, Hedges' g={stat.hedges_g:.4f}, "
                f"magnitude={stat.magnitude}"
            )
        elif isinstance(stat, BootstrapResult):
            lines.append(
                f"  Bootstrap: point estimate={stat.point_estimate:.4f}, "
                f"{stat.confidence_level * 100:.0f}% CI=[{stat.ci_lower:.4f}, {stat.ci_upper:.4f}]"
            )

        if cr.errors:
            lines.append(f"  Errors ({len(cr.errors)}):")
            for e in cr.errors:
                lines.append(f"    - {e}")

        lines.append("")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trizaval", description="Statistically rigorous evaluation tooling for AI systems"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an eval suite from a YAML file")
    run_parser.add_argument("suite_path", help="Path to a suite YAML config file")
    run_parser.add_argument("--format", choices=["text", "json"], default="text")
    run_parser.add_argument(
        "--storage-dir",
        default=None,
        help="If set, persist this run's results as a new row in <storage_dir>/<suite_name>.parquet, "
        "in addition to printing the report.",
    )

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        try:
            suite = load_suite(args.suite_path)
        except SuiteLoadError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        try:
            report = run_suite(suite)
        except RunnerError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        if args.format == "json":
            print(format_json(report))
        else:
            print(format_text(report))

        if args.storage_dir:
            # Imported here, not at module top level, so that
            # `trizaval.cli` doesn't hard-require pyarrow/duckdb for
            # users who only ever run without --storage-dir.
            from trizaval.storage.arrow_store import StorageError, append_run

            try:
                path = append_run(report, args.storage_dir)
            except StorageError as e:
                print(f"warning: run completed but failed to save to storage: {e}", file=sys.stderr)
                return 0  # the eval run itself succeeded; storage failure is reported, not fatal
            print(f"Saved run to {path}", file=sys.stderr)

        return 0

    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())
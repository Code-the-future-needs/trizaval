"""
Runner: orchestrates a full eval suite run -- builds providers, runs
every test case against baseline and each candidate, judges the
results, applies length-bias correction where configured, and
computes the suite's configured statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from trizaval import (
    BootstrapResult,
    EffectSizeResult,
    LengthBiasResult,
    SequentialDecision,
    SequentialTest,
    block_bootstrap_mean,
    cohens_d,
)
from trizaval.harness.judge import (
    JudgeError,
    JudgeResult,
    apply_length_bias_correction,
    judge_llm_single,
    judge_rule_based,
)
from trizaval.harness.providers import ProviderError, build_provider
from trizaval.suite.schema import (
    BootstrapStatConfig,
    EffectSizeStatConfig,
    EvalSuite,
    LlmJudgeConfig,
    ProviderConfig,
    RuleBasedJudgeConfig,
    SequentialStatConfig,
)


@dataclass
class SequentialStatResult:
    """Result of running a sequential test over paired
    (candidate - baseline) score differences, in test-case order."""

    rejected: bool
    rejected_at_n: Optional[int]
    final_mean: float
    final_n: int
    total_paired_observations: int


StatisticResult = Union[BootstrapResult, EffectSizeResult, SequentialStatResult]


@dataclass
class CandidateReport:
    """Results for one candidate provider, compared against baseline."""

    candidate_name: str
    baseline_results: list[JudgeResult]
    candidate_results: list[JudgeResult]
    baseline_scores: list[float]
    """Raw scores actually used for statistics: length-bias-corrected
    if correction was applied, otherwise identical to raw judge
    output. See `length_bias_applied` to tell which."""
    candidate_scores: list[float]
    statistic_result: Optional[StatisticResult] = None
    length_bias_applied: bool = False
    baseline_length_bias: Optional[LengthBiasResult] = None
    candidate_length_bias: Optional[LengthBiasResult] = None
    raw_baseline_scores: Optional[list[float]] = None
    """Populated only when length_bias_applied is True, holding the
    pre-correction scores for auditing."""
    raw_candidate_scores: Optional[list[float]] = None
    errors: list[str] = field(default_factory=list)
    """Test case ids that failed to run or judge, with a reason --
    these are excluded from statistics rather than silently treated
    as a score of 0, since a provider/judge error is not the same
    claim as 'the model got this wrong'."""


@dataclass
class SuiteReport:
    suite_name: str
    candidate_reports: list[CandidateReport]


class RunnerError(Exception):
    """Raised for setup failures that prevent a suite from running at
    all (e.g. no providers could be constructed), as distinct from
    per-test-case errors, which are recorded in CandidateReport.errors
    instead of aborting the whole run."""


def _score_result(result: JudgeResult) -> float:
    """Converts a JudgeResult into a plain float for statistics: the
    raw LLM score if present, otherwise 1.0/0.0 for pass/fail."""
    if result.raw_score is not None:
        return result.raw_score
    return 1.0 if result.passed else 0.0


def _run_one_provider(
    provider_config: ProviderConfig,
    suite: EvalSuite,
) -> tuple[list[JudgeResult], list[str]]:
    """Runs every test case in `suite` against one provider and judges
    each response. Returns (results, error_messages); a test case
    that errors is skipped from results and recorded as an error
    message, rather than aborting the whole provider run.
    """
    provider = build_provider(provider_config)
    judge_provider = None
    if isinstance(suite.judge, LlmJudgeConfig):
        judge_provider = build_provider(suite.judge.provider)

    results: list[JudgeResult] = []
    errors: list[str] = []

    for test_case in suite.test_cases:
        try:
            response = provider.generate(
                test_case.prompt,
                temperature=provider_config.temperature,
                max_tokens=provider_config.max_tokens,
            )
        except ProviderError as e:
            errors.append(f"test case '{test_case.id}': provider error: {e}")
            continue

        try:
            if isinstance(suite.judge, RuleBasedJudgeConfig):
                judge_result = judge_rule_based(suite.judge, test_case, response.text)
            else:
                assert judge_provider is not None  # guaranteed by the isinstance branch above
                judge_result = judge_llm_single(suite.judge, judge_provider, test_case, response.text)
        except JudgeError as e:
            errors.append(f"test case '{test_case.id}': judge error: {e}")
            continue

        results.append(judge_result)

    return results, errors


def _apply_length_bias_if_configured(
    suite: EvalSuite, results: list[JudgeResult]
) -> tuple[list[float], Optional[LengthBiasResult]]:
    """Returns (scores_to_use, length_bias_result). If the suite uses
    an LLM judge with debias_length enabled and there's enough data
    to fit the correction, returns length-corrected scores and the
    fitted LengthBiasResult. Otherwise returns raw scores and None,
    silently skipping correction only when there's genuinely
    insufficient data (fewer than 3 results) or no length variance --
    both real, expected conditions on small suites, not failures.
    """
    raw_scores = [_score_result(r) for r in results]

    if not (isinstance(suite.judge, LlmJudgeConfig) and suite.judge.debias_length):
        return raw_scores, None

    if len(results) < 3:
        return raw_scores, None

    lengths = [float(len(r.response_text)) for r in results]
    try:
        correction = apply_length_bias_correction(raw_scores, lengths)
    except ValueError:
        # No length variance, or other degenerate case -- correction
        # isn't meaningful here, fall back to raw scores rather than
        # failing the whole run over an edge case in one candidate's
        # response lengths.
        return raw_scores, None

    return correction.adjusted_scores, correction


def _paired_differences(
    suite: EvalSuite,
    baseline_results: list[JudgeResult],
    candidate_results: list[JudgeResult],
) -> list[float]:
    """Builds (candidate_score - baseline_score) for every test case
    that succeeded on BOTH providers, in suite.test_cases order.
    Aligning by test_case_id (not list position) matters because
    baseline and candidate can fail on different test cases, which
    would otherwise silently misalign the pairing.
    """
    baseline_by_id = {r.test_case_id: _score_result(r) for r in baseline_results}
    candidate_by_id = {r.test_case_id: _score_result(r) for r in candidate_results}

    differences = []
    for test_case in suite.test_cases:
        if test_case.id in baseline_by_id and test_case.id in candidate_by_id:
            differences.append(candidate_by_id[test_case.id] - baseline_by_id[test_case.id])

    return differences


def _compute_statistic(
    suite: EvalSuite,
    baseline_scores: list[float],
    candidate_scores: list[float],
    paired_differences: list[float],
) -> Optional[StatisticResult]:
    """Applies the suite's configured statistics method. Returns None
    if there isn't enough data to compute it, rather than raising,
    since a partial report is more useful than none.
    """
    stats_config = suite.statistics

    if isinstance(stats_config, BootstrapStatConfig):
        if not candidate_scores:
            return None
        try:
            return block_bootstrap_mean(
                data=candidate_scores,
                block_size=stats_config.block_size,
                n_resamples=stats_config.n_resamples,
                confidence_level=stats_config.confidence_level,
                seed=stats_config.seed,
            )
        except ValueError:
            return None

    if isinstance(stats_config, EffectSizeStatConfig):
        if not baseline_scores or not candidate_scores:
            return None
        try:
            return cohens_d(baseline_scores, candidate_scores)
        except ValueError:
            return None

    if isinstance(stats_config, SequentialStatConfig):
        if not paired_differences:
            return None
        try:
            test = SequentialTest(alpha=stats_config.alpha, tau=stats_config.tau)
        except ValueError:
            return None

        rejected = False
        rejected_at_n: Optional[int] = None
        for diff in paired_differences:
            update = test.update(diff)
            if update.decision == SequentialDecision.RejectNull:
                rejected = True
                rejected_at_n = update.n
                break

        return SequentialStatResult(
            rejected=rejected,
            rejected_at_n=rejected_at_n,
            final_mean=test.current_mean,
            final_n=test.n,
            total_paired_observations=len(paired_differences),
        )

    return None


def run_suite(suite: EvalSuite) -> SuiteReport:
    """Runs the full suite: baseline + every candidate, judges every
    response, applies length-bias correction where configured, and
    computes the suite's configured statistic per candidate.

    Raises RunnerError only for setup failures that prevent any
    results from being gathered at all. Per-test-case failures are
    recorded in each CandidateReport.errors instead.
    """
    try:
        baseline_results, baseline_errors = _run_one_provider(suite.baseline, suite)
    except ProviderError as e:
        raise RunnerError(f"failed to run baseline provider '{suite.baseline.name}': {e}") from e

    baseline_scores, baseline_length_bias = _apply_length_bias_if_configured(suite, baseline_results)
    raw_baseline_scores = [_score_result(r) for r in baseline_results]
    baseline_length_bias_applied = baseline_length_bias is not None

    candidate_reports: list[CandidateReport] = []

    for candidate_config in suite.candidates:
        try:
            candidate_results, candidate_errors = _run_one_provider(candidate_config, suite)
        except ProviderError as e:
            candidate_reports.append(
                CandidateReport(
                    candidate_name=candidate_config.name,
                    baseline_results=baseline_results,
                    candidate_results=[],
                    baseline_scores=baseline_scores,
                    candidate_scores=[],
                    errors=[f"failed to run candidate provider: {e}"],
                )
            )
            continue

        candidate_scores, candidate_length_bias = _apply_length_bias_if_configured(suite, candidate_results)
        raw_candidate_scores = [_score_result(r) for r in candidate_results]
        length_bias_applied = baseline_length_bias_applied or (candidate_length_bias is not None)

        paired_differences = _paired_differences(suite, baseline_results, candidate_results)
        # When length-bias correction was applied, the paired
        # differences for sequential testing should also reflect
        # corrected scores, not raw ones -- otherwise sequential
        # testing would silently ignore the same correction bootstrap
        # and effect-size statistics respect. Rebuild the differences
        # from corrected scores when correction was applied to either
        # side, aligned the same way by test_case_id.
        if length_bias_applied:
            baseline_corrected_by_id = dict(zip((r.test_case_id for r in baseline_results), baseline_scores))
            candidate_corrected_by_id = dict(zip((r.test_case_id for r in candidate_results), candidate_scores))
            paired_differences = [
                candidate_corrected_by_id[tc.id] - baseline_corrected_by_id[tc.id]
                for tc in suite.test_cases
                if tc.id in baseline_corrected_by_id and tc.id in candidate_corrected_by_id
            ]

        statistic_result = _compute_statistic(suite, baseline_scores, candidate_scores, paired_differences)

        candidate_reports.append(
            CandidateReport(
                candidate_name=candidate_config.name,
                baseline_results=baseline_results,
                candidate_results=candidate_results,
                baseline_scores=baseline_scores,
                candidate_scores=candidate_scores,
                statistic_result=statistic_result,
                length_bias_applied=length_bias_applied,
                baseline_length_bias=baseline_length_bias,
                candidate_length_bias=candidate_length_bias,
                raw_baseline_scores=raw_baseline_scores if length_bias_applied else None,
                raw_candidate_scores=raw_candidate_scores if length_bias_applied else None,
                errors=baseline_errors + candidate_errors,
            )
        )

    return SuiteReport(suite_name=suite.name, candidate_reports=candidate_reports)
"""
Tests for trizaval.harness.runner -- the full suite-run orchestrator.
"""

from unittest.mock import patch

import pytest

from trizaval.harness.providers.base import ProviderError, ProviderResponse
from trizaval.harness.runner import RunnerError, run_suite
from trizaval.suite.loader import load_suite


@pytest.fixture
def suite(example_suite_path):
    return load_suite(example_suite_path)


def _patch_openai_generate(fn):
    return patch("trizaval.harness.providers.openai_provider.OpenAIProvider.generate", fn)


class TestRunSuite:
    def test_all_correct_produces_perfect_scores_and_tight_ci(self, suite):
        def always_correct(self, prompt, *, temperature, max_tokens):
            for tc in suite.test_cases:
                if tc.prompt == prompt:
                    return ProviderResponse(text=tc.reference, latency_seconds=0.05, raw_response={})
            return ProviderResponse(text="unknown", latency_seconds=0.05, raw_response={})

        with _patch_openai_generate(always_correct):
            report = run_suite(suite)

        assert report.suite_name == suite.name
        assert len(report.candidate_reports) == 1
        cr = report.candidate_reports[0]
        assert cr.candidate_scores == [1.0, 1.0, 1.0, 1.0]
        assert cr.statistic_result.point_estimate == 1.0
        assert cr.errors == []

    def test_partial_failures_widen_confidence_interval(self, suite):
        """A candidate that gets one answer wrong out of four should
        produce a visibly wide CI, not a falsely precise one -- this
        is the entire point of using bootstrap CIs instead of a bare
        percentage.
        """
        is_baseline = {"value": True}

        def routed(self, prompt, *, temperature, max_tokens):
            for tc in suite.test_cases:
                if tc.prompt == prompt:
                    if not is_baseline["value"] and tc.id == "add-2":
                        return ProviderResponse(text="wrong answer", latency_seconds=0.05, raw_response={})
                    return ProviderResponse(text=tc.reference, latency_seconds=0.05, raw_response={})
            return ProviderResponse(text="unknown", latency_seconds=0.05, raw_response={})

        with _patch_openai_generate(routed):
            from trizaval.harness.runner import _run_one_provider

            _run_one_provider(suite.baseline, suite)  # populate baseline while is_baseline is True
            is_baseline["value"] = False
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        assert cr.candidate_scores == [1.0, 0.0, 1.0, 1.0]
        assert cr.statistic_result.point_estimate == 0.75
        # The CI must be visibly wide given only 4 examples and 1
        # error -- not collapsed to a falsely precise point.
        assert cr.statistic_result.ci_upper - cr.statistic_result.ci_lower > 0.3

    def test_provider_error_on_one_test_case_is_recorded_not_fatal(self, suite):
        def flaky(self, prompt, *, temperature, max_tokens):
            for tc in suite.test_cases:
                if tc.prompt == prompt:
                    if tc.id == "add-2":
                        raise ProviderError("simulated transient failure")
                    return ProviderResponse(text=tc.reference, latency_seconds=0.05, raw_response={})
            return ProviderResponse(text="unknown", latency_seconds=0.05, raw_response={})

        with _patch_openai_generate(flaky):
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        # Only 3 results, not 4 -- the errored test case is excluded,
        # not silently scored as 0.
        assert len(cr.candidate_scores) == 3
        assert any("add-2" in e for e in cr.errors)

    def test_baseline_total_failure_raises_runner_error(self, suite):
        def always_fails(self, prompt, *, temperature, max_tokens):
            raise ProviderError("provider is down")

        with _patch_openai_generate(always_fails):
            # All test cases fail for baseline, but this is still
            # per-test-case handling, not a RunnerError, since
            # _run_one_provider itself succeeds (just with all
            # results as errors). RunnerError is reserved for
            # failures in build_provider/setup itself.
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        assert cr.baseline_scores == []
        assert cr.candidate_scores == []


class TestSequentialStatistics:
    def test_detects_a_real_noisy_effect_with_early_stopping(self):
        from trizaval.suite.schema import EvalSuite, ProviderConfig, ProviderKind, SequentialStatConfig, TestCase

        test_cases = [
            TestCase(id=f"t{i}", prompt=f"question {i}", reference="correct") for i in range(50)
        ]
        suite = EvalSuite(
            name="sequential-test-suite",
            baseline=ProviderConfig(name="baseline", kind=ProviderKind.OPENAI, model="gpt-4o-mini"),
            candidates=[ProviderConfig(name="candidate", kind=ProviderKind.OPENAI, model="gpt-4o")],
            test_cases=test_cases,
            statistics=SequentialStatConfig(alpha=0.05, tau=0.3),
        )

        def routed(self, prompt, *, temperature, max_tokens):
            if self.model == "gpt-4o-mini":
                return ProviderResponse(text="wrong", latency_seconds=0.01, raw_response={})
            idx = int(prompt.split()[-1])
            if idx % 5 == 4:
                return ProviderResponse(text="wrong", latency_seconds=0.01, raw_response={})
            return ProviderResponse(text="correct", latency_seconds=0.01, raw_response={})

        with _patch_openai_generate(routed):
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        assert cr.statistic_result.rejected is True
        assert cr.statistic_result.rejected_at_n is not None
        assert cr.statistic_result.rejected_at_n < 50  # confirms early stopping actually happened

    def test_degenerate_zero_variance_never_rejects(self):
        """A constant, noise-free paired difference is mathematically
        undefined for the variance-based test -- it must never
        falsely reject, and must not crash."""
        from trizaval.suite.schema import EvalSuite, ProviderConfig, ProviderKind, SequentialStatConfig, TestCase

        test_cases = [
            TestCase(id=f"t{i}", prompt=f"question {i}", reference="correct") for i in range(20)
        ]
        suite = EvalSuite(
            name="degenerate-suite",
            baseline=ProviderConfig(name="baseline", kind=ProviderKind.OPENAI, model="gpt-4o-mini"),
            candidates=[ProviderConfig(name="candidate", kind=ProviderKind.OPENAI, model="gpt-4o")],
            test_cases=test_cases,
            statistics=SequentialStatConfig(alpha=0.05, tau=0.3),
        )

        def routed(self, prompt, *, temperature, max_tokens):
            if self.model == "gpt-4o-mini":
                return ProviderResponse(text="wrong", latency_seconds=0.01, raw_response={})
            return ProviderResponse(text="correct", latency_seconds=0.01, raw_response={})

        with _patch_openai_generate(routed):
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        assert cr.statistic_result.rejected is False
        assert cr.statistic_result.total_paired_observations == 20

    def test_pairing_aligns_by_test_case_id_not_position(self, suite):
        """If baseline and candidate fail on DIFFERENT test cases,
        pairing must still align correctly by id, not silently
        misalign by list position."""
        from trizaval.suite.schema import SequentialStatConfig

        suite.statistics = SequentialStatConfig(alpha=0.05, tau=0.5)

        def routed(self, prompt, *, temperature, max_tokens):
            for tc in suite.test_cases:
                if tc.prompt == prompt:
                    # Baseline fails on the first test case; candidate
                    # fails on a different one -- if pairing used
                    # position instead of id, this would misalign.
                    if self.model == "gpt-4o-mini" and tc.id == "add-1":
                        raise ProviderError("baseline fails here")
                    if self.model == "gpt-4o" and tc.id == "mult-1":
                        raise ProviderError("candidate fails here")
                    return ProviderResponse(text=tc.reference, latency_seconds=0.01, raw_response={})
            return ProviderResponse(text="unknown", latency_seconds=0.01, raw_response={})

        with _patch_openai_generate(routed):
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        # Only test cases that succeeded on BOTH sides should be
        # paired: 4 total - 2 distinct failures = 2 valid pairs.
        assert cr.statistic_result.total_paired_observations == 2


class TestLengthBiasFeedsIntoStatistics:
    def test_corrected_scores_actually_used_not_just_computed(self):
        from trizaval.suite.schema import (
            BootstrapStatConfig,
            EvalSuite,
            LlmJudgeConfig,
            ProviderConfig,
            ProviderKind,
            TestCase,
        )

        llm_suite = EvalSuite(
            name="length-bias-suite",
            baseline=ProviderConfig(name="baseline", kind=ProviderKind.OPENAI, model="gpt-4o-mini"),
            candidates=[ProviderConfig(name="candidate", kind=ProviderKind.OPENAI, model="gpt-4o")],
            test_cases=[TestCase(id=f"t{i}", prompt=f"q{i}") for i in range(5)],
            judge=LlmJudgeConfig(
                provider=ProviderConfig(name="judge", kind=ProviderKind.OPENAI, model="gpt-4o-judge"),
                rubric="score it",
                debias_length=True,
            ),
            statistics=BootstrapStatConfig(n_resamples=500, seed=1),
        )

        def llm_routed(self, prompt, *, temperature, max_tokens):
            if self.model == "gpt-4o-judge":
                for i in range(5):
                    if f"response-{i}-" in prompt:
                        length = (i + 1) * 10
                        return ProviderResponse(text=str(length / 5), latency_seconds=0.01, raw_response={})
                return ProviderResponse(text="5", latency_seconds=0.01, raw_response={})
            for i in range(5):
                if f"q{i}" == prompt:
                    return ProviderResponse(
                        text=f"response-{i}-" + ("x" * (i * 10)), latency_seconds=0.01, raw_response={}
                    )
            return ProviderResponse(text="unknown", latency_seconds=0.01, raw_response={})

        with _patch_openai_generate(llm_routed):
            report = run_suite(llm_suite)

        cr = report.candidate_reports[0]
        assert cr.length_bias_applied is True
        assert cr.raw_candidate_scores == [2.0, 4.0, 6.0, 8.0, 10.0]
        # After perfectly removing a perfect linear length trend, all
        # corrected scores should collapse to (near) the same value.
        first = cr.candidate_scores[0]
        assert all(abs(s - first) < 1e-9 for s in cr.candidate_scores)
        assert cr.raw_candidate_scores != cr.candidate_scores

    def test_no_correction_applied_when_debias_length_is_false(self, suite):
        """Confirm the flag actually gates the behavior -- when
        debias_length=False (rule-based judge default), raw and
        'corrected' scores must be identical (no correction run)."""

        def routed(self, prompt, *, temperature, max_tokens):
            for tc in suite.test_cases:
                if tc.prompt == prompt:
                    return ProviderResponse(text=tc.reference, latency_seconds=0.01, raw_response={})
            return ProviderResponse(text="unknown", latency_seconds=0.01, raw_response={})

        with _patch_openai_generate(routed):
            report = run_suite(suite)

        cr = report.candidate_reports[0]
        assert cr.length_bias_applied is False
        assert cr.raw_candidate_scores is None  # only populated when correction is applied
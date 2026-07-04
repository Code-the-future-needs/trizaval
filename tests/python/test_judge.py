"""
Tests for trizaval.harness.judge -- rule-based and LLM-based judging,
including position-bias and length-bias calibration.
"""

from unittest.mock import MagicMock

import pytest

from trizaval.harness.judge import (
    JudgeError,
    apply_length_bias_correction,
    judge_llm_pairwise,
    judge_llm_single,
    judge_rule_based,
)
from trizaval.harness.providers.base import ProviderResponse
from trizaval.suite.schema import LlmJudgeConfig, ProviderConfig, ProviderKind, RuleBasedJudgeConfig, TestCase


def _test_case(reference="4"):
    return TestCase(id="t1", prompt="What is 2+2?", reference=reference)


def _llm_judge_config():
    return LlmJudgeConfig(
        provider=ProviderConfig(name="judge", kind=ProviderKind.OPENAI, model="gpt-4o"),
        rubric="Score the response quality from 0-10.",
    )


class TestJudgeRuleBased:
    def test_exact_match_passes(self):
        tc = _test_case(reference="4")
        cfg = RuleBasedJudgeConfig(match_type="exact")
        result = judge_rule_based(cfg, tc, "4")
        assert result.passed is True

    def test_exact_match_fails_on_extra_text(self):
        tc = _test_case(reference="4")
        cfg = RuleBasedJudgeConfig(match_type="exact")
        result = judge_rule_based(cfg, tc, "The answer is 4")
        assert result.passed is False

    def test_contains_match_passes_with_extra_text(self):
        tc = _test_case(reference="4")
        cfg = RuleBasedJudgeConfig(match_type="contains")
        result = judge_rule_based(cfg, tc, "The answer is 4.")
        assert result.passed is True

    def test_case_insensitive_by_default(self):
        tc = _test_case(reference="Paris")
        cfg = RuleBasedJudgeConfig(match_type="exact", case_sensitive=False)
        result = judge_rule_based(cfg, tc, "paris")
        assert result.passed is True

    def test_case_sensitive_when_configured(self):
        tc = _test_case(reference="Paris")
        cfg = RuleBasedJudgeConfig(match_type="exact", case_sensitive=True)
        result = judge_rule_based(cfg, tc, "paris")
        assert result.passed is False

    def test_missing_reference_raises(self):
        tc = TestCase(id="t1", prompt="open-ended question")  # no reference
        cfg = RuleBasedJudgeConfig()
        with pytest.raises(JudgeError, match="reference"):
            judge_rule_based(cfg, tc, "any response")


class TestJudgeLlmSingle:
    def test_parses_numeric_score(self):
        mock_judge = MagicMock()
        mock_judge.generate.return_value = ProviderResponse(text="8", latency_seconds=0.1, raw_response={})
        result = judge_llm_single(_llm_judge_config(), mock_judge, _test_case(), "The answer is 4.")
        assert result.raw_score == 8.0
        assert result.passed is True  # 8 >= midpoint of 0-10

    def test_below_midpoint_fails(self):
        mock_judge = MagicMock()
        mock_judge.generate.return_value = ProviderResponse(text="2", latency_seconds=0.1, raw_response={})
        result = judge_llm_single(_llm_judge_config(), mock_judge, _test_case(), "wrong answer")
        assert result.passed is False

    def test_score_clamped_to_configured_range(self):
        mock_judge = MagicMock()
        mock_judge.generate.return_value = ProviderResponse(text="99", latency_seconds=0.1, raw_response={})
        result = judge_llm_single(_llm_judge_config(), mock_judge, _test_case(), "response")
        assert result.raw_score == 10.0  # clamped to score_max

    def test_unparseable_output_raises_judge_error(self):
        mock_judge = MagicMock()
        mock_judge.generate.return_value = ProviderResponse(
            text="I cannot score this.", latency_seconds=0.1, raw_response={}
        )
        with pytest.raises(JudgeError, match="could not parse"):
            judge_llm_single(_llm_judge_config(), mock_judge, _test_case(), "response")


class TestJudgeLlmPairwise:
    def test_biased_judge_is_downgraded_to_tie(self):
        """A judge that always prefers whichever response is shown
        first, regardless of content, must be caught and downgraded
        to a Tie -- this is the entire point of the debiasing method.
        """
        biased_judge = MagicMock()
        biased_judge.generate.side_effect = [
            ProviderResponse(text="A", latency_seconds=0.1, raw_response={}),
            ProviderResponse(text="A", latency_seconds=0.1, raw_response={}),
        ]
        outcome = judge_llm_pairwise(
            _llm_judge_config(), biased_judge, _test_case(), response_a="one", response_b="two"
        )
        assert outcome.position_bias_detected is True

    def test_consistent_judge_preference_is_trusted(self):
        """A judge that consistently prefers the same underlying
        content regardless of position should have that preference
        trusted, not flagged as biased.
        """
        consistent_judge = MagicMock()
        consistent_judge.generate.side_effect = [
            ProviderResponse(text="A", latency_seconds=0.1, raw_response={}),  # original: prefers response_a
            ProviderResponse(text="B", latency_seconds=0.1, raw_response={}),  # swapped: response_a is now "B", still preferred
        ]
        outcome = judge_llm_pairwise(
            _llm_judge_config(), consistent_judge, _test_case(), response_a="one", response_b="two"
        )
        assert outcome.position_bias_detected is False

    def test_judge_provider_error_raises_judge_error(self):
        from trizaval.harness.providers.base import ProviderError

        failing_judge = MagicMock()
        failing_judge.generate.side_effect = ProviderError("connection lost")
        with pytest.raises(JudgeError, match="judge model call failed"):
            judge_llm_pairwise(_llm_judge_config(), failing_judge, _test_case(), response_a="one", response_b="two")


class TestLengthBiasCorrection:
    def test_detects_strong_length_correlation(self):
        scores = [2.0, 4.0, 6.0, 8.0, 10.0]
        lengths = [10.0, 20.0, 30.0, 40.0, 50.0]
        result = apply_length_bias_correction(scores, lengths)
        assert result.correlation > 0.99

    def test_raises_on_insufficient_data(self):
        with pytest.raises(ValueError):
            apply_length_bias_correction([1.0, 2.0], [1.0, 2.0])
"""
Judging: turns a provider's response into a score, given a suite's
JudgeConfig.

Two judge kinds are supported, matching suite.schema.JudgeConfig:

- RuleBasedJudgeConfig: deterministic exact/contains match against a
  test case's `reference`. No model call, no bias concerns.
- LlmJudgeConfig: an LLM scores the response against a rubric. Raw
  LLM-judge scores are known to carry systematic biases (position,
  length) -- see trizaval_core.judge_calibration -- so this module
  applies those corrections rather than trusting raw scores.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from trizaval import (
    LengthBiasResult,
    PairwiseDebiasResult,
    Preference,
    debias_pairwise_judgment,
    length_bias_correction,
)
from trizaval.harness.providers.base import Provider, ProviderError
from trizaval.suite.schema import LlmJudgeConfig, RuleBasedJudgeConfig, TestCase


@dataclass
class JudgeResult:
    """Outcome of judging one (test case, response) pair."""

    test_case_id: str
    passed: bool
    """Binary pass/fail, used by rule-based judging and as a
    thresholded view of LLM scores (score >= midpoint of score
    range)."""
    raw_score: Optional[float] = None
    """Only set for LLM judging; None for rule-based."""
    response_text: str = ""
    metadata: dict = field(default_factory=dict)
    """Extra detail for debugging/auditing -- e.g. which judge kind
    was used, whether position bias was detected for this case."""


class JudgeError(Exception):
    """Raised when judging itself fails (e.g. the judge model call
    errors out), as distinct from a low/failing score, which is a
    normal, valid outcome."""


def judge_rule_based(config: RuleBasedJudgeConfig, test_case: TestCase, response_text: str) -> JudgeResult:
    """Deterministic exact/contains match against `test_case.reference`."""
    if test_case.reference is None:
        raise JudgeError(
            f"test case '{test_case.id}' has no reference value, required for rule_based judging"
        )

    candidate = response_text
    target = test_case.reference

    if not config.case_sensitive:
        candidate = candidate.lower()
        target = target.lower()

    if config.match_type == "exact":
        passed = candidate.strip() == target.strip()
    else:  # "contains"
        passed = target in candidate

    return JudgeResult(
        test_case_id=test_case.id,
        passed=passed,
        response_text=response_text,
        metadata={"judge_kind": "rule_based", "match_type": config.match_type},
    )


_SCORE_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_score(judge_output: str, score_min: float, score_max: float) -> float:
    """Extracts the first number found in the judge model's raw text
    output and clamps it into [score_min, score_max].

    LLM judges are instructed to output a single numeric score, but
    are not perfectly reliable about outputting *only* the number, so
    parsing is deliberately lenient rather than requiring an exact
    format the judge model might not consistently follow.
    """
    match = _SCORE_PATTERN.search(judge_output)
    if match is None:
        raise JudgeError(f"could not parse a numeric score from judge output: {judge_output!r}")
    score = float(match.group())
    return max(score_min, min(score_max, score))


def _call_judge_once(judge_provider: Provider, rubric: str, prompt: str, response_text: str, score_min: float, score_max: float) -> float:
    judge_prompt = (
        f"{rubric}\n\n"
        f"Original prompt:\n{prompt}\n\n"
        f"Response to evaluate:\n{response_text}\n\n"
        f"Respond with only a single number between {score_min} and {score_max}, "
        f"representing your score. No explanation, just the number."
    )
    try:
        result = judge_provider.generate(judge_prompt, temperature=0.0, max_tokens=20)
    except ProviderError as e:
        raise JudgeError(f"judge model call failed: {e}") from e

    return _extract_score(result.text, score_min, score_max)


def judge_llm_single(
    config: LlmJudgeConfig,
    judge_provider: Provider,
    test_case: TestCase,
    response_text: str,
) -> JudgeResult:
    """Scores a single response with an LLM judge. If
    `config.debias_length` is set, the score is NOT length-corrected
    here (length correction requires the full batch of scores across
    a suite run, since it's a regression fit) -- see
    `apply_length_bias_correction` below, which the runner calls
    after collecting all raw scores for a run.
    """
    raw_score = _call_judge_once(
        judge_provider, config.rubric, test_case.prompt, response_text, config.score_min, config.score_max
    )

    midpoint = (config.score_min + config.score_max) / 2
    passed = raw_score >= midpoint

    return JudgeResult(
        test_case_id=test_case.id,
        passed=passed,
        raw_score=raw_score,
        response_text=response_text,
        metadata={"judge_kind": "llm"},
    )


def judge_llm_pairwise(
    config: LlmJudgeConfig,
    judge_provider: Provider,
    test_case: TestCase,
    response_a: str,
    response_b: str,
) -> PairwiseDebiasResult:
    """Judges which of two responses (A, B) is better, debiasing for
    judge position bias by asking twice with the responses swapped
    and reconciling via trizaval_core.judge_calibration.

    Only used when config.debias_position is True; callers wanting a
    single un-debiased pairwise judgment should call the provider
    directly, since that's a materially different (cheaper, less
    rigorous) operation this function deliberately doesn't offer as a
    shortcut.
    """
    pairwise_prompt_template = (
        f"{config.rubric}\n\n"
        f"Original prompt:\n{test_case.prompt}\n\n"
        f"Response A:\n{{first}}\n\n"
        f"Response B:\n{{second}}\n\n"
        f"Which response is better, A or B? Respond with only 'A', 'B', or 'TIE'."
    )

    def parse_preference(text: str) -> Preference:
        cleaned = text.strip().upper()
        if cleaned.startswith("A"):
            return Preference.PrefersA
        if cleaned.startswith("B"):
            return Preference.PrefersB
        return Preference.Tie

    # Original order: A shown first.
    prompt_original = pairwise_prompt_template.format(first=response_a, second=response_b)
    try:
        result_original = judge_provider.generate(prompt_original, temperature=0.0, max_tokens=10)
    except ProviderError as e:
        raise JudgeError(f"judge model call failed (original order): {e}") from e
    preference_original = parse_preference(result_original.text)

    # Swapped order: B shown first. The judge is still asked about
    # "A" and "B" by underlying identity, not by which was shown
    # first -- so we swap which response fills the "first" slot, then
    # invert the raw preference back to the original A/B identity.
    prompt_swapped = pairwise_prompt_template.format(first=response_b, second=response_a)
    try:
        result_swapped = judge_provider.generate(prompt_swapped, temperature=0.0, max_tokens=10)
    except ProviderError as e:
        raise JudgeError(f"judge model call failed (swapped order): {e}") from e
    raw_preference_swapped = parse_preference(result_swapped.text)

    # Invert: in the swapped call, "A" in the judge's response refers
    # to our response_b, and vice versa.
    if raw_preference_swapped == Preference.PrefersA:
        preference_swapped = Preference.PrefersB
    elif raw_preference_swapped == Preference.PrefersB:
        preference_swapped = Preference.PrefersA
    else:
        preference_swapped = Preference.Tie

    return debias_pairwise_judgment(preference_original, preference_swapped)


def apply_length_bias_correction(scores: list[float], response_lengths: list[float]) -> LengthBiasResult:
    """Applies trizaval-core's length-bias correction to a full batch
    of raw LLM-judge scores from one suite run. Called once per run
    by the runner after all individual judgments are collected, not
    per-judgment, since the correction is a regression fit requiring
    the full score/length distribution to be meaningful.
    """
    return length_bias_correction(scores, response_lengths)
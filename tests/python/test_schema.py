"""
Tests for trizaval.suite.schema -- the Pydantic models defining a
valid eval suite config.
"""

import pytest
from pydantic import ValidationError

from trizaval.suite.schema import (
    BootstrapStatConfig,
    EvalSuite,
    ProviderConfig,
    ProviderKind,
    RuleBasedJudgeConfig,
    TestCase,
)


def _minimal_provider(name: str = "p") -> ProviderConfig:
    return ProviderConfig(name=name, kind=ProviderKind.OPENAI, model="gpt-4o-mini")


class TestProviderConfig:
    def test_valid_openai_provider(self):
        p = _minimal_provider()
        assert p.kind == ProviderKind.OPENAI
        assert p.temperature == 0.0  # default

    def test_openai_compatible_requires_base_url(self):
        with pytest.raises(ValidationError, match="base_url"):
            ProviderConfig(name="bad", kind=ProviderKind.OPENAI_COMPATIBLE, model="deepseek-chat")

    def test_openai_compatible_with_base_url_is_valid(self):
        p = ProviderConfig(
            name="deepseek",
            kind=ProviderKind.OPENAI_COMPATIBLE,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
        )
        assert p.base_url == "https://api.deepseek.com/v1"

    def test_temperature_bounds_enforced(self):
        with pytest.raises(ValidationError):
            ProviderConfig(name="p", kind=ProviderKind.OPENAI, model="gpt-4o", temperature=3.0)

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            ProviderConfig(name="p", kind=ProviderKind.OPENAI, model="gpt-4o", max_tokens=0)


class TestEvalSuite:
    def test_valid_minimal_suite(self):
        suite = EvalSuite(
            name="test-suite",
            baseline=_minimal_provider("baseline"),
            candidates=[_minimal_provider("candidate")],
            test_cases=[TestCase(id="t1", prompt="hi")],
        )
        assert suite.name == "test-suite"
        # Defaults applied
        assert isinstance(suite.judge, RuleBasedJudgeConfig)
        assert isinstance(suite.statistics, BootstrapStatConfig)
        assert suite.correction is None

    def test_duplicate_test_case_ids_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            EvalSuite(
                name="dup",
                baseline=_minimal_provider("baseline"),
                candidates=[_minimal_provider("candidate")],
                test_cases=[TestCase(id="t1", prompt="a"), TestCase(id="t1", prompt="b")],
            )

    def test_duplicate_candidate_names_rejected(self):
        with pytest.raises(ValidationError, match="unique"):
            EvalSuite(
                name="dup-candidates",
                baseline=_minimal_provider("baseline"),
                candidates=[_minimal_provider("same"), _minimal_provider("same")],
                test_cases=[TestCase(id="t1", prompt="a")],
            )

    def test_requires_at_least_one_candidate(self):
        with pytest.raises(ValidationError):
            EvalSuite(
                name="no-candidates",
                baseline=_minimal_provider("baseline"),
                candidates=[],
                test_cases=[TestCase(id="t1", prompt="a")],
            )

    def test_requires_at_least_one_test_case(self):
        with pytest.raises(ValidationError):
            EvalSuite(
                name="no-cases",
                baseline=_minimal_provider("baseline"),
                candidates=[_minimal_provider("candidate")],
                test_cases=[],
            )
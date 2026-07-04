"""
Pydantic schema for trizaval eval-suite config files (YAML/TOML).

This is the declarative interface teams actually write by hand: what
to test, which providers to compare, how to judge responses, and
which statistical method from trizaval-core to apply to the results.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class ProviderKind(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

    OPENAI_COMPATIBLE = "openai_compatible"


class ProviderConfig(BaseModel):
    """A single model/provider to run test cases against."""

    name: str = Field(..., description="Human-readable identifier, e.g. 'baseline-gpt4o'")
    kind: ProviderKind
    model: str = Field(..., description="Provider-specific model identifier")
    base_url: Optional[str] = Field(
        default=None,
        description="API endpoint; required for kind='openai_compatible' "
        "(e.g. 'https://api.deepseek.com/v1', 'http://localhost:11434/v1')",
    )
    api_key_env_var: Optional[str] = Field(
        default=None,
        description="Name of the environment variable holding this provider's API "
        "key. Defaults to OPENAI_API_KEY / ANTHROPIC_API_KEY for those kinds if unset; "
        "required to be set explicitly for kind='openai_compatible' unless the "
        "endpoint needs no key (e.g. a local server).",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def openai_compatible_requires_base_url(self) -> "ProviderConfig":
        if self.kind == ProviderKind.OPENAI_COMPATIBLE and not self.base_url:
            raise ValueError("providers with kind='openai_compatible' must set base_url")
        return self

class TestCase(BaseModel):
    """A single evaluation example."""


    __test__ = False

    id: str
    prompt: str
    reference: Optional[str] = Field(
        default=None, description="Expected/reference answer, if applicable to the judge"
    )
    metadata: dict = Field(default_factory=dict)


class RuleBasedJudgeConfig(BaseModel):
    """Deterministic scoring: exact match or substring match against `reference`."""

    kind: Literal["rule_based"] = "rule_based"
    match_type: Literal["exact", "contains"] = "exact"
    case_sensitive: bool = False


class LlmJudgeConfig(BaseModel):
    """LLM-as-judge scoring, subject to the bias-calibration methods
    already implemented in trizaval-core (position bias, length
    bias)."""

    kind: Literal["llm"] = "llm"
    provider: ProviderConfig
    rubric: str = Field(..., description="Instructions given to the judge model")
    score_min: float = 0.0
    score_max: float = 10.0
    debias_position: bool = Field(
        default=True, description="Judge each pairwise comparison twice with positions swapped"
    )
    debias_length: bool = Field(
        default=True, description="Apply length-bias correction to raw scores"
    )


JudgeConfig = Union[RuleBasedJudgeConfig, LlmJudgeConfig]


class BootstrapStatConfig(BaseModel):
    method: Literal["bootstrap"] = "bootstrap"
    block_size: int = Field(default=1, ge=1)
    n_resamples: int = Field(default=2000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: Optional[int] = None


class SequentialStatConfig(BaseModel):
    method: Literal["sequential"] = "sequential"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    tau: float = Field(default=0.5, gt=0.0)


class EffectSizeStatConfig(BaseModel):
    method: Literal["effect_size"] = "effect_size"


StatisticsConfig = Union[BootstrapStatConfig, SequentialStatConfig, EffectSizeStatConfig]


class CorrectionConfig(BaseModel):
    """Applied across all metrics in the suite when there is more
    than one, to control for multiple comparisons."""

    method: Literal["bonferroni", "benjamini_hochberg"] = "benjamini_hochberg"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)


class EvalSuite(BaseModel):
    """Top-level schema for a trizaval eval suite config file."""

    name: str
    description: Optional[str] = None

    baseline: ProviderConfig
    candidates: list[ProviderConfig] = Field(..., min_length=1)

    test_cases: list[TestCase] = Field(..., min_length=1)

    judge: JudgeConfig = Field(default_factory=RuleBasedJudgeConfig)
    statistics: StatisticsConfig = Field(default_factory=BootstrapStatConfig)
    correction: Optional[CorrectionConfig] = None

    @field_validator("test_cases")
    @classmethod
    def unique_test_case_ids(cls, cases: list[TestCase]) -> list[TestCase]:
        ids = [c.id for c in cases]
        if len(ids) != len(set(ids)):
            duplicates = {i for i in ids if ids.count(i) > 1}
            raise ValueError(f"test case ids must be unique, duplicates found: {duplicates}")
        return cases

    @field_validator("candidates")
    @classmethod
    def unique_candidate_names(cls, candidates: list[ProviderConfig]) -> list[ProviderConfig]:
        names = [c.name for c in candidates]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"candidate provider names must be unique, duplicates found: {duplicates}")
        return candidates
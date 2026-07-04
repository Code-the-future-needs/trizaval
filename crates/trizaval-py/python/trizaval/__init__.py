"""
Trizaval: statistically rigorous evaluation for non-deterministic AI systems.

This top-level package re-exports the compiled Rust core. Pure-Python
orchestration (harness, providers, storage) will live alongside this
as separate submodules .
"""

from ._trizaval_core import (
    BootstrapResult,
    block_bootstrap,
    block_bootstrap_mean,
    SequentialDecision,
    SequentialUpdate,
    SequentialTest,
    CorrectionMethod,
    CorrectionResult,
    correct_p_values,
    EffectMagnitude,
    EffectSizeResult,
    cohens_d,
    Preference,
    PairwiseDebiasResult,
    debias_pairwise_judgment,
    LengthBiasResult,
    length_bias_correction,
)

__all__ = [
    "BootstrapResult",
    "block_bootstrap",
    "block_bootstrap_mean",
    "SequentialDecision",
    "SequentialUpdate",
    "SequentialTest",
    "CorrectionMethod",
    "CorrectionResult",
    "correct_p_values",
    "EffectMagnitude",
    "EffectSizeResult",
    "cohens_d",
    "Preference",
    "PairwiseDebiasResult",
    "debias_pairwise_judgment",
    "LengthBiasResult",
    "length_bias_correction",
]
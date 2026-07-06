# LLM judge bias calibration

## The problem

Using an LLM to score or compare AI-generated responses ("LLM-as-judge") is common practice, but LLM judges are not neutral measuring instruments. Two well-documented systematic biases are:

- **Position bias**: in pairwise comparisons (which of two responses is better), judges tend to favor whichever response appears in a particular position, independent of actual content quality (Zheng et al., 2023; Wang et al., 2023).
- **Length bias**: judges tend to assign higher scores to longer responses, independent of whether the additional length reflects better quality (Zheng et al., 2023; Dubois et al., 2024, discussing length-controlled evaluation).

Trusting raw judge output without correcting for these means an eval result may partly reflect these biases rather than genuine quality differences.

## The methods

**Position bias correction.** Each pairwise comparison is judged twice: once with response A shown first, once with the same pair shown in swapped order. If both judgments agree on the same underlying preference (identified by content, not by position), that preference is trusted. If they disagree, the disagreement is itself evidence the judge was tracking position rather than content, and the result is downgraded to an inconclusive tie rather than trusting either judgment. This is a conservative, detection-based correction: it does not attempt to statistically model and subtract out the bias, but instead refuses to draw a conclusion when bias is caught in the act.

**Length bias correction.** An ordinary least squares regression is fit between judge scores and response lengths across a batch of judged examples: `score = intercept + slope * length`. The **residual** — the portion of each score not explained by length — is then re-centered on the original mean score, producing length-adjusted scores on a comparable scale to the raw scores. A strong correlation between score and length indicates substantial length bias in that batch; a correlation near zero indicates the judge's scores are not being meaningfully driven by length.

## Implementation

See `crates/trizaval-core/src/judge_calibration.rs`, functions `debias_pairwise_judgment` and `length_bias_correction`.

- Position-bias debiasing is a pure function operating on two already-obtained preferences (`Preference::PrefersA`, `PrefersB`, or `Tie`); it does not itself call a judge model. Callers are responsible for obtaining both judgments with positions correctly swapped, and for translating the judge's raw positional answer back into a stable, position-independent identity before calling this function — the function assumes both `original_order` and `swapped_order` are already expressed in terms of the same underlying A/B identity, not "whichever was shown first."
- Length-bias correction requires at least 3 (score, length) pairs to fit a regression, and requires genuine variance in the length values (otherwise a slope cannot be estimated); both conditions produce an explicit error rather than a degenerate fit.
- In Trizaval's harness (`crates/trizaval-py/python/trizaval/harness/judge.py` and `runner.py`), length-bias correction is applied per-provider over the full batch of that provider's scores in one run, and the corrected scores, not the raw ones, are what feed into whichever statistic the suite is configured to compute, when `debias_length` is enabled in the suite's judge configuration.

## Assumptions and limitations

- Position-bias correction doubles the number of judge calls required for pairwise comparisons (each pair judged twice). This is a deliberate, documented cost of rigor, not an oversight.
- Length-bias correction assumes a roughly linear relationship between length and score. If the true relationship is non-linear (e.g. a judge penalizes both very short and very long responses), a linear fit will only partially correct the bias. This is not currently addressed with a non-linear alternative.
- Neither method corrects for other known LLM-judge biases, such as self-preference bias (a model favoring outputs similar to its own style) or verbosity-independent stylistic biases; these are out of scope for the current implementation.

## References

- Zheng, L., Chiang, W.-L., Sheng, Y., et al. (2023). "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *NeurIPS 2023 Datasets and Benchmarks Track*.
- Wang, P., Li, L., Chen, L., et al. (2023). "Large Language Models are not Fair Evaluators." *arXiv:2305.17926*.
- Dubois, Y., Galambosi, B., Liang, P., & Hashimoto, T. B. (2024). "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." *arXiv:2404.04475*.
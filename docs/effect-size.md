# Effect size estimation

## The problem

Statistical significance answers "is this difference likely to be real?" It does not answer "does this difference matter?" With a large enough sample, even a trivial difference between two models can be statistically significant. Effect size measures the magnitude of a difference in a way that is comparable across different metrics, sample sizes, and eval sets.

## The method

**Cohen's d** (Cohen, 1988) standardizes the difference between two group means by the pooled standard deviation: d = (mean_treatment - mean_baseline) / pooled_std_dev

where the pooled standard deviation weights each group's variance by its degrees of freedom. Conventional (though contested and context-dependent) interpretive thresholds are: |d| < 0.2 negligible, < 0.5 small, < 0.8 medium, otherwise large.

Cohen's d is known to be **biased upward** for small samples (a systematic overestimate of the true population effect size). **Hedges' g** (Hedges, 1981) applies a correction factor:
J = 1 - 3 / (4(n1 + n2 - 2) - 1)
g = d * J

For large samples, `J` approaches 1 and `g` converges to `d`; for small samples (roughly fewer than 20 observations per group, a common regime for eval test suites), the correction meaningfully shrinks the estimate toward zero.

## Implementation

See `crates/trizaval-core/src/effect_size.rs`, function `cohens_d`.

- Both `d` and `g` are always returned together; callers should prefer `g` when group sizes are small.
- Requires at least 2 observations per group (a variance cannot be estimated from a single point) and a non-zero pooled standard deviation. Two groups with identical, constant values produce a genuinely undefined effect size (division by zero variance) and the function returns an explicit error rather than `NaN` or `Infinity`.
- The sign of the result indicates direction: positive means the treatment/candidate group scored higher than baseline; negative means lower.

## Assumptions and limitations

- Cohen's d assumes the two groups have roughly similar variances (homogeneity of variance); when they differ substantially, the pooled standard deviation is a less meaningful denominator, and alternative formulations (e.g. Glass's delta, which uses only the control group's standard deviation) may be more appropriate. These are not currently implemented.
- The conventional magnitude thresholds (negligible/small/medium/large) are widely used defaults but were derived from behavioral science contexts; what counts as a "large" effect in an AI eval context is domain-specific and these labels should be treated as a rough heuristic, not a universal standard.

## References

- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
- Hedges, L. V. (1981). "Distribution Theory for Glass's Estimator of Effect Size and Related Estimators." *Journal of Educational Statistics*, 6(2), 107–128.
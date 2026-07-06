# Sequential hypothesis testing (mixture SPRT)

## The problem

A fixed-sample hypothesis test (e.g. a t-test computed once on a predetermined sample size) is only valid if you look at the result exactly once, after collecting exactly that many observations. In practice, teams often check results as data accumulates and stop once something looks significant. This practice, sometimes called "peeking," inflates the false positive rate well above the nominal significance level, because each additional look is another chance for noise to appear significant.

## The method

Trizaval implements a **mixture sequential probability ratio test** (mSPRT), based on Robbins (1970) and the mixture-martingale framework popularized more recently for "always-valid" online A/B testing (Johari, Koomen, Pekelis & Walsh, 2017). Instead of a single decision at a fixed sample size, the test maintains a likelihood ratio that is updated after every new observation, using a Gaussian mixture prior with variance `tau^2` over the possible effect size.

The test statistic is: Λ_n = sqrt(σ² / (σ² + nτ²)) * exp( n²τ²x̄² / (2σ²(σ² + nτ²)) )

where `n` is the number of observations, `x̄` is their running mean, `σ²` is the (estimated) observation variance, and `τ²` is the prior variance representing the effect size worth detecting. The null hypothesis (no effect) is rejected once `Λ_n ≥ 1/alpha`, and this decision rule is valid to stop on at any time `n`, not only a single predetermined sample size, without inflating the Type I error rate.

`tau` is a tuning parameter: smaller values give more power to detect small effects at the cost of less power for large ones, and vice versa. It should be chosen to represent the smallest effect size that would be practically meaningful to detect.

## Implementation

See `crates/trizaval-core/src/sequential.rs`, struct `SequentialTest`.

- Variance is estimated online via **Welford's algorithm**, avoiding numerical instability from naive running sums of squares.
- With fewer than 2 observations, or when the estimated variance is too close to zero to divide by safely, the test returns `Continue` rather than attempting an ill-posed computation. This is a real, tested edge case: identical, zero-variance observations never spuriously reject.
- The typical use in Trizaval's harness is on **paired differences** (candidate score minus baseline score per test case), so the null hypothesis being tested is "the paired difference has mean zero."

## Assumptions and limitations

- The variance estimate is a plug-in (empirical) estimate, not a known fixed value. This means the validity guarantee is asymptotic — accurate once a reasonable amount of data has accumulated, not exact from the very first observations. This is standard practice but worth stating explicitly.
- The test assumes roughly Gaussian-distributed observations (or differences) for the likelihood ratio's exact form to hold; it is reasonably robust to mild departures from normality given the central limit theorem's effect on the running mean, but is not distribution-free.
- Choosing `tau` requires domain judgment; the test's power is sensitive to this choice, particularly for detecting effects much smaller or larger than `tau`.

## References

- Robbins, H. (1970). "Statistical Methods Related to the Law of the Iterated Logarithm." *The Annals of Mathematical Statistics*, 41(5), 1397–1409.
- Johari, R., Koomen, P., Pekelis, L., & Walsh, D. (2017). "Peeking at A/B Tests: Why It Matters, and What to Do About It." *KDD '17*.
- Wald, A. (1945). "Sequential Tests of Statistical Hypotheses." *The Annals of Mathematical Statistics*, 16(2), 117–186. (The original SPRT, of which the mixture SPRT is a generalization avoiding the need to pre-specify a single alternative hypothesis.)
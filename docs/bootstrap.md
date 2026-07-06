# Block bootstrap confidence intervals

## The problem

Evaluation scores from an AI system are a sample, not the full population of possible outputs. Reporting a single number, like "mean accuracy is 0.84," implies more precision than a finite sample actually supports. A confidence interval expresses the range of plausible true values given the sample size and its variability.

The ordinary bootstrap (Efron, 1979) estimates this by resampling the data with replacement many times and computing the statistic of interest on each resample. Its assumption is that observations are independent and identically distributed (i.i.d.). LLM evaluation data frequently violates this: consecutive test cases drawn from the same topic, template, or difficulty tier tend to produce correlated scores. Treating them as independent understates the true uncertainty.

## The method

The **moving block bootstrap** (Künsch, 1989; Liu & Singh, 1992) resamples contiguous blocks of `block_size` consecutive observations, with replacement, instead of single points. This preserves local correlation structure within each block. Concatenating resampled blocks until the resample reaches the original sample size, then computing the statistic on that resample, and repeating many times, produces a distribution of the statistic that better reflects genuine sampling uncertainty when observations are correlated.

Setting `block_size = 1` recovers the ordinary i.i.d. bootstrap as a special case.

The confidence interval is constructed using the **percentile method**: after sorting the resampled statistics, the interval bounds are the `alpha/2` and `1 - alpha/2` quantiles, where `alpha = 1 - confidence_level`.

## Implementation

See `crates/trizaval-core/src/bootstrap.rs`, function `block_bootstrap`.

- The statistic function is generic (`Fn(&[f64]) -> Result<f64, String>`), so any reduction can be bootstrapped, not only the mean. `bootstrap::mean` is provided as the common case.
- The statistic is fallible, not because Rust-native statistics fail, but because the Python and R bindings allow user-supplied callables as the statistic, which can raise exceptions or return invalid values. The same interface is used uniformly for both native and foreign statistics.
- Resampling uses `ChaCha8Rng`, a deterministic, seedable pseudorandom generator, so identical inputs and seeds produce bit-identical results across every language binding (Rust, Python, R, WebAssembly).
- Valid block starting positions are `0..=(n - block_size)`, guaranteeing every block stays within the original data.

## Assumptions and limitations

- The percentile method is simple and widely used but is known to have coverage issues (the true coverage can deviate from the nominal confidence level) for small samples or highly skewed statistics. More sophisticated methods (BCa, studentized bootstrap) exist and are not currently implemented.
- Choosing `block_size` requires judgment about the correlation structure of the data. Too small underestimates correlation (converging to the ordinary bootstrap); too large reduces the effective number of independent blocks and can widen intervals unnecessarily. No automatic block-size selection is implemented.
- The number of resamples (`n_resamples`) trades off precision against computation time. 2000 is a common default; fewer resamples produce coarser, less stable interval estimates.

## References

- Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife." *The Annals of Statistics*, 7(1), 1–26.
- Künsch, H. R. (1989). "The Jackknife and the Bootstrap for General Stationary Observations." *The Annals of Statistics*, 17(3), 1217–1241.
- Liu, R. Y., & Singh, K. (1992). "Moving Blocks Jackknife and Bootstrap Capture Weak Dependence." In *Exploring the Limits of Bootstrap*, 225–248.
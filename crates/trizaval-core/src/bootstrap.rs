//! Block bootstrap for confidence interval estimation on
//! (possibly correlated) evaluation data.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Result of a bootstrap confidence interval estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapResult {
    /// The statistic computed on the original, unresampled data.
    pub point_estimate: f64,
    /// Lower bound of the confidence interval (percentile method).
    pub ci_lower: f64,
    /// Upper bound of the confidence interval (percentile method).
    pub ci_upper: f64,
    /// Confidence level used, e.g. 0.95 for a 95% CI.
    pub confidence_level: f64,
    /// Number of bootstrap resamples performed.
    pub n_resamples: usize,
}

/// Errors that can occur during bootstrap estimation.
#[derive(Debug, Clone, PartialEq)]
pub enum BootstrapError {
    EmptyData,
    InvalidBlockSize,
    InvalidConfidenceLevel,
    InvalidResampleCount,
    /// The statistic function itself failed (e.g. a Python callback
    /// raised an exception, or returned a non-numeric value). Carries
    /// a human-readable description of what went wrong.
    StatisticFailed(String),
}

impl std::fmt::Display for BootstrapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BootstrapError::EmptyData => write!(f, "input data must not be empty"),
            BootstrapError::InvalidBlockSize => {
                write!(f, "block_size must be >= 1 and <= data.len()")
            }
            BootstrapError::InvalidConfidenceLevel => {
                write!(f, "confidence_level must be in (0.0, 1.0)")
            }
            BootstrapError::InvalidResampleCount => {
                write!(f, "n_resamples must be >= 1")
            }
            BootstrapError::StatisticFailed(msg) => {
                write!(f, "statistic function failed: {msg}")
            }
        }
    }
}

impl std::error::Error for BootstrapError {}

/// Performs a moving block bootstrap on `data`, computing a confidence
/// interval for whatever `statistic` measures.
///
/// Block bootstrap resamples contiguous *blocks* of `block_size`
/// consecutive observations (with replacement) instead of single
/// points, which preserves local correlation structure. This matters
/// for LLM eval data, where errors on similar/adjacent prompts are
/// rarely independent — a plain point-wise bootstrap would understate
/// the true uncertainty.
///
/// Use `block_size = 1` to recover an ordinary i.i.d. bootstrap.
///
/// `seed = Some(x)` gives fully reproducible results across runs.
pub fn block_bootstrap<F>(
    data: &[f64],
    block_size: usize,
    n_resamples: usize,
    confidence_level: f64,
    statistic: F,
    seed: Option<u64>,
) -> Result<BootstrapResult, BootstrapError>
where
    F: Fn(&[f64]) -> Result<f64, String>,
{
    if data.is_empty() {
        return Err(BootstrapError::EmptyData);
    }
    if block_size == 0 || block_size > data.len() {
        return Err(BootstrapError::InvalidBlockSize);
    }
    if !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err(BootstrapError::InvalidConfidenceLevel);
    }
    if n_resamples == 0 {
        return Err(BootstrapError::InvalidResampleCount);
    }

    let n = data.len();
    let point_estimate = statistic(data).map_err(BootstrapError::StatisticFailed)?;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    // Valid starting indices for a block of `block_size` are
    // 0..=(n - block_size), so every block stays inside the data.
    let max_start = n - block_size;

    let mut resample_stats: Vec<f64> = Vec::with_capacity(n_resamples);
    let mut buffer: Vec<f64> = Vec::with_capacity(n + block_size);

    for _ in 0..n_resamples {
        buffer.clear();
        while buffer.len() < n {
            let start = rng.gen_range(0..=max_start);
            buffer.extend_from_slice(&data[start..start + block_size]);
        }
        buffer.truncate(n);
        let stat = statistic(&buffer).map_err(BootstrapError::StatisticFailed)?;
        resample_stats.push(stat);
    }

    resample_stats.sort_by(|a, b| a.partial_cmp(b).expect("statistic produced NaN"));

    let alpha = 1.0 - confidence_level;
    let lower_idx = ((alpha / 2.0) * n_resamples as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_resamples as f64).ceil() as usize;
    let upper_idx = upper_idx.min(n_resamples - 1);

    Ok(BootstrapResult {
        point_estimate,
        ci_lower: resample_stats[lower_idx],
        ci_upper: resample_stats[upper_idx],
        confidence_level,
        n_resamples,
    })
}

/// Convenience helper: arithmetic mean, the most common eval statistic
/// (e.g. mean accuracy, mean judge score).
///
/// Returns `Result` (always `Ok` here) to match the fallible
/// `F: Fn(&[f64]) -> Result<f64, String>` signature required by
/// `block_bootstrap`. That signature exists because statistics
/// supplied from Python can genuinely fail (raised exceptions, wrong
/// return type) — one uniform interface is used for both native Rust
/// and Python-supplied statistics, rather than maintaining two
/// parallel code paths.
pub fn mean(data: &[f64]) -> Result<f64, String> {
    if data.is_empty() {
        return Err("mean is undefined for empty data".to_string());
    }
    Ok(data.iter().sum::<f64>() / data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_data() {
        let result = block_bootstrap(&[], 1, 100, 0.95, mean, Some(42));
        assert_eq!(result, Err(BootstrapError::EmptyData));
    }

    #[test]
    fn rejects_block_size_larger_than_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = block_bootstrap(&data, 10, 100, 0.95, mean, Some(42));
        assert_eq!(result, Err(BootstrapError::InvalidBlockSize));
    }

    #[test]
    fn rejects_invalid_confidence_level() {
        let data = vec![1.0, 2.0, 3.0];
        let result = block_bootstrap(&data, 1, 100, 1.5, mean, Some(42));
        assert_eq!(result, Err(BootstrapError::InvalidConfidenceLevel));
    }

    #[test]
    fn point_estimate_matches_direct_statistic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = block_bootstrap(&data, 1, 1000, 0.95, mean, Some(42)).unwrap();
        assert!((result.point_estimate - 3.0).abs() < 1e-9);
    }

    #[test]
    fn ci_contains_point_estimate() {
        let data = vec![0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.92, 0.7];
        let result = block_bootstrap(&data, 2, 2000, 0.95, mean, Some(7)).unwrap();
        assert!(result.ci_lower <= result.point_estimate);
        assert!(result.point_estimate <= result.ci_upper);
    }

    #[test]
    fn same_seed_gives_reproducible_results() {
        let data = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0];
        let r1 = block_bootstrap(&data, 2, 500, 0.9, mean, Some(123)).unwrap();
        let r2 = block_bootstrap(&data, 2, 500, 0.9, mean, Some(123)).unwrap();
        assert_eq!(r1, r2);
    }

    #[test]
    fn different_seeds_can_give_different_results() {
        // Non-round, high-precision values make exact percentile
        // collisions between independent seeds vanishingly unlikely,
        // avoiding flaky coincidental matches.
        let data = vec![
            1.13171, 5.29283, 3.71519, 8.44561, 2.90233, 9.16729, 4.55671,
            6.33128, 7.82441, 0.51937,
        ];
        let r1 = block_bootstrap(&data, 2, 2000, 0.9, mean, Some(1)).unwrap();
        let r2 = block_bootstrap(&data, 2, 2000, 0.9, mean, Some(2)).unwrap();
        // point_estimate is computed on the original data, not the
        // resamples, so it is seed-independent by design — only the
        // CI bounds should differ here.
        assert!(
            r1.ci_lower != r2.ci_lower || r1.ci_upper != r2.ci_upper,
            "expected different seeds to produce different CI bounds"
        );
    }

#[test]
    fn wider_confidence_level_gives_wider_interval() {
        let data = vec![0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.92, 0.7];
        let narrow = block_bootstrap(&data, 1, 3000, 0.80, mean, Some(99)).unwrap();
        let wide = block_bootstrap(&data, 1, 3000, 0.99, mean, Some(99)).unwrap();
        let narrow_width = narrow.ci_upper - narrow.ci_lower;
        let wide_width = wide.ci_upper - wide.ci_lower;
        assert!(wide_width >= narrow_width);
    }

    #[test]
    fn propagates_statistic_errors_instead_of_panicking() {
        let data = vec![1.0, 2.0, 3.0];
        let failing_stat = |_: &[f64]| -> Result<f64, String> { Err("boom".to_string()) };
        let result = block_bootstrap(&data, 1, 10, 0.95, failing_stat, Some(1));
        match result {
            Err(BootstrapError::StatisticFailed(msg)) => assert_eq!(msg, "boom"),
            other => panic!("expected StatisticFailed, got {other:?}"),
        }
    }
}
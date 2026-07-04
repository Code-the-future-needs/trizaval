//! Effect size estimation for comparing two sets of evaluation
//! scores (e.g. baseline model vs. candidate model).
//!
//! A statistically significant difference doesn't tell you whether
//! it *matters* - a p-value can be tiny with enough samples even for
//! a trivial difference. Effect size gives the magnitude, in units
//! that are comparable across different metrics and eval sets.

/// Errors that can occur during effect size estimation.
#[derive(Debug, Clone, PartialEq)]
pub enum EffectSizeError {
    InsufficientData,
    ZeroVariance,
}

impl std::fmt::Display for EffectSizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EffectSizeError::InsufficientData => {
                write!(f, "each group must have at least 2 observations")
            }
            EffectSizeError::ZeroVariance => write!(
                f,
                "pooled standard deviation is zero - groups have no variance, effect size is undefined"
            ),
        }
    }
}

impl std::error::Error for EffectSizeError {}

/// Qualitative interpretation bucket for a standardized effect size,
/// using Cohen's conventional (if somewhat arbitrary) thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectMagnitude {
    Negligible,
    Small,
    Medium,
    Large,
}

fn interpret(abs_d: f64) -> EffectMagnitude {
    if abs_d < 0.2 {
        EffectMagnitude::Negligible
    } else if abs_d < 0.5 {
        EffectMagnitude::Small
    } else if abs_d < 0.8 {
        EffectMagnitude::Medium
    } else {
        EffectMagnitude::Large
    }
}

/// Result of an effect size computation.
#[derive(Debug, Clone, PartialEq)]
pub struct EffectSizeResult {
    /// Cohen's d: (mean_treatment - mean_baseline) / pooled_std_dev.
    pub cohens_d: f64,
    /// Hedges' g: Cohen's d with a small-sample bias correction
    /// applied. Prefer this over raw Cohen's d when group sizes are
    /// small (roughly n < 20 per group), since Cohen's d
    /// systematically overestimates effect size in that regime.
    pub hedges_g: f64,
    pub magnitude: EffectMagnitude,
    pub n_baseline: usize,
    pub n_treatment: usize,
}

/// Computes Cohen's d and Hedges' g for the difference between
/// `treatment` and `baseline` sample means, using the pooled
/// standard deviation.
///
/// Positive values mean `treatment` scored higher than `baseline`;
/// negative means lower.
pub fn cohens_d(
    baseline: &[f64],
    treatment: &[f64],
) -> Result<EffectSizeResult, EffectSizeError> {
    let n1 = baseline.len();
    let n2 = treatment.len();

    if n1 < 2 || n2 < 2 {
        return Err(EffectSizeError::InsufficientData);
    }

    let mean1 = baseline.iter().sum::<f64>() / n1 as f64;
    let mean2 = treatment.iter().sum::<f64>() / n2 as f64;

    let var1 = sample_variance(baseline, mean1);
    let var2 = sample_variance(treatment, mean2);

    // Pooled standard deviation, weighted by degrees of freedom of
    // each group.
    let pooled_var = ((n1 as f64 - 1.0) * var1 + (n2 as f64 - 1.0) * var2)
        / (n1 as f64 + n2 as f64 - 2.0);
    let pooled_std = pooled_var.sqrt();

    if pooled_std <= 1e-12 {
        return Err(EffectSizeError::ZeroVariance);
    }

    let d = (mean2 - mean1) / pooled_std;

    // Hedges' correction factor J, applied to convert d -> g.
    // Standard approximation: J = 1 - 3 / (4*(n1+n2-2) - 1)
    let df = n1 as f64 + n2 as f64 - 2.0;
    let correction = 1.0 - 3.0 / (4.0 * df - 1.0);
    let g = d * correction;

    Ok(EffectSizeResult {
        cohens_d: d,
        hedges_g: g,
        magnitude: interpret(d.abs()),
        n_baseline: n1,
        n_treatment: n2,
    })
}

fn sample_variance(data: &[f64], mean: f64) -> f64 {
    let n = data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_insufficient_data() {
        let result = cohens_d(&[1.0], &[1.0, 2.0]);
        assert_eq!(result.unwrap_err(), EffectSizeError::InsufficientData);
    }

    #[test]
    fn rejects_zero_variance() {
        // Both groups are constant, and identical, so pooled std dev is 0.
        let result = cohens_d(&[5.0, 5.0, 5.0], &[5.0, 5.0, 5.0]);
        assert_eq!(result.unwrap_err(), EffectSizeError::ZeroVariance);
    }

    #[test]
    fn zero_difference_gives_zero_effect_size() {
        let baseline = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = cohens_d(&baseline, &treatment).unwrap();
        assert!(result.cohens_d.abs() < 1e-9);
        assert_eq!(result.magnitude, EffectMagnitude::Negligible);
    }

    #[test]
    fn positive_direction_when_treatment_scores_higher() {
        let baseline = vec![0.5, 0.6, 0.55, 0.52, 0.58];
        let treatment = vec![0.9, 0.92, 0.88, 0.91, 0.89];
        let result = cohens_d(&baseline, &treatment).unwrap();
        assert!(result.cohens_d > 0.0);
        assert_eq!(result.magnitude, EffectMagnitude::Large);
    }

    #[test]
    fn negative_direction_when_treatment_scores_lower() {
        let baseline = vec![0.9, 0.92, 0.88, 0.91, 0.89];
        let treatment = vec![0.5, 0.6, 0.55, 0.52, 0.58];
        let result = cohens_d(&baseline, &treatment).unwrap();
        assert!(result.cohens_d < 0.0);
    }

    #[test]
    fn hedges_g_is_smaller_magnitude_than_cohens_d() {
        // Hedges' g should shrink the estimate toward zero relative
        // to Cohen's d (the small-sample bias correction), for any
        // finite sample.
        let baseline = vec![0.5, 0.6, 0.55, 0.52, 0.58];
        let treatment = vec![0.9, 0.92, 0.88, 0.91, 0.89];
        let result = cohens_d(&baseline, &treatment).unwrap();
        assert!(result.hedges_g.abs() < result.cohens_d.abs());
    }

    #[test]
    fn magnitude_thresholds_are_correct() {
        assert_eq!(interpret(0.1), EffectMagnitude::Negligible);
        assert_eq!(interpret(0.3), EffectMagnitude::Small);
        assert_eq!(interpret(0.6), EffectMagnitude::Medium);
        assert_eq!(interpret(1.0), EffectMagnitude::Large);
    }

    #[test]
    fn sample_sizes_recorded_correctly() {
        let baseline = vec![1.0, 2.0, 3.0];
        let treatment = vec![4.0, 5.0, 6.0, 7.0];
        let result = cohens_d(&baseline, &treatment).unwrap();
        assert_eq!(result.n_baseline, 3);
        assert_eq!(result.n_treatment, 4);
    }
}
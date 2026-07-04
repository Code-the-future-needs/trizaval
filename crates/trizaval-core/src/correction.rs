//! Multiple-comparisons correction for evaluation suites that check
//! many metrics at once.
//!
//! Running an eval suite with many metrics and flagging "significant"
//! ones without correction inflates false positives - with 20
//! independent metrics at alpha=0.05, you'd expect roughly one false
//! "improvement" by pure chance even if nothing changed. These
//! methods correct for that.

/// Which correction method to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionMethod {
    /// Controls family-wise error rate (probability of *any* false
    /// positive). Strict - best when false positives are costly.
    Bonferroni,
    /// Controls false discovery rate (expected proportion of false
    /// positives among rejected hypotheses). More powerful than
    /// Bonferroni when testing many metrics at once.
    BenjaminiHochberg,
}

/// Result of applying a correction to a set of p-values.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionResult {
    /// Adjusted p-values, in the same order as the input.
    pub adjusted_p_values: Vec<f64>,
    /// Whether each hypothesis is rejected (i.e. "significant") at
    /// `alpha` after correction, same order as input.
    pub rejected: Vec<bool>,
    pub method: CorrectionMethod,
    pub alpha: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CorrectionError {
    EmptyInput,
    InvalidAlpha,
    InvalidPValue,
}

impl std::fmt::Display for CorrectionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrectionError::EmptyInput => write!(f, "p_values must not be empty"),
            CorrectionError::InvalidAlpha => write!(f, "alpha must be in (0.0, 1.0)"),
            CorrectionError::InvalidPValue => {
                write!(f, "all p-values must be in [0.0, 1.0]")
            }
        }
    }
}

impl std::error::Error for CorrectionError {}

/// Applies the chosen multiple-comparisons correction to `p_values`.
pub fn correct_p_values(
    p_values: &[f64],
    alpha: f64,
    method: CorrectionMethod,
) -> Result<CorrectionResult, CorrectionError> {
    if p_values.is_empty() {
        return Err(CorrectionError::EmptyInput);
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(CorrectionError::InvalidAlpha);
    }
    if p_values.iter().any(|&p| !(0.0..=1.0).contains(&p)) {
        return Err(CorrectionError::InvalidPValue);
    }

    let (adjusted_p_values, rejected) = match method {
        CorrectionMethod::Bonferroni => bonferroni(p_values, alpha),
        CorrectionMethod::BenjaminiHochberg => benjamini_hochberg(p_values, alpha),
    };

    Ok(CorrectionResult {
        adjusted_p_values,
        rejected,
        method,
        alpha,
    })
}

fn bonferroni(p_values: &[f64], alpha: f64) -> (Vec<f64>, Vec<bool>) {
    let m = p_values.len() as f64;
    let adjusted: Vec<f64> = p_values.iter().map(|&p| (p * m).min(1.0)).collect();
    let rejected: Vec<bool> = adjusted.iter().map(|&p| p <= alpha).collect();
    (adjusted, rejected)
}

fn benjamini_hochberg(p_values: &[f64], alpha: f64) -> (Vec<f64>, Vec<bool>) {
    let m = p_values.len();

    // Sort indices by p-value ascending, keeping track of original position.
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("p-value was NaN"));

    // Raw BH-adjusted q-values at each rank: q_(i) = p_(i) * m / (i+1)
    let mut adjusted_sorted: Vec<f64> = indexed
        .iter()
        .enumerate()
        .map(|(rank, &(_, p))| (p * m as f64 / (rank as f64 + 1.0)).min(1.0))
        .collect();

    // Enforce monotonicity: adjusted q-values must not decrease as
    // rank decreases, so we sweep from the largest rank down, taking
    // running minimums.
    for i in (0..adjusted_sorted.len() - 1).rev() {
        if adjusted_sorted[i] > adjusted_sorted[i + 1] {
            adjusted_sorted[i] = adjusted_sorted[i + 1];
        }
    }

    // Map back to original order.
    let mut adjusted = vec![0.0; m];
    for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
        adjusted[orig_idx] = adjusted_sorted[rank];
    }

    let rejected: Vec<bool> = adjusted.iter().map(|&q| q <= alpha).collect();
    (adjusted, rejected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_input() {
        let result = correct_p_values(&[], 0.05, CorrectionMethod::Bonferroni);
        assert_eq!(result.unwrap_err(), CorrectionError::EmptyInput);
    }

    #[test]
    fn rejects_invalid_alpha() {
        let result = correct_p_values(&[0.01], 1.5, CorrectionMethod::Bonferroni);
        assert_eq!(result.unwrap_err(), CorrectionError::InvalidAlpha);
    }

    #[test]
    fn rejects_invalid_p_value() {
        let result = correct_p_values(&[0.5, 1.2], 0.05, CorrectionMethod::Bonferroni);
        assert_eq!(result.unwrap_err(), CorrectionError::InvalidPValue);
    }

    #[test]
    fn bonferroni_scales_by_count() {
        let p_values = vec![0.01, 0.02, 0.03];
        let result = correct_p_values(&p_values, 0.05, CorrectionMethod::Bonferroni).unwrap();
        // m = 3, so adjusted = p * 3
        assert!((result.adjusted_p_values[0] - 0.03).abs() < 1e-9);
        assert!((result.adjusted_p_values[1] - 0.06).abs() < 1e-9);
        assert!((result.adjusted_p_values[2] - 0.09).abs() < 1e-9);
    }

    #[test]
    fn bonferroni_caps_at_one() {
        let p_values = vec![0.9, 0.9];
        let result = correct_p_values(&p_values, 0.05, CorrectionMethod::Bonferroni).unwrap();
        assert!(result.adjusted_p_values.iter().all(|&p| p <= 1.0));
    }

    #[test]
    fn bh_is_less_conservative_than_bonferroni() {
        // A batch where several p-values are genuinely small — BH
        // should reject at least as many, typically more, than
        // Bonferroni on the same data.
        let p_values = vec![0.001, 0.004, 0.008, 0.02, 0.6, 0.7, 0.8, 0.9];
        let bonf = correct_p_values(&p_values, 0.05, CorrectionMethod::Bonferroni).unwrap();
        let bh = correct_p_values(&p_values, 0.05, CorrectionMethod::BenjaminiHochberg).unwrap();

        let bonf_rejections = bonf.rejected.iter().filter(|&&r| r).count();
        let bh_rejections = bh.rejected.iter().filter(|&&r| r).count();
        assert!(bh_rejections >= bonf_rejections);
    }

    #[test]
    fn bh_adjusted_p_values_are_monotonic_with_sorted_input() {
        let p_values = vec![0.001, 0.01, 0.02, 0.03, 0.5];
        let result =
            correct_p_values(&p_values, 0.05, CorrectionMethod::BenjaminiHochberg).unwrap();
        // Input is already sorted ascending, so adjusted values must
        // also be non-decreasing.
        for window in result.adjusted_p_values.windows(2) {
            assert!(window[0] <= window[1] + 1e-12);
        }
    }

    #[test]
    fn preserves_original_order() {
        // Deliberately unsorted input — output must correspond to
        // the same positions as the input, not sorted order.
        let p_values = vec![0.5, 0.001, 0.3];
        let result =
            correct_p_values(&p_values, 0.05, CorrectionMethod::BenjaminiHochberg).unwrap();
        assert_eq!(result.adjusted_p_values.len(), 3);
        // The smallest input p-value (index 1) must still have the
        // smallest adjusted p-value.
        let min_idx = result
            .adjusted_p_values
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(min_idx, 1);
    }
}
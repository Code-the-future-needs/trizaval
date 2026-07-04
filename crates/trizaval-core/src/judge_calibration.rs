//! Calibration for known systematic biases in LLM-as-judge scoring.
//!
//! LLM judges are not neutral measuring instruments — they exhibit
//! well-documented biases, most notably:
//!
//! - **Position bias**: in pairwise A/B comparisons, judges favor
//!   whichever response appears in a particular position, regardless
//!   of quality.
//! - **Length bias**: judges tend to score longer responses higher,
//!   independent of actual quality.
//!
//! This module provides principled corrections for both, rather than
//! trusting raw judge output at face value.

/// A single pairwise preference outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preference {
    PrefersA,
    PrefersB,
    Tie,
}

/// Result of debiasing one pairwise comparison judged twice, with
/// response positions swapped between the two judgments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairwiseDebiasResult {
    /// The final, debiased preference. `Tie` includes both genuine
    /// ties and inconclusive cases where position bias could not be
    /// ruled out.
    pub preference: Preference,
    /// True if the judge's preference flipped along with the swap in
    /// position (i.e. it was tracking position, not content) rather
    /// than being consistent — a signal of position bias having been
    /// caught and neutralized.
    pub position_bias_detected: bool,
}

/// Debiases a single pairwise judgment by comparing two judgments of
/// the *same* pair of responses, made with their positions swapped.
///
/// `original_order`: preference when response A was shown first.
/// `swapped_order`: preference when response B was shown first (i.e.
/// positions swapped), expressed in terms of the *same* underlying
/// A/B identity as `original_order` (not "first/second").
///
/// If both judgments agree on the same underlying preference, that
/// preference is trusted. If they disagree, the disagreement itself
/// is evidence of position bias, so the result is downgraded to a
/// `Tie` rather than trusting either judgment.
pub fn debias_pairwise_judgment(
    original_order: Preference,
    swapped_order: Preference,
) -> PairwiseDebiasResult {
    if original_order == swapped_order {
        PairwiseDebiasResult {
            preference: original_order,
            position_bias_detected: false,
        }
    } else {
        PairwiseDebiasResult {
            preference: Preference::Tie,
            position_bias_detected: true,
        }
    }
}

/// Errors that can occur during length bias correction.
#[derive(Debug, Clone, PartialEq)]
pub enum JudgeCalibrationError {
    MismatchedLengths,
    InsufficientData,
    NoLengthVariance,
}

impl std::fmt::Display for JudgeCalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JudgeCalibrationError::MismatchedLengths => {
                write!(f, "scores and lengths must have the same number of elements")
            }
            JudgeCalibrationError::InsufficientData => {
                write!(f, "need at least 3 (score, length) pairs to fit a regression")
            }
            JudgeCalibrationError::NoLengthVariance => write!(
                f,
                "all lengths are identical — length effect cannot be estimated"
            ),
        }
    }
}

impl std::error::Error for JudgeCalibrationError {}

/// Result of fitting a length-bias correction.
#[derive(Debug, Clone, PartialEq)]
pub struct LengthBiasResult {
    /// Fitted slope: estimated score change per unit increase in
    /// length. Near zero means little to no length bias detected.
    pub slope: f64,
    pub intercept: f64,
    /// Pearson correlation between raw score and length, in [-1, 1].
    /// Large magnitude indicates strong length bias.
    pub correlation: f64,
    /// Length-adjusted scores: original score minus the portion
    /// explained by length alone, re-centered on the original mean
    /// score so the adjusted scores stay on a comparable scale.
    pub adjusted_scores: Vec<f64>,
}

/// Fits an ordinary least squares regression of `scores` on
/// `lengths` and returns length-adjusted (residualized) scores.
pub fn length_bias_correction(
    scores: &[f64],
    lengths: &[f64],
) -> Result<LengthBiasResult, JudgeCalibrationError> {
    if scores.len() != lengths.len() {
        return Err(JudgeCalibrationError::MismatchedLengths);
    }
    if scores.len() < 3 {
        return Err(JudgeCalibrationError::InsufficientData);
    }

    let n = scores.len() as f64;
    let mean_score = scores.iter().sum::<f64>() / n;
    let mean_length = lengths.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_length = 0.0;
    let mut var_score = 0.0;

    for i in 0..scores.len() {
        let ds = scores[i] - mean_score;
        let dl = lengths[i] - mean_length;
        cov += ds * dl;
        var_length += dl * dl;
        var_score += ds * ds;
    }

    if var_length <= 1e-12 {
        return Err(JudgeCalibrationError::NoLengthVariance);
    }

    let slope = cov / var_length;
    let intercept = mean_score - slope * mean_length;

    let correlation = if var_score <= 1e-12 {
        // Scores have no variance at all — length explains nothing
        // because there's nothing to explain.
        0.0
    } else {
        cov / (var_length.sqrt() * var_score.sqrt())
    };

    // Residual = actual score - predicted score from length alone,
    // then re-centered on the original mean so adjusted scores stay
    // on the same rough scale as the raw scores (a residual on its
    // own is centered at 0, which is harder to interpret alongside
    // e.g. a 1-10 judge scale).
    let adjusted_scores: Vec<f64> = scores
        .iter()
        .zip(lengths.iter())
        .map(|(&s, &l)| {
            let predicted = intercept + slope * l;
            let residual = s - predicted;
            residual + mean_score
        })
        .collect();

    Ok(LengthBiasResult {
        slope,
        intercept,
        correlation,
        adjusted_scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agreeing_judgments_are_trusted() {
        let result = debias_pairwise_judgment(Preference::PrefersA, Preference::PrefersA);
        assert_eq!(result.preference, Preference::PrefersA);
        assert!(!result.position_bias_detected);
    }

    #[test]
    fn disagreeing_judgments_become_a_tie() {
        let result = debias_pairwise_judgment(Preference::PrefersA, Preference::PrefersB);
        assert_eq!(result.preference, Preference::Tie);
        assert!(result.position_bias_detected);
    }

    #[test]
    fn agreeing_ties_stay_ties_without_flagging_bias() {
        let result = debias_pairwise_judgment(Preference::Tie, Preference::Tie);
        assert_eq!(result.preference, Preference::Tie);
        assert!(!result.position_bias_detected);
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let result = length_bias_correction(&[1.0, 2.0], &[1.0, 2.0, 3.0]);
        assert_eq!(result.unwrap_err(), JudgeCalibrationError::MismatchedLengths);
    }

    #[test]
    fn rejects_insufficient_data() {
        let result = length_bias_correction(&[1.0, 2.0], &[1.0, 2.0]);
        assert_eq!(result.unwrap_err(), JudgeCalibrationError::InsufficientData);
    }

    #[test]
    fn rejects_no_length_variance() {
        let result = length_bias_correction(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0]);
        assert_eq!(result.unwrap_err(), JudgeCalibrationError::NoLengthVariance);
    }

    #[test]
    fn detects_strong_length_bias() {
        // Score is a near-perfect linear function of length --
        // correlation should be close to 1.0, slope clearly positive.
        let lengths = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let scores = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = length_bias_correction(&scores, &lengths).unwrap();
        assert!(result.correlation > 0.99);
        assert!(result.slope > 0.0);
    }

    #[test]
    fn adjusted_scores_remove_length_trend() {
        let lengths = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let scores = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = length_bias_correction(&scores, &lengths).unwrap();
        // After perfectly removing a perfect linear trend, all
        // adjusted scores should collapse to (near) the same value.
        let first = result.adjusted_scores[0];
        for &s in &result.adjusted_scores {
            assert!((s - first).abs() < 1e-9);
        }
    }

    #[test]
    fn no_bias_when_length_and_score_are_uncorrelated() {
        // Lengths increase steadily but scores don't track them at
        // all -- correlation should be near zero.
        let lengths = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let scores = vec![7.0, 3.0, 8.0, 4.0, 6.0];
        let result = length_bias_correction(&scores, &lengths).unwrap();
        assert!(result.correlation.abs() < 0.5);
    }
}
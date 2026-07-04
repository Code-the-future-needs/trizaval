use extendr_api::prelude::*;
use trizaval_core::bootstrap;
use trizaval_core::correction::{self, CorrectionMethod};
use trizaval_core::effect_size;
use trizaval_core::judge_calibration::{self, Preference};
use trizaval_core::sequential::{SequentialDecision, SequentialTest};

/// Computes a block-bootstrap confidence interval for the mean of a
/// numeric vector.
///
/// @param data A numeric vector of evaluation scores.
/// @param block_size Length of contiguous blocks to resample. Use 1
///   for an ordinary independent bootstrap; use a larger value when
///   observations are correlated (e.g. consecutive similar prompts).
/// @param n_resamples Number of bootstrap resamples to perform.
/// @param confidence_level Confidence level, e.g. 0.95 for a 95% CI.
/// @param seed Optional integer seed for reproducible results. Pass
///   NULL for a non-reproducible, entropy-seeded run.
/// @return A named list with point_estimate, ci_lower, ci_upper,
///   confidence_level, and n_resamples.
/// @export
#[extendr]
fn block_bootstrap_mean(
    data: Vec<f64>,
    block_size: i32,
    n_resamples: i32,
    confidence_level: f64,
    seed: Nullable<i32>,
) -> extendr_api::Result<List> {
    let seed_u64 = match seed {
        Nullable::NotNull(s) => Some(s as u64),
        Nullable::Null => None,
    };

    let result = bootstrap::block_bootstrap(
        &data,
        block_size as usize,
        n_resamples as usize,
        confidence_level,
        bootstrap::mean,
        seed_u64,
    )
    .map_err(|e| Error::Other(e.to_string()))?;

    Ok(list!(
        point_estimate = result.point_estimate,
        ci_lower = result.ci_lower,
        ci_upper = result.ci_upper,
        confidence_level = result.confidence_level,
        n_resamples = result.n_resamples as i32
    ))
}

/// Computes Cohen's d and Hedges' g effect size between two groups
/// of scores.
///
/// @param baseline A numeric vector of baseline scores.
/// @param treatment A numeric vector of treatment/candidate scores.
/// @return A named list with cohens_d, hedges_g, magnitude,
///   n_baseline, and n_treatment.
/// @export
#[extendr]
fn cohens_d(baseline: Vec<f64>, treatment: Vec<f64>) -> extendr_api::Result<List> {
    let result = effect_size::cohens_d(&baseline, &treatment).map_err(|e| Error::Other(e.to_string()))?;

    Ok(list!(
        cohens_d = result.cohens_d,
        hedges_g = result.hedges_g,
        magnitude = format!("{:?}", result.magnitude),
        n_baseline = result.n_baseline as i32,
        n_treatment = result.n_treatment as i32
    ))
}

/// A stateful sequential hypothesis test (mixture SPRT). Feed it
/// observations one at a time via `$update(x)`; stop collecting data
/// as soon as `$update()` returns `rejected = TRUE`.
///
/// @export
#[extendr]
struct RSequentialTest {
    inner: SequentialTest,
}

#[extendr]
impl RSequentialTest {
    /// Constructs a new sequential test.
    /// @param alpha Desired Type I error rate, e.g. 0.05.
    /// @param tau Prior standard deviation on the effect size worth
    ///   detecting.
    fn new(alpha: f64, tau: f64) -> extendr_api::Result<Self> {
        let inner = SequentialTest::new(alpha, tau).map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Feeds one new observation into the test.
    /// @param x The new observation (e.g. candidate_score - baseline_score).
    /// @return A named list with n, likelihood_ratio, and rejected.
    fn update(&mut self, x: f64) -> List {
        let update = self.inner.update(x);
        list!(
            n = update.n as i32,
            likelihood_ratio = update.likelihood_ratio,
            rejected = update.decision == SequentialDecision::RejectNull
        )
    }

    /// Current running mean of observations fed so far.
    fn current_mean(&self) -> f64 {
        self.inner.current_mean()
    }

    /// Number of observations fed so far.
    fn n(&self) -> i32 {
        self.inner.n() as i32
    }
}

/// Applies a multiple-comparisons correction to a vector of p-values.
///
/// @param p_values A numeric vector of p-values.
/// @param alpha Significance threshold, e.g. 0.05.
/// @param method Either "bonferroni" or "benjamini_hochberg".
/// @return A named list with adjusted_p_values and rejected (a
///   logical vector).
/// @export
#[extendr]
fn correct_p_values(p_values: Vec<f64>, alpha: f64, method: &str) -> extendr_api::Result<List> {
    let parsed_method = match method {
        "bonferroni" => CorrectionMethod::Bonferroni,
        "benjamini_hochberg" => CorrectionMethod::BenjaminiHochberg,
        other => {
            return Err(Error::Other(format!(
                "unknown method '{other}', expected 'bonferroni' or 'benjamini_hochberg'"
            )))
        }
    };

    let result = correction::correct_p_values(&p_values, alpha, parsed_method)
        .map_err(|e| Error::Other(e.to_string()))?;

    Ok(list!(
        adjusted_p_values = result.adjusted_p_values,
        rejected = result.rejected
    ))
}

/// Debiases a single pairwise judge comparison judged twice with
/// positions swapped, to correct for position bias.
///
/// @param original_order Preference when response A was shown first:
///   one of "prefers_a", "prefers_b", "tie".
/// @param swapped_order Preference when response B was shown first
///   (expressed in terms of the same underlying A/B identity).
/// @return A named list with preference and position_bias_detected.
/// @export
#[extendr]
fn debias_pairwise_judgment(original_order: &str, swapped_order: &str) -> extendr_api::Result<List> {
    fn parse_preference(s: &str) -> extendr_api::Result<Preference> {
        match s {
            "prefers_a" => Ok(Preference::PrefersA),
            "prefers_b" => Ok(Preference::PrefersB),
            "tie" => Ok(Preference::Tie),
            other => Err(Error::Other(format!(
                "unknown preference '{other}', expected 'prefers_a', 'prefers_b', or 'tie'"
            ))),
        }
    }

    let original = parse_preference(original_order)?;
    let swapped = parse_preference(swapped_order)?;

    let result = judge_calibration::debias_pairwise_judgment(original, swapped);

    let preference_str = match result.preference {
        Preference::PrefersA => "prefers_a",
        Preference::PrefersB => "prefers_b",
        Preference::Tie => "tie",
    };

    Ok(list!(
        preference = preference_str,
        position_bias_detected = result.position_bias_detected
    ))
}

/// Fits and removes length bias from a batch of LLM-judge scores.
///
/// @param scores A numeric vector of judge scores.
/// @param lengths A numeric vector of response lengths, same length
///   and order as `scores`.
/// @return A named list with slope, intercept, correlation, and
///   adjusted_scores.
/// @export
#[extendr]
fn length_bias_correction(scores: Vec<f64>, lengths: Vec<f64>) -> extendr_api::Result<List> {
    let result =
        judge_calibration::length_bias_correction(&scores, &lengths).map_err(|e| Error::Other(e.to_string()))?;

    Ok(list!(
        slope = result.slope,
        intercept = result.intercept,
        correlation = result.correlation,
        adjusted_scores = result.adjusted_scores
    ))
}

// Macro to generate exports.
// This ensures exported functions are registered with R.
// See corresponding C code in `entrypoint.c`.
extendr_module! {
    mod trizavalr;
    fn block_bootstrap_mean;
    fn cohens_d;
    fn correct_p_values;
    fn debias_pairwise_judgment;
    fn length_bias_correction;
    impl RSequentialTest;
}
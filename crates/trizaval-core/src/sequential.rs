//! Sequential hypothesis testing via mixture SPRT (Robbins' mixture
//! sequential probability ratio test).
//!
//! Unlike a fixed-sample test, this stays statistically valid no
//! matter when you stop observing data - you can check the result
//! after every single new observation without inflating the false
//! positive rate, as long as you stop the moment `RejectNull` is
//! returned rather than continuing to peek afterward.
//!
//! Tests H0: true mean effect = 0, using a Gaussian mixture prior
//! with variance `tau^2` over the alternative effect size. `tau`
//! is a tuning parameter representing "the effect size we care
//! about detecting" — smaller `tau` gives more power to detect
//! small effects but less power for large ones, and vice versa.
//!
//!  this implementation uses an online plug-in estimate of the
//! observation variance (via Welford's algorithm) rather than a
//! known fixed variance. This is standard practice but means the
//! validity guarantee is asymptotic (accurate once n is reasonably
//! large), not exact for the very first few observations.

/// Errors that can occur when constructing or using a sequential test.
#[derive(Debug, Clone, PartialEq)]
pub enum SequentialError {
    InvalidAlpha,
    InvalidTau,
}

impl std::fmt::Display for SequentialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SequentialError::InvalidAlpha => write!(f, "alpha must be in (0.0, 1.0)"),
            SequentialError::InvalidTau => write!(f, "tau must be > 0.0"),
        }
    }
}

impl std::error::Error for SequentialError {}

/// Decision after incorporating the latest observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SequentialDecision {
    /// Not enough evidence yet — keep collecting data.
    Continue,
    /// Evidence crossed the threshold: reject H0 (no effect) at
    /// the configured `alpha` level. Stop collecting data now.
    RejectNull,
}

/// Result of processing one new observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SequentialUpdate {
    pub n: usize,
    /// The mixture likelihood ratio. Reject when this crosses 1/alpha.
    pub likelihood_ratio: f64,
    pub decision: SequentialDecision,
}

/// A stateful sequential test — feed it observations one at a time
/// via `update`, and check the returned decision after each one.
#[derive(Debug)]
pub struct SequentialTest {
    alpha: f64,
    tau: f64,
    n: usize,
    mean: f64,
    m2: f64, 
}

impl SequentialTest {
    /// `alpha`: desired Type I error rate, e.g. 0.05.
    /// `tau`: prior standard deviation on the effect size under the
    /// alternative hypothesis — represents "the effect size worth
    /// detecting" for this test.
    pub fn new(alpha: f64, tau: f64) -> Result<Self, SequentialError> {
        if !(0.0 < alpha && alpha < 1.0) {
            return Err(SequentialError::InvalidAlpha);
        }
        if !(tau > 0.0) {
            return Err(SequentialError::InvalidTau);
        }
        Ok(Self {
            alpha,
            tau,
            n: 0,
            mean: 0.0,
            m2: 0.0,
        })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn current_mean(&self) -> f64 {
        self.mean
    }

    /// Unbiased sample variance estimate. `None` until at least 2
    /// observations have been seen.
    pub fn variance_estimate(&self) -> Option<f64> {
        if self.n < 2 {
            None
        } else {
            Some(self.m2 / (self.n as f64 - 1.0))
        }
    }

    /// Feed one new observation into the test and get back the
    /// updated decision.
    pub fn update(&mut self, x: f64) -> SequentialUpdate {
        // Welford's online mean/variance update.
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        let sigma2 = match self.variance_estimate() {
            Some(v) if v > 1e-12 => v,
            // Not enough data, or variance too small to estimate
            // reliably yet — can't safely test, keep collecting.
            _ => {
                return SequentialUpdate {
                    n: self.n,
                    likelihood_ratio: 1.0,
                    decision: SequentialDecision::Continue,
                }
            }
        };

        let n = self.n as f64;
        let tau2 = self.tau * self.tau;

        // Robbins' mixture SPRT statistic for a Gaussian mixture
        // prior N(0, tau^2) on the effect size:
        //   Λ_n = sqrt(σ²/(σ²+nτ²)) * exp( n²τ² x̄² / (2σ²(σ²+nτ²)) )
        let denom = sigma2 + n * tau2;
        let sqrt_term = (sigma2 / denom).sqrt();
        let exponent = (n * n * tau2 * self.mean * self.mean) / (2.0 * sigma2 * denom);
        let likelihood_ratio = sqrt_term * exponent.exp();

        let decision = if likelihood_ratio >= 1.0 / self.alpha {
            SequentialDecision::RejectNull
        } else {
            SequentialDecision::Continue
        };

        SequentialUpdate {
            n: self.n,
            likelihood_ratio,
            decision,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

#[test]
    fn rejects_invalid_alpha() {
        assert_eq!(SequentialTest::new(0.0, 1.0).unwrap_err(), SequentialError::InvalidAlpha);
        assert_eq!(SequentialTest::new(1.0, 1.0).unwrap_err(), SequentialError::InvalidAlpha);
    }

    #[test]
    fn rejects_invalid_tau() {
        assert_eq!(SequentialTest::new(0.05, 0.0).unwrap_err(), SequentialError::InvalidTau);
        assert_eq!(SequentialTest::new(0.05, -1.0).unwrap_err(), SequentialError::InvalidTau);
    }

    #[test]
    fn continues_with_insufficient_data() {
        let mut test = SequentialTest::new(0.05, 1.0).unwrap();
        let update = test.update(5.0); // only 1 observation, no variance yet
        assert_eq!(update.decision, SequentialDecision::Continue);
    }

    #[test]
    fn detects_a_real_effect_eventually() {
        // Simulated eval scores with a clear true effect (mean ~1.0,
        // small noise) — a deterministic pseudo-noise pattern so the
        // test is reproducible without needing an RNG dependency here.
        let mut test = SequentialTest::new(0.05, 0.5).unwrap();
        let noise = [0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.04, 0.03, -0.05];
        let mut rejected = false;
        for i in 0..500 {
            let x = 1.0 + noise[i % noise.len()];
            let update = test.update(x);
            if update.decision == SequentialDecision::RejectNull {
                rejected = true;
                break;
            }
        }
        assert!(rejected, "expected a clear true effect to eventually be detected");
    }

    #[test]
    fn does_not_reject_under_a_true_null() {
        // Data genuinely centered at 0 (alternating +/-1) should
        // not trigger a rejection.
        let mut test = SequentialTest::new(0.05, 0.5).unwrap();
        let mut rejected = false;
        for i in 0..500 {
            let x = if i % 2 == 0 { 1.0 } else { -1.0 };
            let update = test.update(x);
            if update.decision == SequentialDecision::RejectNull {
                rejected = true;
                break;
            }
        }
        assert!(!rejected, "expected no false rejection under a true null effect");
    }

    #[test]
    fn n_and_mean_tracked_correctly() {
        let mut test = SequentialTest::new(0.05, 1.0).unwrap();
        test.update(2.0);
        test.update(4.0);
        test.update(6.0);
        assert_eq!(test.n(), 3);
        assert!((test.current_mean() - 4.0).abs() < 1e-9);
    }
}
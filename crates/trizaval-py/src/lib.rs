//! PyO3 bindings exposing trizaval-core's statistical primitives to
//! Python. This crate is a thin translation layer only  all actual
//! statistical logic lives in trizaval-core.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use pyo3::Bound;

use trizaval_core::bootstrap;
use trizaval_core::correction::{self, CorrectionMethod};
use trizaval_core::effect_size::{self, EffectMagnitude};
use trizaval_core::judge_calibration::{self, Preference};
use trizaval_core::sequential::{SequentialDecision, SequentialTest};

/// Python-visible result of a block bootstrap confidence interval
/// estimation.
#[pyclass(name = "BootstrapResult")]
#[derive(Clone)]
struct PyBootstrapResult {
    #[pyo3(get)]
    point_estimate: f64,
    #[pyo3(get)]
    ci_lower: f64,
    #[pyo3(get)]
    ci_upper: f64,
    #[pyo3(get)]
    confidence_level: f64,
    #[pyo3(get)]
    n_resamples: usize,
}

#[pymethods]
impl PyBootstrapResult {
    fn __repr__(&self) -> String {
        format!(
            "BootstrapResult(point_estimate={:.6}, ci_lower={:.6}, ci_upper={:.6}, confidence_level={}, n_resamples={})",
            self.point_estimate, self.ci_lower, self.ci_upper, self.confidence_level, self.n_resamples
        )
    }
}

fn to_py_result(result: Result<bootstrap::BootstrapResult, bootstrap::BootstrapError>) -> PyResult<PyBootstrapResult> {
    let result = result.map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyBootstrapResult {
        point_estimate: result.point_estimate,
        ci_lower: result.ci_lower,
        ci_upper: result.ci_upper,
        confidence_level: result.confidence_level,
        n_resamples: result.n_resamples,
    })
}

/// Computes a block-bootstrap confidence interval using an arbitrary
/// Python callable as the statistic.
///
/// `statistic` is called once per resample as `statistic(list_of_floats)`
/// and must return a single float. Exceptions raised inside
/// `statistic`, or non-numeric return values, are caught and reported
/// as a Python `ValueError` rather than crashing the process.
///
/// Note on performance: this calls back into Python once per
/// resample, which carries real overhead (GIL round-trip + list
/// construction) compared to `block_bootstrap_mean`. For the common
/// case of bootstrapping a mean, prefer `block_bootstrap_mean`, which
/// never leaves Rust.
#[pyfunction]
#[pyo3(signature = (data, block_size, n_resamples, confidence_level, statistic, seed=None))]
fn block_bootstrap(
    py: Python<'_>,
    data: Vec<f64>,
    block_size: usize,
    n_resamples: usize,
    confidence_level: f64,
    statistic: Py<PyAny>,
    seed: Option<u64>,
) -> PyResult<PyBootstrapResult> {
    // Released implicitly at the end of `py.allow_threads` scope if
    // we ever parallelize this; for now we run on this thread with
    // the GIL held throughout, since we call back into Python on
    // every resample anyway.
    let py_stat = move |arr: &[f64]| -> Result<f64, String> {
        Python::with_gil(|py| {
            let list = PyList::new_bound(py, arr);
            let call_result = statistic.call1(py, (list,));
            match call_result {
                Ok(value) => value
                    .bind(py)
                    .extract::<f64>()
                    .map_err(|e| format!("statistic callable returned a non-float value: {e}")),
                Err(e) => Err(format!("statistic callable raised an exception: {e}")),
            }
        })
    };

    let _ = py; // GIL token for this outer call is already implicitly held throughout.
    let result = bootstrap::block_bootstrap(&data, block_size, n_resamples, confidence_level, py_stat, seed);
    to_py_result(result)
}

/// Computes a block-bootstrap confidence interval for the mean of
/// `data`. Fast path: runs entirely in Rust, no Python callback
/// overhead. Prefer this over `block_bootstrap` whenever the
/// statistic of interest is a plain mean (the common case: mean
/// accuracy, mean judge score).
#[pyfunction]
#[pyo3(signature = (data, block_size, n_resamples, confidence_level, seed=None))]
fn block_bootstrap_mean(
    data: Vec<f64>,
    block_size: usize,
    n_resamples: usize,
    confidence_level: f64,
    seed: Option<u64>,
) -> PyResult<PyBootstrapResult> {
    let result = bootstrap::block_bootstrap(
        &data,
        block_size,
        n_resamples,
        confidence_level,
        bootstrap::mean,
        seed,
    );
    to_py_result(result)
}

/// Decision returned after each observation fed into a
/// `SequentialTest`.
#[pyclass(name = "SequentialDecision", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum PySequentialDecision {
    Continue,
    RejectNull,
}

impl From<SequentialDecision> for PySequentialDecision {
    fn from(d: SequentialDecision) -> Self {
        match d {
            SequentialDecision::Continue => PySequentialDecision::Continue,
            SequentialDecision::RejectNull => PySequentialDecision::RejectNull,
        }
    }
}

/// Result of feeding one new observation into a `SequentialTest`.
#[pyclass(name = "SequentialUpdate")]
#[derive(Clone)]
struct PySequentialUpdate {
    #[pyo3(get)]
    n: usize,
    #[pyo3(get)]
    likelihood_ratio: f64,
    #[pyo3(get)]
    decision: PySequentialDecision,
}

#[pymethods]
impl PySequentialUpdate {
    fn __repr__(&self) -> String {
        format!(
            "SequentialUpdate(n={}, likelihood_ratio={:.6}, decision={:?})",
            self.n,
            self.likelihood_ratio,
            match self.decision {
                PySequentialDecision::Continue => "Continue",
                PySequentialDecision::RejectNull => "RejectNull",
            }
        )
    }
}

/// A stateful sequential hypothesis test (mixture SPRT). Feed it
/// observations one at a time via `.update(x)`; stop collecting data
/// as soon as a returned `SequentialUpdate.decision` is
/// `SequentialDecision.RejectNull`.
///
/// `alpha`: desired Type I error rate, e.g. 0.05.
/// `tau`: prior standard deviation on the effect size under the
/// alternative hypothesis — represents "the effect size worth
/// detecting".
#[pyclass(name = "SequentialTest")]
struct PySequentialTest {
    inner: SequentialTest,
}

#[pymethods]
impl PySequentialTest {
    #[new]
    fn new(alpha: f64, tau: f64) -> PyResult<Self> {
        let inner = SequentialTest::new(alpha, tau).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Feed one new observation into the test and get back the
    /// updated decision.
    fn update(&mut self, x: f64) -> PySequentialUpdate {
        let update = self.inner.update(x);
        PySequentialUpdate {
            n: update.n,
            likelihood_ratio: update.likelihood_ratio,
            decision: update.decision.into(),
        }
    }

    #[getter]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    fn current_mean(&self) -> f64 {
        self.inner.current_mean()
    }

    #[getter]
    fn variance_estimate(&self) -> Option<f64> {
        self.inner.variance_estimate()
    }
}



/// Which multiple-comparisons correction method to apply.
#[pyclass(name = "CorrectionMethod", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum PyCorrectionMethod {
    Bonferroni,
    BenjaminiHochberg,
}

impl From<PyCorrectionMethod> for CorrectionMethod {
    fn from(m: PyCorrectionMethod) -> Self {
        match m {
            PyCorrectionMethod::Bonferroni => CorrectionMethod::Bonferroni,
            PyCorrectionMethod::BenjaminiHochberg => CorrectionMethod::BenjaminiHochberg,
        }
    }
}

/// Result of applying a multiple-comparisons correction to a set of
/// p-values.
#[pyclass(name = "CorrectionResult")]
#[derive(Clone)]
struct PyCorrectionResult {
    #[pyo3(get)]
    adjusted_p_values: Vec<f64>,
    #[pyo3(get)]
    rejected: Vec<bool>,
    #[pyo3(get)]
    alpha: f64,
}

#[pymethods]
impl PyCorrectionResult {
    fn __repr__(&self) -> String {
        format!(
            "CorrectionResult(adjusted_p_values={:?}, rejected={:?}, alpha={})",
            self.adjusted_p_values, self.rejected, self.alpha
        )
    }
}

/// Applies a multiple-comparisons correction to `p_values`.
///
/// Use `CorrectionMethod.Bonferroni` for strict family-wise error
/// control, or `CorrectionMethod.BenjaminiHochberg` for higher-power
/// false-discovery-rate control when checking many metrics at once
/// (the more realistic case for eval suites).
#[pyfunction]
#[pyo3(signature = (p_values, alpha, method))]
fn correct_p_values(
    p_values: Vec<f64>,
    alpha: f64,
    method: PyCorrectionMethod,
) -> PyResult<PyCorrectionResult> {
    let result = correction::correct_p_values(&p_values, alpha, method.into())
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyCorrectionResult {
        adjusted_p_values: result.adjusted_p_values,
        rejected: result.rejected,
        alpha: result.alpha,
    })
}

/// Qualitative interpretation bucket for a standardized effect size.
#[pyclass(name = "EffectMagnitude", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum PyEffectMagnitude {
    Negligible,
    Small,
    Medium,
    Large,
}

impl From<EffectMagnitude> for PyEffectMagnitude {
    fn from(m: EffectMagnitude) -> Self {
        match m {
            EffectMagnitude::Negligible => PyEffectMagnitude::Negligible,
            EffectMagnitude::Small => PyEffectMagnitude::Small,
            EffectMagnitude::Medium => PyEffectMagnitude::Medium,
            EffectMagnitude::Large => PyEffectMagnitude::Large,
        }
    }
}

/// Result of a Cohen's d / Hedges' g effect size computation.
#[pyclass(name = "EffectSizeResult")]
#[derive(Clone)]
struct PyEffectSizeResult {
    #[pyo3(get)]
    cohens_d: f64,
    #[pyo3(get)]
    hedges_g: f64,
    #[pyo3(get)]
    magnitude: PyEffectMagnitude,
    #[pyo3(get)]
    n_baseline: usize,
    #[pyo3(get)]
    n_treatment: usize,
}

#[pymethods]
impl PyEffectSizeResult {
    fn __repr__(&self) -> String {
        format!(
            "EffectSizeResult(cohens_d={:.6}, hedges_g={:.6}, n_baseline={}, n_treatment={})",
            self.cohens_d, self.hedges_g, self.n_baseline, self.n_treatment
        )
    }
}

/// Computes Cohen's d and Hedges' g for the difference between
/// `treatment` and `baseline` sample means, using the pooled
/// standard deviation. Positive values mean `treatment` scored
/// higher than `baseline`; negative means lower.
///
/// Prefer `hedges_g` over `cohens_d` when group sizes are small
/// (roughly n < 20 per group) — Cohen's d systematically
/// overestimates effect size in that regime.
#[pyfunction]
#[pyo3(signature = (baseline, treatment))]
fn cohens_d(baseline: Vec<f64>, treatment: Vec<f64>) -> PyResult<PyEffectSizeResult> {
    let result = effect_size::cohens_d(&baseline, &treatment)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyEffectSizeResult {
        cohens_d: result.cohens_d,
        hedges_g: result.hedges_g,
        magnitude: result.magnitude.into(),
        n_baseline: result.n_baseline,
        n_treatment: result.n_treatment,
    })
}

/// A pairwise preference outcome between response A and response B.
#[pyclass(name = "Preference", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum PyPreference {
    PrefersA,
    PrefersB,
    Tie,
}

impl From<Preference> for PyPreference {
    fn from(p: Preference) -> Self {
        match p {
            Preference::PrefersA => PyPreference::PrefersA,
            Preference::PrefersB => PyPreference::PrefersB,
            Preference::Tie => PyPreference::Tie,
        }
    }
}

impl From<PyPreference> for Preference {
    fn from(p: PyPreference) -> Self {
        match p {
            PyPreference::PrefersA => Preference::PrefersA,
            PyPreference::PrefersB => Preference::PrefersB,
            PyPreference::Tie => Preference::Tie,
        }
    }
}

/// Result of debiasing one pairwise comparison judged twice, with
/// response positions swapped between the two judgments.
#[pyclass(name = "PairwiseDebiasResult")]
#[derive(Clone)]
struct PyPairwiseDebiasResult {
    #[pyo3(get)]
    preference: PyPreference,
    #[pyo3(get)]
    position_bias_detected: bool,
}

#[pymethods]
impl PyPairwiseDebiasResult {
    fn __repr__(&self) -> String {
        format!(
            "PairwiseDebiasResult(preference={:?}, position_bias_detected={})",
            match self.preference {
                PyPreference::PrefersA => "PrefersA",
                PyPreference::PrefersB => "PrefersB",
                PyPreference::Tie => "Tie",
            },
            self.position_bias_detected
        )
    }
}

/// Debiases a single pairwise judgment by comparing two judgments of
/// the same pair of responses, made with positions swapped. If the
/// two judgments disagree, that disagreement is itself evidence of
/// position bias, and the result is downgraded to a Tie rather than
/// trusting either judgment.
#[pyfunction]
#[pyo3(signature = (original_order, swapped_order))]
fn debias_pairwise_judgment(
    original_order: PyPreference,
    swapped_order: PyPreference,
) -> PyPairwiseDebiasResult {
    let result = judge_calibration::debias_pairwise_judgment(
        original_order.into(),
        swapped_order.into(),
    );
    PyPairwiseDebiasResult {
        preference: result.preference.into(),
        position_bias_detected: result.position_bias_detected,
    }
}

/// Result of fitting a length-bias correction to judge scores.
#[pyclass(name = "LengthBiasResult")]
#[derive(Clone)]
struct PyLengthBiasResult {
    #[pyo3(get)]
    slope: f64,
    #[pyo3(get)]
    intercept: f64,
    #[pyo3(get)]
    correlation: f64,
    #[pyo3(get)]
    adjusted_scores: Vec<f64>,
}

#[pymethods]
impl PyLengthBiasResult {
    fn __repr__(&self) -> String {
        format!(
            "LengthBiasResult(slope={:.6}, intercept={:.6}, correlation={:.6}, adjusted_scores={:?})",
            self.slope, self.intercept, self.correlation, self.adjusted_scores
        )
    }
}

/// Fits an ordinary least squares regression of `scores` on
/// `lengths` and returns length-adjusted (residualized) scores,
/// correcting for the tendency of LLM judges to score longer
/// responses higher independent of actual quality.
#[pyfunction]
#[pyo3(signature = (scores, lengths))]
fn length_bias_correction(scores: Vec<f64>, lengths: Vec<f64>) -> PyResult<PyLengthBiasResult> {
    let result = judge_calibration::length_bias_correction(&scores, &lengths)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyLengthBiasResult {
        slope: result.slope,
        intercept: result.intercept,
        correlation: result.correlation,
        adjusted_scores: result.adjusted_scores,
    })
}

#[pymodule]
fn _trizaval_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBootstrapResult>()?;
    m.add_function(wrap_pyfunction!(block_bootstrap, m)?)?;
    m.add_function(wrap_pyfunction!(block_bootstrap_mean, m)?)?;
    m.add_class::<PySequentialDecision>()?;
    m.add_class::<PySequentialUpdate>()?;
    m.add_class::<PySequentialTest>()?;
    m.add_class::<PyCorrectionMethod>()?;
    m.add_class::<PyCorrectionResult>()?;
    m.add_function(wrap_pyfunction!(correct_p_values, m)?)?;
    m.add_class::<PyEffectMagnitude>()?;
    m.add_class::<PyEffectSizeResult>()?;
    m.add_function(wrap_pyfunction!(cohens_d, m)?)?;
    m.add_class::<PyPreference>()?;
    m.add_class::<PyPairwiseDebiasResult>()?;
    m.add_function(wrap_pyfunction!(debias_pairwise_judgment, m)?)?;
    m.add_class::<PyLengthBiasResult>()?;
    m.add_function(wrap_pyfunction!(length_bias_correction, m)?)?;
    Ok(())
}
//! WASM bindings exposing trizaval-core's statistical primitives to
//! JavaScript, for the browser-based dashboard. Compiled via
//! wasm-pack. Currently wraps bootstrap, effect size, and sequential
//! testing -- the three methods the dashboard's charts need
//! (confidence interval bars, effect size comparisons, and
//! sequential test trajectories). Judge calibration (position/length
//! bias) is not yet exposed here, since no dashboard chart currently
//! visualizes it.

use serde::Serialize;
use trizaval_core::bootstrap;
use trizaval_core::effect_size;
use trizaval_core::sequential::{SequentialDecision, SequentialTest};
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
pub struct WasmBootstrapResult {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub confidence_level: f64,
    pub n_resamples: usize,
}

/// Computes a block-bootstrap confidence interval for the mean of
/// `data`. Mirrors `block_bootstrap_mean` from the Python bindings,
/// same underlying Rust implementation.
#[wasm_bindgen]
pub fn block_bootstrap_mean(
    data: Vec<f64>,
    block_size: usize,
    n_resamples: usize,
    confidence_level: f64,
    seed: Option<u32>,
) -> Result<JsValue, JsError> {
    let result = bootstrap::block_bootstrap(
        &data,
        block_size,
        n_resamples,
        confidence_level,
        bootstrap::mean,
        seed.map(|s| s as u64),
    )
    .map_err(|e| JsError::new(&e.to_string()))?;

    let wasm_result = WasmBootstrapResult {
        point_estimate: result.point_estimate,
        ci_lower: result.ci_lower,
        ci_upper: result.ci_upper,
        confidence_level: result.confidence_level,
        n_resamples: result.n_resamples,
    };

    serde_wasm_bindgen::to_value(&wasm_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
pub struct WasmEffectSizeResult {
    pub cohens_d: f64,
    pub hedges_g: f64,
    pub magnitude: String,
    pub n_baseline: usize,
    pub n_treatment: usize,
}

/// Computes Cohen's d / Hedges' g between two groups of scores.
/// Mirrors `cohens_d` from the Python bindings.
#[wasm_bindgen]
pub fn cohens_d(baseline: Vec<f64>, treatment: Vec<f64>) -> Result<JsValue, JsError> {
    let result = effect_size::cohens_d(&baseline, &treatment).map_err(|e| JsError::new(&e.to_string()))?;

    let wasm_result = WasmEffectSizeResult {
        cohens_d: result.cohens_d,
        hedges_g: result.hedges_g,
        magnitude: format!("{:?}", result.magnitude),
        n_baseline: result.n_baseline,
        n_treatment: result.n_treatment,
    };

    serde_wasm_bindgen::to_value(&wasm_result).map_err(|e| JsError::new(&e.to_string()))
}

#[derive(Serialize)]
pub struct WasmSequentialUpdate {
    pub n: usize,
    pub likelihood_ratio: f64,
    pub rejected: bool,
}

/// Stateful sequential test wrapper for JS. Construct once per
/// trajectory, then call `update(x)` once per paired observation, in
/// order. Mirrors `SequentialTest` from the Python bindings.
#[wasm_bindgen]
pub struct WasmSequentialTest {
    inner: SequentialTest,
}

#[wasm_bindgen]
impl WasmSequentialTest {
    #[wasm_bindgen(constructor)]
    pub fn new(alpha: f64, tau: f64) -> Result<WasmSequentialTest, JsError> {
        let inner = SequentialTest::new(alpha, tau).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmSequentialTest { inner })
    }

    pub fn update(&mut self, x: f64) -> Result<JsValue, JsError> {
        let update = self.inner.update(x);
        let wasm_update = WasmSequentialUpdate {
            n: update.n,
            likelihood_ratio: update.likelihood_ratio,
            rejected: update.decision == SequentialDecision::RejectNull,
        };
        serde_wasm_bindgen::to_value(&wasm_update).map_err(|e| JsError::new(&e.to_string()))
    }
}
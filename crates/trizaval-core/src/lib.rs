//! trizaval-core: statistical primitives for rigorous evaluation of
//! non-deterministic AI systems.
//!
//! This crate is intentionally dependency-light and has zero Python
//! or I/O concerns — it is pure statistics, safe to embed anywhere
//! (native, WASM, or via FFI bindings).

pub mod bootstrap;
pub mod correction;
pub mod effect_size;
pub mod judge_calibration;
pub mod sequential;
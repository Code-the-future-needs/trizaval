# trizavalr

R bindings for Trizaval's statistical engine, the same Rust core used by Trizaval's Python package and command line tool.

## What this provides

All five of `trizaval-core`'s statistical methods, backed by the real Rust crate (not a reimplementation):

- `block_bootstrap_mean()`: block bootstrap confidence interval for the mean of a numeric vector, correctly accounting for correlation between nearby observations.
- `cohens_d()`: Cohen's d and Hedges' g effect size between two groups.
- `RSequentialTest`: a stateful mixture-SPRT sequential hypothesis test, constructed with `RSequentialTest$new(alpha, tau)` and fed observations one at a time via `$update(x)`, allowing early stopping once significance is reached.
- `correct_p_values()`: Bonferroni or Benjamini-Hochberg multiple-comparisons correction across a vector of p-values.
- `debias_pairwise_judgment()`: corrects for LLM-judge position bias by reconciling two judgments of the same pair made with positions swapped.
- `length_bias_correction()`: fits and removes the portion of LLM-judge scores explained by response length alone.

Results are numerically identical to what the Python bindings, Rust CLI, and WebAssembly dashboard produce, since all of these call the same underlying Rust implementation.

## Installation

This package is not yet on CRAN. Install from source:

```r
# From the repository root
devtools::install("bindings/r/trizavalr")
```

Requires a Rust toolchain (`cargo`, `rustc >= 1.65.0`) available on your system, since the package compiles a Rust crate as part of installation.

## Usage

```r
library(trizavalr)

result <- block_bootstrap_mean(
  data = c(0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.92, 0.7),
  block_size = 2L,
  n_resamples = 2000L,
  confidence_level = 0.95,
  seed = 42L
)
print(result)

effect <- cohens_d(
  baseline = c(0.5, 0.6, 0.55, 0.52, 0.58),
  treatment = c(0.9, 0.92, 0.88, 0.91, 0.89)
)
print(effect)

# Sequential testing: feed observations one at a time, stop early
# once significance is reached
test <- RSequentialTest$new(alpha = 0.05, tau = 0.3)
for (x in c(1.05, 0.97, 1.02, 0.99, 1.04)) {
  update <- test$update(x)
  if (update$rejected) {
    cat("Rejected null at n =", update$n, "\n")
    break
  }
}

# Multiple comparisons correction across several metrics at once
p_values <- c(0.001, 0.004, 0.008, 0.02, 0.6)
correction <- correct_p_values(p_values, alpha = 0.05, method = "benjamini_hochberg")
print(correction)

# LLM judge position-bias debiasing: judge the same pair twice with
# positions swapped, then reconcile
debiased <- debias_pairwise_judgment(original_order = "prefers_a", swapped_order = "prefers_b")
print(debiased)

# LLM judge length-bias correction
lengths <- c(10.0, 20.0, 30.0, 40.0, 50.0)
scores <- c(6.0, 6.5, 7.0, 8.5, 9.0)
length_bias <- length_bias_correction(scores, lengths)
print(length_bias)
```

## Development

```r
devtools::document()  # regenerate docs and recompile the Rust crate
devtools::test()      # run the testthat suite
```
## Scope

This binding covers all five of `trizaval-core`'s statistical modules: bootstrap, effect size, sequential testing, multiple-comparisons correction, and judge calibration. It does not include the eval-suite harness, provider adapters, or storage layer that the Python package provides; this package is intentionally scoped to the statistical core only.
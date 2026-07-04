test_that("block_bootstrap_mean computes a sensible confidence interval", {
  result <- block_bootstrap_mean(
    data = c(0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.92, 0.7),
    block_size = 2L,
    n_resamples = 2000L,
    confidence_level = 0.95,
    seed = 42L
  )

  expect_true(result$ci_lower <= result$point_estimate)
  expect_true(result$point_estimate <= result$ci_upper)
  expect_equal(result$confidence_level, 0.95)
  expect_equal(result$n_resamples, 2000L)
})

test_that("block_bootstrap_mean is reproducible with the same seed", {
  data <- c(1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0, 6.0, 7.0, 0.5)

  r1 <- block_bootstrap_mean(data, block_size = 2L, n_resamples = 500L, confidence_level = 0.9, seed = 123L)
  r2 <- block_bootstrap_mean(data, block_size = 2L, n_resamples = 500L, confidence_level = 0.9, seed = 123L)

  expect_equal(r1$ci_lower, r2$ci_lower)
  expect_equal(r1$ci_upper, r2$ci_upper)
})

test_that("block_bootstrap_mean accepts NULL seed for non-reproducible runs", {
  result <- block_bootstrap_mean(
    data = c(0.1, 0.2, 0.3, 0.4, 0.5),
    block_size = 1L,
    n_resamples = 100L,
    confidence_level = 0.95,
    seed = NULL
  )
  expect_true(is.numeric(result$point_estimate))
})

test_that("cohens_d computes a correct positive effect size", {
  baseline <- c(0.5, 0.6, 0.55, 0.52, 0.58)
  treatment <- c(0.9, 0.92, 0.88, 0.91, 0.89)

  result <- cohens_d(baseline, treatment)

  expect_true(result$cohens_d > 0)
  expect_equal(result$magnitude, "Large")
  expect_equal(result$n_baseline, 5L)
  expect_equal(result$n_treatment, 5L)
  # Hedges' g should be a shrunken version of Cohen's d (small-sample
  # bias correction), so it must be smaller in magnitude.
  expect_true(abs(result$hedges_g) < abs(result$cohens_d))
})

test_that("cohens_d errors cleanly on zero-variance groups", {
  expect_error(
    cohens_d(c(5.0, 5.0, 5.0), c(5.0, 5.0, 5.0)),
    "variance"
  )
})

test_that("cohens_d errors cleanly on insufficient data", {
  expect_error(
    cohens_d(c(1.0), c(1.0, 2.0)),
    regexp = NULL  # just confirm it errors, message wording may vary
  )
})

test_that("RSequentialTest detects a real noisy effect with early stopping", {
  test <- RSequentialTest$new(alpha = 0.05, tau = 0.3)
  noise <- c(0.05, -0.03, 0.02, -0.01, 0.04, -0.02, 0.01, -0.04, 0.03, -0.05)
  rejected_at <- NULL
  for (i in 1:50) {
    x <- 1.0 + noise[(i %% length(noise)) + 1]
    update <- test$update(x)
    if (update$rejected) {
      rejected_at <- update$n
      break
    }
  }
  expect_false(is.null(rejected_at))
  expect_true(rejected_at < 50)
})

test_that("RSequentialTest does not reject under a true null effect", {
  test <- RSequentialTest$new(alpha = 0.05, tau = 0.5)
  rejected <- FALSE
  for (i in 1:500) {
    x <- if (i %% 2 == 0) 1.0 else -1.0
    update <- test$update(x)
    if (update$rejected) {
      rejected <- TRUE
      break
    }
  }
  expect_false(rejected)
})

test_that("correct_p_values: Benjamini-Hochberg rejects at least as many as Bonferroni", {
  p_values <- c(0.001, 0.004, 0.008, 0.02, 0.6, 0.7, 0.8, 0.9)
  bonf <- correct_p_values(p_values, alpha = 0.05, method = "bonferroni")
  bh <- correct_p_values(p_values, alpha = 0.05, method = "benjamini_hochberg")
  expect_true(sum(bh$rejected) >= sum(bonf$rejected))
})

test_that("correct_p_values errors on an unknown method", {
  expect_error(
    correct_p_values(c(0.01, 0.02), alpha = 0.05, method = "not_a_real_method"),
    "unknown method"
  )
})

test_that("debias_pairwise_judgment catches a biased judge (disagreement -> tie)", {
  result <- debias_pairwise_judgment(original_order = "prefers_a", swapped_order = "prefers_b")
  expect_equal(result$preference, "tie")
  expect_true(result$position_bias_detected)
})

test_that("debias_pairwise_judgment trusts a consistent judge (agreement)", {
  result <- debias_pairwise_judgment(original_order = "prefers_a", swapped_order = "prefers_a")
  expect_equal(result$preference, "prefers_a")
  expect_false(result$position_bias_detected)
})

test_that("debias_pairwise_judgment errors on an invalid preference string", {
  expect_error(
    debias_pairwise_judgment(original_order = "not_a_real_preference", swapped_order = "tie"),
    "unknown preference"
  )
})

test_that("length_bias_correction detects and removes a perfect linear length trend", {
  scores <- c(2.0, 4.0, 6.0, 8.0, 10.0)
  lengths <- c(10.0, 20.0, 30.0, 40.0, 50.0)
  result <- length_bias_correction(scores, lengths)

  expect_equal(result$correlation, 1.0, tolerance = 1e-9)
  # After perfectly removing a perfect linear trend, all adjusted
  # scores should collapse to the same value.
  expect_true(all(abs(result$adjusted_scores - result$adjusted_scores[1]) < 1e-9))
})

test_that("length_bias_correction errors on insufficient data", {
  expect_error(
    length_bias_correction(c(1.0, 2.0), c(1.0, 2.0)),
    regexp = NULL
  )
})
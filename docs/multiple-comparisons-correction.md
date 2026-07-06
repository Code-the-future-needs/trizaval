# Multiple comparisons correction

## The problem

An eval suite that checks many metrics, or compares many candidates against a baseline, runs many hypothesis tests at once. Even if none of them reflect a real effect, each test has some chance (typically the significance level, e.g. 5%) of appearing significant purely by chance. With enough tests, a false positive becomes close to guaranteed: with 20 independent tests at alpha=0.05, the expected number of false positives under a true null is 1, and the probability of at least one is roughly 64%.

## The method

Trizaval implements two standard corrections, selectable per suite:

**Bonferroni correction** (Bonferroni, 1936; commonly attributed via Dunn, 1961) multiplies each p-value by the number of tests `m`, capping at 1.0. This controls the **family-wise error rate** — the probability of making *any* false positive across the whole family of tests — at the chosen `alpha`. It is simple and conservative: as `m` grows, individual tests need increasingly small p-values to be called significant, which sacrifices statistical power.

**Benjamini-Hochberg procedure** (Benjamini & Hochberg, 1995) instead controls the **false discovery rate** — the expected *proportion* of false positives among the tests called significant, not the probability of any false positive at all. P-values are sorted ascending, each is compared against a rank-dependent threshold `(rank/m) * alpha`, and rejections are determined by finding the largest rank at which the sorted p-value is below its threshold, then rejecting all tests at or below that rank. This is meaningfully more powerful than Bonferroni when checking many metrics at once, which is the realistic case for an eval suite with many test cases or many candidates.

## Implementation

See `crates/trizaval-core/src/correction.rs`, function `correct_p_values`.

- Benjamini-Hochberg's adjusted p-values are computed as `p_(i) * m / rank`, then a monotonicity correction is applied (adjusted values must not decrease as rank decreases; the running minimum is enforced from the largest rank down), matching the standard implementation used in statistical software such as R's `p.adjust`.
- Both methods return the same result shape: adjusted p-values and a boolean rejection decision per input, in the original input order (not sorted order), so results map directly back to the metrics or candidates they came from.

## Assumptions and limitations

- Benjamini-Hochberg's false discovery rate guarantee formally assumes the tests are either independent or exhibit a specific form of positive dependence (Benjamini & Yekutieli, 2001 extend the guarantee to arbitrary dependence with a different, more conservative procedure, not currently implemented). In practice, moderate positive correlation between eval metrics, as is common when scores derive from overlapping underlying model behavior, is not expected to badly violate this.
- Choosing between the two methods is a real methodological decision, not automatic: Bonferroni is appropriate when any single false positive would be costly (e.g. deciding whether to ship a model change based on one flagged regression); Benjamini-Hochberg is appropriate when reviewing many metrics and some tolerance for a controlled fraction of false positives is acceptable in exchange for substantially higher power.

## References

- Bonferroni, C. E. (1936). "Teoria statistica delle classi e calcolo delle probabilità." *Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commerciali di Firenze*, 8, 3–62.
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). "The Control of the False Discovery Rate in Multiple Testing under Dependency." *The Annals of Statistics*, 29(4), 1165–1188.
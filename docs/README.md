# Trizaval documentation

Methodology references for each statistical method Trizaval implements. These documents cover the actual derivations, citations, and assumptions behind the code in `crates/trizaval-core`, at a level of depth the root README does not attempt.

- [Block bootstrap confidence intervals](bootstrap.md)
- [Sequential hypothesis testing (mixture SPRT)](sequential-testing.md)
- [Multiple comparisons correction](multiple-comparisons-correction.md)
- [Effect size estimation](effect-size.md)
- [LLM judge bias calibration](judge-bias-calibration.md)

Each document follows the same structure: what problem it solves, the method itself, the exact implementation in `trizaval-core`, known limitations, and references.
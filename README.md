# Trizaval

Statistically rigorous evaluation tooling for non-deterministic AI systems.

Trizaval exists because most LLM and agent evaluation today is done with vibes, small manual spot checks, or tools that report a single accuracy number with no confidence interval, no correction for testing many metrics at once, and no correction for known biases in LLM-as-judge scoring. Trizaval brings real statistical rigor to that process: bootstrap confidence intervals, sequential hypothesis testing with early stopping, multiple comparisons correction, effect size estimation, and bias calibration for LLM judges.

The statistical core is written in Rust for correctness and speed, with full Python bindings so it fits naturally into existing ML and eval workflows.

## Why this exists

Teams comparing two models, two prompts, or two agent configurations usually look at something like "accuracy went from 82% to 84%" on a few hundred examples and call it an improvement. Without a confidence interval, there is no way to know if that difference is real or noise. Without correction for multiple comparisons, checking twenty metrics at once all but guarantees a false positive somewhere. Without bias calibration, an LLM judge's raw score may simply be tracking response length or which response was shown first, not actual quality.

Trizaval is built specifically to close those gaps, as a foundational, dependency light statistics layer that other tools and workflows can build on top of.

## What is in this repository

- `crates/trizaval-core`: the pure Rust statistical engine. No I/O, no Python dependency, fully unit tested.
- `crates/trizaval-py`: PyO3 bindings exposing the Rust core to Python, plus a full evaluation harness (suite schema, YAML loader, provider adapters, LLM judge with bias calibration, and a suite runner) and a command line entry point.
- `crates/trizaval-cli`: a standalone Rust binary exposing every core statistical method as a CI friendly command line tool, with text and JSON output.
- `suites/`: example eval suite configuration files.
- `tests/python`: the Python side regression test suite (pytest).

## Statistical methods implemented

**Block bootstrap confidence intervals**
Resamples contiguous blocks of observations rather than single points, preserving correlation between nearby examples such as consecutive prompts on a similar topic. Use `block_size = 1` for an ordinary independent bootstrap.

**Sequential hypothesis testing (mixture SPRT)**
Lets you stop collecting data the moment there is real evidence of an effect, instead of always running a fixed, large sample. Stays statistically valid no matter when you choose to look at the result, as long as you stop the first time it rejects.

**Multiple comparisons correction**
Bonferroni for strict control of any false positive across many metrics, and Benjamini Hochberg for higher powered false discovery rate control when checking many metrics in one suite, which is the more realistic case for eval suites with many test cases.

**Effect size estimation**
Cohen's d and Hedges' g (a small sample bias corrected version of Cohen's d), so a difference between two models can be reported by magnitude, not just by whether it crossed a significance threshold.

**LLM judge bias calibration**
Position bias correction, judging each pairwise comparison twice with the responses swapped and only trusting the result if both judgments agree, otherwise treating it as inconclusive. Length bias correction, fitting and removing the portion of a judge's score explained by response length alone.

## Provider support

Trizaval ships with three provider adapters:

- `OpenAIProvider`, a native integration with the OpenAI API.
- `AnthropicProvider`, a native integration with the Anthropic API.
- `OpenAICompatibleProvider`, a single generic adapter for any provider exposing an OpenAI compatible chat completions endpoint. This covers DeepSeek, xAI's Grok, Moonshot's Kimi, Mistral, Groq, Together, Google's Gemini OpenAI compatible endpoint, and locally hosted models served through Ollama or vLLM.

Adding a new OpenAI compatible provider is a configuration change, not a code change. Set `kind: openai_compatible`, a `base_url`, and an `api_key_env_var` in your suite file.

## Installation

### Python

Trizaval is not yet published to PyPI. To build and install it locally from source:

```bash
git clone https://github.com/Code-the-future-needs/trizaval.git
cd trizaval/crates/trizaval-py
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop
```

This builds the Rust core and installs an importable `trizaval` package into your active virtual environment.

### Rust CLI

```bash
git clone https://github.com/Code-the-future-needs/trizaval.git
cd trizaval
cargo build --release --package trizaval-cli
./target/release/trizaval --help
```

## Quick start: Python statistics API

```python
import trizaval

result = trizaval.block_bootstrap_mean(
    data=[0.8, 0.9, 0.7, 0.85, 0.75, 0.95, 0.6, 0.88, 0.92, 0.7],
    block_size=2,
    n_resamples=2000,
    confidence_level=0.95,
    seed=42,
)

print(result.point_estimate)
print(result.ci_lower, result.ci_upper)
```

The bootstrap function also accepts an arbitrary Python callable as the statistic, not just the mean:

```python
def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

result = trizaval.block_bootstrap(
    data=[0.8, 0.9, 0.7, 0.85, 0.75],
    block_size=1,
    n_resamples=2000,
    confidence_level=0.95,
    statistic=median,
    seed=42,
)
```

## Quick start: running a full eval suite

Write a suite file describing what you want to test:

```yaml
name: arithmetic-sanity-check

baseline:
  name: baseline-gpt4o-mini
  kind: openai
  model: gpt-4o-mini
  temperature: 0.0

candidates:
  - name: candidate-gpt4o
    kind: openai
    model: gpt-4o
    temperature: 0.0

test_cases:
  - id: add-1
    prompt: "What is 2 + 2? Answer with just the number."
    reference: "4"

judge:
  kind: rule_based
  match_type: contains
  case_sensitive: false

statistics:
  method: bootstrap
  block_size: 1
  n_resamples: 2000
  confidence_level: 0.95
  seed: 42
```

Then run it:

```bash
python3 -m trizaval.cli run suites/example_suite.yaml
python3 -m trizaval.cli run suites/example_suite.yaml --format json
```

Set your provider credentials as environment variables first, for example `OPENAI_API_KEY`.

To persist the run's results for later trend analysis, add `--storage-dir`:

```bash
python3 -m trizaval.cli run suites/example_suite.yaml --storage-dir ./eval-history
```

Each run appends a new row to `<storage-dir>/<suite-name>.parquet`, so running the same suite repeatedly builds a queryable history over time. See the Storage and querying eval history section below.

## Quick start: Rust CLI

```bash
trizaval bootstrap --input scores.json --block-size 2 --n-resamples 2000 --confidence-level 0.95 --seed 42
trizaval sequential --input scores.json --alpha 0.05 --tau 0.3
trizaval correction --input p_values.json --method benjamini-hochberg
trizaval effect-size --baseline baseline.json --treatment treatment.json
trizaval judge-length-bias --scores scores.json --lengths lengths.json
trizaval judge-pairwise --original-order prefers-a --swapped-order prefers-b
```

Every command supports `--format json` for CI pipelines.

## Storage and querying eval history

Every suite's run history accumulates in `<storage-dir>/<suite-name>.parquet`, one Parquet file per suite. This can be queried directly with DuckDB, either through trizaval's own helper functions or with arbitrary SQL.

```python
from trizaval.storage.duckdb_store import score_trend, latest_run, query

# Mean score for one candidate across every recorded run, oldest to newest
trend = score_trend("./eval-history", "arithmetic-sanity-check", "candidate-gpt4o")

# Every candidate's results from the most recent run only
latest = latest_run("./eval-history", "arithmetic-sanity-check")

# Arbitrary SQL against the suite's history file
rows = query(
    "./eval-history",
    "arithmetic-sanity-check",
    "SELECT DISTINCT candidate_name FROM {table}",
)
```

`{table}` in a custom query is replaced with a reference to the suite's Parquet file, so any valid DuckDB SQL can be used, including joins, aggregations, and filters on `run_timestamp`.

## Testing

Rust tests, covering the statistical core:

```bash
cargo test --package trizaval-core
```

Python tests, covering the schema, loader, providers, judge, runner, and CLI:

```bash
pytest tests/python/ -v
```

Both suites run automatically on every push and pull request through GitHub Actions.

## Design principles

Statistical correctness first. Every method here has a known name, a known citation, and a documented limitation where one exists, rather than an invented ad hoc approach.

No silent narrowing of scope. Where a limitation exists, such as sequential testing needing a well defined variance estimate, or length bias correction needing at least three data points, the code returns a clear result or error rather than guessing.

Broad provider support without per company code. New OpenAI compatible providers are a configuration change.

Small, dependency light core. The Rust statistics engine has no I/O and no Python dependency, so it can be embedded anywhere, including a future WebAssembly build for browser based visualization.

## Roadmap

Implemented: persistent storage of eval run history (Arrow and Parquet backed, queryable through DuckDB), and a browser based dashboard (`dashboard/`) for visualizing confidence intervals, effect sizes, sequential test trajectories, and score trends over time, computed by the same Rust core compiled to WebAssembly.

Not yet implemented: R language bindings for the statistical core.

**Hosted multi-tenant gateway.** The original design included a `gateway/` layer (Go, per-team authentication, hosted storage) for teams who would rather point their CI at a hosted endpoint than self-host. This is deliberately not built yet. Trizaval is a free, self-hosted library, and running a multi-tenant hosted service means ongoing server costs, security responsibility, and uptime commitments that are out of scope for a project maintained without funding. If a company or volunteer wants to sponsor hosting or contribute this layer, it is a welcome addition. See `crates/` for where a `trizaval-gateway` crate would live, and open an issue if you would like to help build or fund it.

## License

Apache-2.0
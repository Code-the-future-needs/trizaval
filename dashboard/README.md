# Trizaval Dashboard

A browser-based dashboard for visualizing Trizaval eval results: confidence intervals and score trends, computed by the same Rust statistical engine used everywhere else in the project, compiled to WebAssembly and run entirely client side.

No backend, no server, no data leaves your browser. You load a JSON file exported by the Trizaval CLI, and the dashboard recomputes and renders the statistics locally.

## Why WebAssembly

The dashboard does not reimplement any statistics in JavaScript. Instead, `crates/trizaval-wasm` compiles the real `trizaval-core` Rust crate, the same one used by the Python bindings and the Rust CLI, to a WebAssembly module. This guarantees the numbers shown here are identical to what the CLI and Python library would compute, with no risk of a second implementation silently drifting from the first.

## What is currently supported

**Confidence interval chart**
Upload a suite report produced by:

```bash
python3 -m trizaval.cli run suite.yaml --format json
```

The dashboard reads each candidate's raw scores from the report, recomputes a block bootstrap confidence interval live in the browser, and renders it as a bar chart with asymmetric error bars showing the true lower and upper bounds.

**Score trend chart**
Upload a trend export produced by:

```bash
python3 -m trizaval.cli trend <storage_dir> <suite_name> <candidate_name>
```

Renders a candidate's mean score across every recorded run in your eval history (see the root README's Storage section for how run history accumulates), as a line chart over time.

## Running locally

From this directory:

```bash
npm install
npm run dev
```

Then open the printed local URL (typically `http://localhost:5173`) in your browser.

The dashboard imports a compiled WebAssembly module from `./wasm`, which is not committed to the repository (it is a build artifact, like `target/` or a `.so` file). Build it first from the repository root:

```bash
cd ../crates/trizaval-wasm
wasm-pack build --target web --out-dir ../../dashboard/wasm
```

You will need [`wasm-pack`](https://rustwasm.github.io/wasm-pack/installer/) installed:

```bash
cargo install wasm-pack
```

## Building for production

```bash
npm run build
```

Output goes to `dist/`, a static site with no server requirements, deployable to any static host.

## Project structure

dashboard/
├── src/
│   ├── App.tsx                     # top-level layout, file upload handling
│   ├── ConfidenceIntervalChart.tsx # loads WASM, computes and renders bootstrap CIs
│   ├── ScoreTrendChart.tsx         # renders a score trend export as a line chart
│   └── sampleData.ts               # sample data matching real CLI output shapes,
│                                       shown before a user uploads their own file
├── wasm/                           # NOT committed; build output from trizaval-wasm
└── vite.config.ts                  # includes vite-plugin-wasm and esnext build target,
both required for loading the WASM module

## Known limitations

This is an early version, intentionally scoped narrowly rather than built out speculatively:

- Only two chart types exist: bootstrap confidence intervals and score trends. Sequential test trajectories and effect size visualizations are not yet implemented.
- Only `block_bootstrap_mean` and `cohens_d` are exposed through the WASM bindings so far, matching what the current charts need. See `crates/trizaval-wasm/src/lib.rs` for the exact exposed surface.
- File loading is manual (a file picker), not a live connection to a running eval or a storage directory. Every chart update requires exporting a new JSON file via the CLI and re-uploading it.
import { useEffect, useState } from 'react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ErrorBar,
  ResponsiveContainer,
} from 'recharts'
import type { SuiteReportData } from './sampleData'

interface ChartRow {
  candidate: string
  mean: number
  ciLower: number
  ciUpper: number
  errorLow: number
  errorHigh: number
}

interface WasmModule {
  block_bootstrap_mean: (
    data: Float64Array,
    blockSize: number,
    nResamples: number,
    confidenceLevel: number,
    seed?: number
  ) => { point_estimate: number; ci_lower: number; ci_upper: number }
}

export function ConfidenceIntervalChart({ report }: { report: SuiteReportData }) {
  const [rows, setRows] = useState<ChartRow[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function computeFromWasm() {
      try {
        // Dynamic import so the WASM binary only loads when this
        // component actually mounts, not on initial app load.
        const wasmModule = (await import(
          /* @vite-ignore */ '../wasm/trizaval_wasm.js'
        )) as { default: (path?: unknown) => Promise<unknown> } & WasmModule

        await wasmModule.default()

        const computed: ChartRow[] = report.candidate_reports.map((cr) => {
          const result = wasmModule.block_bootstrap_mean(
            new Float64Array(cr.candidate_scores),
            1,
            2000,
            0.95,
            42
          )
          return {
            candidate: cr.candidate_name,
            mean: result.point_estimate,
            ciLower: result.ci_lower,
            ciUpper: result.ci_upper,
            errorLow: result.point_estimate - result.ci_lower,
            errorHigh: result.ci_upper - result.point_estimate,
          }
        })

        if (!cancelled) setRows(computed)
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    }

    computeFromWasm()
    return () => {
      cancelled = true
    }
  }, [report])

  if (error) {
    return <div style={{ color: 'crimson' }}>Failed to compute statistics in-browser: {error}</div>
  }

  if (!rows) {
    return <div>Loading WASM statistics engine...</div>
  }

  return (
    <div>
      <h3>{report.suite_name}: candidate mean score with 95% CI (computed live in-browser via WASM)</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={rows} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="candidate" />
          <YAxis domain={[0, 1]} />
<Tooltip
            formatter={(value, name) => [
              typeof value === 'number' ? value.toFixed(4) : String(value),
              String(name),
            ]}
          />
<Bar dataKey="mean" fill="#4f7cff">
            <ErrorBar
              dataKey={(row: ChartRow) => [row.errorLow, row.errorHigh]}
              width={6}
              strokeWidth={2}
              stroke="#1f2d5c"
              direction="y"
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
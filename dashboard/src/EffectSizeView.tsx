import { useEffect, useState } from 'react'
import type { SuiteReportData } from './sampleData'
import { loadWasm } from './wasmLoader'

interface WasmEffectSizeResult {
  cohens_d: number
  hedges_g: number
  magnitude: string
  n_baseline: number
  n_treatment: number
}

interface WasmModule {
  cohens_d: (baseline: Float64Array, treatment: Float64Array) => WasmEffectSizeResult
}

const MAGNITUDE_COLORS: Record<string, string> = {
  Negligible: '#888',
  Small: '#4f9dff',
  Medium: '#ff9f4f',
  Large: '#ff4f6a',
}

export function EffectSizeView({ report }: { report: SuiteReportData }) {
  const [results, setResults] = useState<
    Array<{ candidate: string; result: WasmEffectSizeResult | null; error: string | null }>
    | null
  >(null)

  useEffect(() => {
    let cancelled = false

async function computeFromWasm() {
      const wasmModule = (await loadWasm()) as unknown as WasmModule

      const computed = report.candidate_reports.map((cr) => {
        try {
          const result = wasmModule.cohens_d(
            new Float64Array(cr.baseline_scores),
            new Float64Array(cr.candidate_scores)
          )
          return { candidate: cr.candidate_name, result, error: null }
        } catch (e) {
          return {
            candidate: cr.candidate_name,
            result: null,
            error: e instanceof Error ? e.message : String(e),
          }
        }
      })

      if (!cancelled) setResults(computed)
    }

    computeFromWasm()
    return () => {
      cancelled = true
    }
  }, [report])

  if (!results) {
    return <div>Loading WASM statistics engine...</div>
  }

  return (
    <div>
      <h3>{report.suite_name}: effect size vs. baseline (computed live in-browser via WASM)</h3>
      {results.map(({ candidate, result, error }) => (
        <div
          key={candidate}
          style={{
            border: '1px solid #333',
            borderRadius: 8,
            padding: '1rem',
            marginBottom: '0.75rem',
          }}
        >
          <strong>{candidate}</strong>
          {error && (
            <div style={{ color: 'crimson', marginTop: '0.5rem' }}>
              Could not compute effect size: {error}
            </div>
          )}
          {result && (
            <div style={{ marginTop: '0.5rem', display: 'flex', gap: '2rem', alignItems: 'center' }}>
              <span>
                Cohen&apos;s d: <strong>{result.cohens_d.toFixed(4)}</strong>
              </span>
              <span>
                Hedges&apos; g: <strong>{result.hedges_g.toFixed(4)}</strong>
              </span>
              <span
                style={{
                  color: MAGNITUDE_COLORS[result.magnitude] ?? '#888',
                  fontWeight: 'bold',
                  border: `1px solid ${MAGNITUDE_COLORS[result.magnitude] ?? '#888'}`,
                  borderRadius: 4,
                  padding: '0.2rem 0.6rem',
                }}
              >
                {result.magnitude}
              </span>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
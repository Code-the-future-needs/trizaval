import { useEffect, useState } from 'react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { SuiteReportData } from './sampleData'
import { loadWasm } from './wasmLoader'

interface WasmSequentialUpdate {
  n: number
  likelihood_ratio: number
  rejected: boolean
}

interface WasmSequentialTestCtor {
  new (alpha: number, tau: number): {
    update: (x: number) => WasmSequentialUpdate
  }
}

interface WasmModule {
  WasmSequentialTest: WasmSequentialTestCtor
}

type CandidateReportDataWithIds = SuiteReportData['candidate_reports'][number] & {
  baseline_test_case_ids?: string[]
  candidate_test_case_ids?: string[]
}

const ALPHA = 0.05
const TAU = 0.5
const REJECTION_THRESHOLD = 1 / ALPHA


function pairedDifferencesById(
  baselineScores: number[],
  baselineIds: string[] | undefined,
  candidateScores: number[],
  candidateIds: string[] | undefined
): number[] {
  if (!baselineIds || !candidateIds) {
    throw new Error(
      'this report is missing per-test-case ids (baseline_test_case_ids / candidate_test_case_ids); ' +
        're-export it with a version of the CLI that includes them'
    )
  }

  const candidateById = new Map<string, number>()
  candidateIds.forEach((id, i) => candidateById.set(id, candidateScores[i]))

  const differences: number[] = []
  baselineIds.forEach((id, i) => {
    if (candidateById.has(id)) {
      differences.push(candidateById.get(id)! - baselineScores[i])
    }
  })
  return differences
}

export function SequentialTrajectoryChart({ report }: { report: SuiteReportData }) {
  const [trajectories, setTrajectories] = useState<Array<{ candidate: string; points: Array<{ n: number; likelihood_ratio: number }>; rejectedAt: number | null }> | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    async function computeFromWasm() {
      try {
        const wasmModule = (await loadWasm()) as unknown as WasmModule

        const computed = report.candidate_reports.map((cr: CandidateReportDataWithIds) => {
          const differences = pairedDifferencesById(
            cr.baseline_scores,
            cr.baseline_test_case_ids,
            cr.candidate_scores,
            cr.candidate_test_case_ids
          )

          const test = new wasmModule.WasmSequentialTest(ALPHA, TAU)
          const points: Array<{ n: number; likelihood_ratio: number }> = []
          let rejectedAt: number | null = null

          for (const diff of differences) {
            const update = test.update(diff)
            points.push({ n: update.n, likelihood_ratio: update.likelihood_ratio })
            if (update.rejected && rejectedAt === null) {
              rejectedAt = update.n
            }
          }

          return { candidate: cr.candidate_name, points, rejectedAt }
        })

        if (!cancelled) setTrajectories(computed)
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
    return <div style={{ color: 'crimson' }}>Failed to compute sequential trajectory: {error}</div>
  }

  if (!trajectories) {
    return <div>Loading WASM statistics engine...</div>
  }

  return (
    <div>
      <h3>
        Sequential test trajectory (alpha={ALPHA}, tau={TAU}), computed live in-browser via WASM
      </h3>
      <p style={{ color: '#888', fontSize: '0.9rem' }}>
        Likelihood ratio per paired (candidate - baseline) observation, aligned by test case id.
        Crossing the dashed threshold line means the null hypothesis (no effect) is rejected at
        that point.
      </p>
      {trajectories.map(({ candidate, points, rejectedAt }) => (
        <div key={candidate} style={{ marginBottom: '2rem' }}>
          <strong>
            {candidate}
            {rejectedAt !== null ? ` (rejected at n=${rejectedAt})` : ' (not rejected)'}
          </strong>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={points} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="n" label={{ value: 'Observation n', position: 'insideBottom', offset: -5 }} />
              <YAxis />
              <Tooltip formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : String(value))} />
              <ReferenceLine
                y={REJECTION_THRESHOLD}
                stroke="crimson"
                strokeDasharray="4 4"
                label={{ value: 'Rejection threshold', position: 'insideTopRight', fill: 'crimson' }}
              />
              <Line type="monotone" dataKey="likelihood_ratio" stroke="#4f7cff" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      ))}
    </div>
  )
}
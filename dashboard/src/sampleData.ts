// Sample data matching the real shape produced by
// `python3 -m trizaval.cli run suite.yaml --format json`.
// Used for initial dashboard development before wiring up live file
// loading; the shape here is not invented, it mirrors what the CLI
// actually outputs (see crates/trizaval-py/python/trizaval/cli.py).

export interface CandidateReportData {
  candidate_name: string
  baseline_scores: number[]
  baseline_test_case_ids?: string[]
  candidate_scores: number[]
  candidate_test_case_ids?: string[]
  statistic_result: {
    method: string
    point_estimate?: number
    ci_lower?: number
    ci_upper?: number
    confidence_level?: number
    cohens_d?: number
    hedges_g?: number
    magnitude?: string
  } | null
  errors: string[]
}

export interface SuiteReportData {
  suite_name: string
  candidate_reports: CandidateReportData[]
}

export const sampleReport: SuiteReportData = {
  suite_name: 'arithmetic-sanity-check',
  candidate_reports: [
{
      candidate_name: 'candidate-gpt4o',
      baseline_scores: [1.0, 1.0, 1.0, 1.0],
      baseline_test_case_ids: ['add-1', 'add-2', 'mult-1', 'sub-1'],
      candidate_scores: [1.0, 0.0, 1.0, 1.0],
      candidate_test_case_ids: ['add-1', 'add-2', 'mult-1', 'sub-1'],
      statistic_result: {
        method: 'bootstrap',
        point_estimate: 0.75,
        ci_lower: 0.25,
        ci_upper: 1.0,
        confidence_level: 0.95,
      },
      errors: [],
    },
  ],
}

// Sample data matching `python3 -m trizaval.cli trend <dir> <suite> <candidate>` output.
export interface TrendPoint {
  run_id: string
  run_timestamp: string
  mean_score: number
}

export interface TrendData {
  suite_name: string
  candidate_name: string
  trend: TrendPoint[]
}

export const sampleTrend: TrendData = {
  suite_name: 'arithmetic-sanity-check',
  candidate_name: 'candidate-gpt4o',
  trend: [
    { run_id: 'run-1', run_timestamp: '2026-01-01T00:00:00', mean_score: 0.25 },
    { run_id: 'run-2', run_timestamp: '2026-01-08T00:00:00', mean_score: 0.5 },
    { run_id: 'run-3', run_timestamp: '2026-01-15T00:00:00', mean_score: 0.75 },
    { run_id: 'run-4', run_timestamp: '2026-01-22T00:00:00', mean_score: 1.0 },
  ],
}
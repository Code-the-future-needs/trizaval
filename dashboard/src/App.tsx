import { useState } from 'react'
import { ConfidenceIntervalChart } from './ConfidenceIntervalChart'
import { ScoreTrendChart } from './ScoreTrendChart'
import { sampleReport, sampleTrend, type SuiteReportData, type TrendData } from './sampleData'

function App() {
  const [report, setReport] = useState<SuiteReportData>(sampleReport)
  const [reportLoadError, setReportLoadError] = useState<string | null>(null)

  const [trend, setTrend] = useState<TrendData>(sampleTrend)
  const [trendLoadError, setTrendLoadError] = useState<string | null>(null)

  async function handleReportFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    setReportLoadError(null)
    try {
      const text = await file.text()
      const parsed = JSON.parse(text) as SuiteReportData

      if (!parsed.suite_name || !Array.isArray(parsed.candidate_reports)) {
        throw new Error(
          "file does not match trizaval's suite report shape (missing suite_name or candidate_reports)"
        )
      }

      setReport(parsed)
    } catch (e) {
      setReportLoadError(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleTrendFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    setTrendLoadError(null)
    try {
      const text = await file.text()
      const parsed = JSON.parse(text) as TrendData

      if (!parsed.suite_name || !parsed.candidate_name || !Array.isArray(parsed.trend)) {
        throw new Error(
          "file does not match trizaval's trend export shape (missing suite_name, candidate_name, or trend)"
        )
      }

      setTrend(parsed)
    } catch (e) {
      setTrendLoadError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Trizaval Dashboard</h1>

      <div style={{ marginBottom: '1.5rem' }}>
        <label htmlFor="report-upload" style={{ display: 'block', marginBottom: '0.5rem' }}>
          Load a suite report (from{' '}
          <code>python3 -m trizaval.cli run suite.yaml --format json</code>):
        </label>
        <input id="report-upload" type="file" accept=".json" onChange={handleReportFileChange} />
        {reportLoadError && (
          <div style={{ color: 'crimson', marginTop: '0.5rem' }}>
            Failed to load file: {reportLoadError}
          </div>
        )}
        {report === sampleReport && !reportLoadError && (
          <div style={{ color: '#888', marginTop: '0.5rem', fontStyle: 'italic' }}>
            Showing sample data. Upload a report file to see your own results.
          </div>
        )}
      </div>

      <ConfidenceIntervalChart report={report} />

      <hr style={{ margin: '2rem 0', borderColor: '#333' }} />

      <div style={{ marginBottom: '1.5rem' }}>
        <label htmlFor="trend-upload" style={{ display: 'block', marginBottom: '0.5rem' }}>
          Load a score trend (from{' '}
          <code>python3 -m trizaval.cli trend &lt;storage_dir&gt; &lt;suite_name&gt; &lt;candidate_name&gt;</code>):
        </label>
        <input id="trend-upload" type="file" accept=".json" onChange={handleTrendFileChange} />
        {trendLoadError && (
          <div style={{ color: 'crimson', marginTop: '0.5rem' }}>
            Failed to load file: {trendLoadError}
          </div>
        )}
        {trend === sampleTrend && !trendLoadError && (
          <div style={{ color: '#888', marginTop: '0.5rem', fontStyle: 'italic' }}>
            Showing sample data. Upload a trend export to see your own history.
          </div>
        )}
      </div>

      <ScoreTrendChart trend={trend} />
    </div>
  )
}

export default App
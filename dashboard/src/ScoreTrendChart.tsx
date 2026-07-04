import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import type { TrendData } from './sampleData'

export function ScoreTrendChart({ trend }: { trend: TrendData }) {
  const chartData = trend.trend.map((point) => ({
    // Shorten to a readable date for the x-axis; full timestamp
    // still available in the tooltip via the raw point data.
    label: new Date(point.run_timestamp).toLocaleDateString(),
    mean_score: point.mean_score,
    run_id: point.run_id,
  }))

  return (
    <div>
      <h3>
        {trend.candidate_name} mean score over time ({trend.suite_name})
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="label" />
          <YAxis domain={[0, 1]} />
<Tooltip
            formatter={(value) => (typeof value === 'number' ? value.toFixed(4) : String(value))}
            labelFormatter={(label) => `Run date: ${label}`}
          />
          <Line type="monotone" dataKey="mean_score" stroke="#4f7cff" strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
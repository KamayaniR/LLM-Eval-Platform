import { useState, useEffect, useCallback } from 'react'

function ScoreBadge({ label, value }) {
  if (value === null || value === undefined) {
    return (
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-300">—</span>
      </div>
    )
  }

  const pct = Math.round(value * 100)
  let color = 'text-teal-700 bg-teal-50'
  if (pct < 75) color = 'text-amber-700 bg-amber-50'
  if (pct < 50) color = 'text-rose-700 bg-rose-50'

  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-slate-500">{label}</span>
      <span className={`font-mono px-2 py-0.5 rounded-md ${color}`}>{pct}%</span>
    </div>
  )
}

export default function ResultsView({ jobId }) {
  const [results, setResults] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState(null)

  const fetchData = useCallback(async () => {
    try {
      const [resultsRes, statusRes] = await Promise.all([
        fetch(`/api/results/${jobId}`),
        fetch(`/api/jobs/${jobId}`)
      ])
      const resultsData = await resultsRes.json()
      const statusData = await statusRes.json()
      setResults(resultsData.results || [])
      setStatus(statusData.status)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }, [jobId])

  useEffect(() => {
    setLoading(true)
    setExpanded(null)
    fetchData()

    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [jobId, fetchData])

  if (loading) {
    return (
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <p className="text-sm text-slate-400">Loading results...</p>
      </div>
    )
  }

  const avgComposite = results && results.length > 0
    ? results.reduce((sum, r) => sum + (r.composite_score || 0), 0) / results.length
    : null

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-base font-medium text-slate-900">
          Results
          <span className="ml-2 text-xs font-mono text-slate-300">{jobId.slice(0, 8)}</span>
        </h2>
        {avgComposite !== null && (
          <div className="text-sm bg-slate-50 rounded-lg px-3 py-1.5">
            <span className="text-slate-400">Avg composite </span>
            <span className="font-mono font-medium text-slate-700">{Math.round(avgComposite * 100)}%</span>
          </div>
        )}
      </div>

      {(!results || results.length === 0) && (
        <p className="text-sm text-slate-400">
          {status === 'pending' ? 'Job is processing, results will appear here...' : 'No results yet'}
        </p>
      )}

      <div className="space-y-2">
        {results && results.map((r, i) => (
          <div key={i} className="border border-slate-100 rounded-lg overflow-hidden">
            <button
              onClick={() => setExpanded(expanded === i ? null : i)}
              className="w-full text-left px-4 py-3 hover:bg-slate-50 transition flex items-center justify-between gap-3"
            >
              <span className="text-sm text-slate-700 truncate flex-1">{r.prompt}</span>
              <span className="text-xs font-mono text-slate-400 flex-shrink-0 bg-slate-100 px-2 py-0.5 rounded-md">
                {r.composite_score !== null ? `${Math.round(r.composite_score * 100)}%` : '—'}
              </span>
            </button>

            {expanded === i && (
              <div className="px-4 pb-4 border-t border-slate-100 pt-3 space-y-3 bg-slate-50/50">
                <div>
                  <p className="text-xs text-slate-400 mb-1">Response</p>
                  <p className="text-sm text-slate-600 whitespace-pre-wrap">{r.response}</p>
                </div>
                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-slate-100">
                  <ScoreBadge label="Toxicity" value={r.toxicity_score} />
                  <ScoreBadge label="Instruction" value={r.instruction_score} />
                  <ScoreBadge label="Factuality" value={r.factuality_score} />
                  <ScoreBadge label="Reward" value={r.reward_score} />
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

import { useState, useEffect, useCallback } from 'react'

const MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
]

function avgScore(results, key) {
  const vals = results.map(r => r[key]).filter(v => v !== null && v !== undefined)
  if (vals.length === 0) return null
  return vals.reduce((a, b) => a + b, 0) / vals.length
}

function MetricRow({ label, a, b }) {
  const aPct = a !== null ? Math.round(a * 100) : null
  const bPct = b !== null ? Math.round(b * 100) : null
  const diff = (aPct !== null && bPct !== null) ? bPct - aPct : null

  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <span className="text-sm text-gray-600">{label}</span>
      <div className="flex items-center" style={{gap: "1rem"}}>
        <span className="text-sm font-mono w-12 text-right">{aPct !== null ? `${aPct}%` : '—'}</span>
        <span className="text-sm font-mono w-12 text-right">{bPct !== null ? `${bPct}%` : '—'}</span>
        <span className={`text-xs font-mono w-14 text-right ${
          diff === null ? 'text-gray-300' : diff > 0 ? 'text-green-600' : diff < 0 ? 'text-red-600' : 'text-gray-400'
        }`}>
          {diff === null ? '—' : diff > 0 ? `+${diff}` : diff}
        </span>
      </div>
    </div>
  )
}

function Verdict({ resultsA, resultsB, modelA, modelB }) {
  const metrics = ['composite_score', 'toxicity_score', 'instruction_score', 'factuality_score']
  let scoreA = 0, scoreB = 0, counted = 0

  metrics.forEach(m => {
    const a = avgScore(resultsA, m)
    const b = avgScore(resultsB, m)
    if (a !== null && b !== null) {
      const aPct = Math.round(a * 100)
      const bPct = Math.round(b * 100)
      counted++
      if (aPct > bPct) scoreA++
      else if (bPct > aPct) scoreB++
    }
  })

  const labelA = MODELS.find(x => x.value === modelA)?.label || modelA
  const labelB = MODELS.find(x => x.value === modelB)?.label || modelB

  let verdict, detail
  if (scoreA === scoreB) {
    verdict = "Roughly tied"
    detail = `${labelA} and ${labelB} performed similarly across the measured dimensions.`
  } else if (scoreA > scoreB) {
    verdict = `${labelA} edges ahead`
    detail = `${labelA} scored higher on ${scoreA} of ${counted} dimensions compared to ${labelB}.`
  } else {
    verdict = `${labelB} edges ahead`
    detail = `${labelB} scored higher on ${scoreB} of ${counted} dimensions compared to ${labelA}.`
  }

  const compA = avgScore(resultsA, 'composite_score')
  const compB = avgScore(resultsB, 'composite_score')
  const compDiff = compA !== null && compB !== null ? Math.abs(Math.round(compB * 100) - Math.round(compA * 100)) : null

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-teal-50 border border-indigo-100 rounded-lg p-4 mb-4">
      <p className="text-sm font-medium text-slate-900">{verdict}</p>
      <p className="text-xs text-slate-500 mt-1">{detail}</p>
      {compDiff !== null && (
        <p className="text-xs text-slate-400 mt-1">
          Composite score difference: {compDiff} percentage point{compDiff === 1 ? '' : 's'}
        </p>
      )}
    </div>
  )
}

export default function CompareView() {
  const [source, setSource] = useState('truthfulqa')
  const [samplePrompts, setSamplePrompts] = useState([])
  const [selectedPrompts, setSelectedPrompts] = useState([])
  const [customText, setCustomText] = useState('')
  const [modelA, setModelA] = useState('gemini-2.5-flash')
  const [modelB, setModelB] = useState('gemini-2.5-pro')
  const [jobA, setJobA] = useState(null)
  const [jobB, setJobB] = useState(null)
  const [resultsA, setResultsA] = useState(null)
  const [resultsB, setResultsB] = useState(null)
  const [loading, setLoading] = useState(false)
  const [activePrompts, setActivePrompts] = useState([])

  useEffect(() => {
    fetch('/api/sample-prompts')
      .then(res => res.json())
      .then(data => {
        setSamplePrompts(data.prompts)
        setSelectedPrompts(data.prompts.slice(0, 5))
      })
      .catch(() => {})
  }, [])

  const togglePrompt = (p) => {
    setSelectedPrompts(prev =>
      prev.includes(p) ? prev.filter(x => x !== p) : [...prev, p]
    )
  }

  const getPromptsToRun = () => {
    if (source === 'truthfulqa') return selectedPrompts
    return customText.split('\n').map(p => p.trim()).filter(p => p.length > 0)
  }

  const runComparison = async () => {
    const prompts = getPromptsToRun()
    if (prompts.length === 0) return
    setLoading(true)
    setResultsA(null)
    setResultsB(null)
    setActivePrompts(prompts)

    try {
      const [resA, resB] = await Promise.all([
        fetch('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompts, model: modelA })
        }).then(r => r.json()),
        fetch('/api/jobs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompts, model: modelB })
        }).then(r => r.json())
      ])
      setJobA(resA.job_id)
      setJobB(resB.job_id)
    } catch (err) {
      console.error(err)
      setLoading(false)
    }
  }

  const pollResults = useCallback(async () => {
    if (!jobA || !jobB) return
    try {
      const [a, b] = await Promise.all([
        fetch(`/api/results/${jobA}`).then(r => r.json()),
        fetch(`/api/results/${jobB}`).then(r => r.json())
      ])
      setResultsA(a.results || [])
      setResultsB(b.results || [])

      if ((a.results || []).length >= activePrompts.length &&
          (b.results || []).length >= activePrompts.length) {
        setLoading(false)
      }
    } catch (err) {
      console.error(err)
    }
  }, [jobA, jobB, activePrompts.length])

  useEffect(() => {
    if (!jobA || !jobB) return
    pollResults()
    const interval = setInterval(pollResults, 4000)
    return () => clearInterval(interval)
  }, [jobA, jobB, pollResults])

  const promptsToRun = getPromptsToRun()

  return (
    <div className="space-y-6">
      <div className="bg-white border border-gray-200 rounded-lg p-5">
        <h2 className="text-base font-medium text-gray-900 mb-3">Model comparison</h2>

        <div className="flex" style={{gap: "0.5rem", marginBottom: "1rem"}}>
          <button
            onClick={() => setSource('truthfulqa')}
            className={`text-xs px-3 py-1.5 rounded-md font-medium transition ${
              source === 'truthfulqa' ? 'bg-gray-900 text-white' : 'text-gray-500 bg-gray-100 hover:bg-gray-200'
            }`}
          >
            TruthfulQA samples
          </button>
          <button
            onClick={() => setSource('custom')}
            className={`text-xs px-3 py-1.5 rounded-md font-medium transition ${
              source === 'custom' ? 'bg-gray-900 text-white' : 'text-gray-500 bg-gray-100 hover:bg-gray-200'
            }`}
          >
            Custom prompts
          </button>
        </div>

        {source === 'truthfulqa' ? (
          <>
            <p className="text-sm text-gray-500 mb-4">
              Questions designed to expose hallucinations on common misconceptions.
            </p>
            <div className="space-y-1.5 mb-4">
              {samplePrompts.map((p, i) => (
                <label key={i} className="flex items-start gap-2 text-sm cursor-pointer hover:bg-gray-50 px-2 py-1.5 rounded-md">
                  <input
                    type="checkbox"
                    checked={selectedPrompts.includes(p)}
                    onChange={() => togglePrompt(p)}
                    className="mt-0.5"
                  />
                  <span className="text-gray-700">{p}</span>
                </label>
              ))}
            </div>
          </>
        ) : (
          <>
            <p className="text-sm text-gray-500 mb-2">
              Enter your own prompts, one per line — useful for testing domain-specific or use-case-specific questions.
            </p>
            <textarea
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              placeholder="Explain how a transformer model works&#10;What are the side effects of ibuprofen?&#10;Summarize the plot of Hamlet"
              rows={5}
              className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none mb-4"
            />
          </>
        )}

        <div className="flex items-center flex-wrap" style={{gap: "0.75rem"}}>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Model A</label>
            <select value={modelA} onChange={e => setModelA(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm">
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
          <div>
            <label className="block text-xs text-gray-500 mb-1">Model B</label>
            <select value={modelB} onChange={e => setModelB(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2 text-sm">
              {MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
            </select>
          </div>
          <button
            onClick={runComparison}
            disabled={loading || promptsToRun.length === 0}
            className="mt-5 bg-gray-900 text-white text-sm font-medium px-4 py-2 rounded-md hover:bg-gray-800 disabled:opacity-50 transition"
          >
            {loading ? 'Running...' : `Compare on ${promptsToRun.length} prompt${promptsToRun.length === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>

      {(resultsA !== null && resultsB !== null) && (
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <h3 className="text-sm font-medium text-gray-900 mb-1">Results</h3>
          <p className="text-xs text-gray-400 mb-4">
            {resultsA.length}/{activePrompts.length} · {resultsB.length}/{activePrompts.length} completed
            {loading && ' — still processing...'}
          </p>

          {!loading && resultsA.length === activePrompts.length && resultsB.length === activePrompts.length && (
            <Verdict resultsA={resultsA} resultsB={resultsB} modelA={modelA} modelB={modelB} />
          )}

          <div className="flex items-center justify-between text-xs text-gray-400 mb-2 px-2">
            <span>Metric</span>
            <div className="flex items-center" style={{gap: "1rem"}}>
              <span className="w-12 text-right">{MODELS.find(m => m.value === modelA)?.label.split(' ').pop()}</span>
              <span className="w-12 text-right">{MODELS.find(m => m.value === modelB)?.label.split(' ').pop()}</span>
              <span className="w-14 text-right">Diff</span>
            </div>
          </div>

          <MetricRow label="Composite" a={avgScore(resultsA, 'composite_score')} b={avgScore(resultsB, 'composite_score')} />
          <MetricRow label="Toxicity" a={avgScore(resultsA, 'toxicity_score')} b={avgScore(resultsB, 'toxicity_score')} />
          <MetricRow label="Instruction" a={avgScore(resultsA, 'instruction_score')} b={avgScore(resultsB, 'instruction_score')} />
          <MetricRow label="Factuality" a={avgScore(resultsA, 'factuality_score')} b={avgScore(resultsB, 'factuality_score')} />

          <div className="mt-4 pt-4 border-t border-gray-100">
            <h4 className="text-xs text-gray-400 mb-2">Per-question factuality</h4>
            <div className="space-y-1.5">
              {activePrompts.map((p, i) => {
                const a = resultsA[i]
                const b = resultsB[i]
                return (
                  <div key={i} className="flex items-center justify-between text-xs" style={{gap: "0.5rem"}}>
                    <span className="text-gray-600 truncate flex-1">{p}</span>
                    <span className="font-mono w-12 text-right">
                      {a?.factuality_score !== undefined && a?.factuality_score !== null ? `${Math.round(a.factuality_score * 100)}%` : '—'}
                    </span>
                    <span className="font-mono w-12 text-right">
                      {b?.factuality_score !== undefined && b?.factuality_score !== null ? `${Math.round(b.factuality_score * 100)}%` : '—'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

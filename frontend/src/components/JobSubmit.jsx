import { useState } from 'react'

const MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
]

export default function JobSubmit({ onJobCreated }) {
  const [promptsText, setPromptsText] = useState('')
  const [model, setModel] = useState('gemini-2.5-flash')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    const prompts = promptsText
      .split('\n')
      .map(p => p.trim())
      .filter(p => p.length > 0)

    if (prompts.length === 0) {
      setError('Enter at least one prompt')
      return
    }

    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/api/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompts, model })
      })
      const data = await res.json()
      onJobCreated({
        job_id: data.job_id,
        status: data.status,
        prompt_count: data.prompt_count,
        model,
        created_at: new Date().toISOString()
      })
      setPromptsText('')
    } catch (err) {
      setError('Failed to submit job. Is the API running on port 8080?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
      <h2 className="text-base font-medium text-slate-900 mb-1">Submit eval job</h2>
      <p className="text-sm text-slate-400 mb-4">One prompt per line, sent to the selected model and scored automatically.</p>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-slate-600 mb-1.5">Prompts</label>
          <textarea
            value={promptsText}
            onChange={(e) => setPromptsText(e.target.value)}
            placeholder="What is the capital of France?&#10;Who wrote Romeo and Juliet?"
            rows={4}
            className="w-full border border-slate-200 rounded-lg px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-200 focus:border-indigo-300 resize-none bg-slate-50"
          />
        </div>

        <div className="flex items-center" style={{gap: "0.75rem"}}>
          <div>
            <label className="block text-sm text-slate-600 mb-1.5">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="border border-slate-200 rounded-lg px-3 py-2.5 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-indigo-200"
            >
              {MODELS.map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-6 bg-indigo-600 text-white text-sm font-medium px-5 py-2.5 rounded-lg hover:bg-indigo-700 disabled:opacity-50 transition shadow-sm"
          >
            {loading ? 'Submitting...' : 'Submit job'}
          </button>
        </div>

        {error && (
          <p className="text-sm text-rose-500">{error}</p>
        )}
      </form>
    </div>
  )
}

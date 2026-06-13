function formatTime(iso) {
  const d = new Date(iso)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const MODEL_COLORS = {
  'gemini-2.5-flash': 'bg-teal-50 text-teal-700',
  'gemini-2.5-pro': 'bg-indigo-50 text-indigo-700',
}

export default function JobList({ jobs, activeJobId, onSelect }) {
  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
      <h2 className="text-sm font-medium text-slate-900 mb-3">Recent jobs</h2>
      {jobs.length === 0 ? (
        <p className="text-sm text-slate-400">No jobs submitted yet</p>
      ) : (
        <div className="space-y-2">
          {jobs.map(job => (
            <button
              key={job.job_id}
              onClick={() => onSelect(job.job_id)}
              className={`w-full text-left px-3 py-2.5 rounded-lg border text-sm transition ${
                activeJobId === job.job_id
                  ? 'border-indigo-300 bg-indigo-50/50'
                  : 'border-slate-100 hover:border-slate-200 hover:bg-slate-50'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${MODEL_COLORS[job.model] || 'bg-slate-100 text-slate-600'}`}>
                  {job.model.replace('gemini-2.5-', '')}
                </span>
                <span className="text-xs text-slate-400">{formatTime(job.created_at)}</span>
              </div>
              <div className="flex items-center justify-between mt-1.5">
                <span className="text-xs text-slate-500">{job.prompt_count} prompts</span>
                <span className="text-xs font-mono text-slate-300">
                  {job.job_id.slice(0, 8)}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

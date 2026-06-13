import { useState } from 'react'
import JobSubmit from './components/JobSubmit'
import JobList from './components/JobList'
import ResultsView from './components/ResultsView'
import CompareView from './components/CompareView'

function App() {
  const [activeJobId, setActiveJobId] = useState(null)
  const [jobs, setJobs] = useState([])
  const [mode, setMode] = useState('single')

  const addJob = (job) => {
    setJobs(prev => [job, ...prev])
    setActiveJobId(job.job_id)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <header className="bg-gradient-to-br from-slate-800 via-indigo-900 to-slate-800 text-white">
        <div className="max-w-5xl mx-auto px-6 py-10">
          <div className="inline-flex items-center gap-2 bg-white/10 border border-white/10 rounded-full px-3 py-1 text-xs font-medium text-indigo-200 mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-teal-400"></span>
            RLHF eval platform
          </div>
          <h1 className="text-3xl font-medium mb-2">LLM Eval Platform</h1>
          <p className="text-indigo-100/80 max-w-2xl leading-relaxed mb-5">
            Submit prompts to Gemini and score the responses across four dimensions —
            toxicity, instruction following, factuality, and a custom reward model
            trained on human preference data. Compare models side by side on
            TruthfulQA or your own prompts.
          </p>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6 text-xs">
            <div className="bg-white/5 border border-white/10 rounded-lg px-3 py-2">
              <p className="font-medium text-white">Toxicity</p>
              <p className="text-indigo-200/70 mt-0.5">Detoxify model — safety of language</p>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-lg px-3 py-2">
              <p className="font-medium text-white">Instruction</p>
              <p className="text-indigo-200/70 mt-0.5">Did it follow the prompt format</p>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-lg px-3 py-2">
              <p className="font-medium text-white">Factuality</p>
              <p className="text-indigo-200/70 mt-0.5">Gemini-as-judge accuracy check</p>
            </div>
            <div className="bg-white/5 border border-white/10 rounded-lg px-3 py-2">
              <p className="font-medium text-white">Reward</p>
              <p className="text-indigo-200/70 mt-0.5">DeBERTa trained on HH-RLHF</p>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={() => setMode('single')}
              className={`text-sm px-4 py-2 rounded-lg font-medium transition ${
                mode === 'single'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-indigo-100 bg-white/5 border border-white/10 hover:bg-white/10'
              }`}
            >
              Submit job
            </button>
            <button
              onClick={() => setMode('compare')}
              className={`text-sm px-4 py-2 rounded-lg font-medium transition ${
                mode === 'compare'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-indigo-100 bg-white/5 border border-white/10 hover:bg-white/10'
              }`}
            >
              Compare models
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8 space-y-6">
        {mode === 'single' ? (
          <>
            <JobSubmit onJobCreated={addJob} />
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-1">
                <JobList jobs={jobs} activeJobId={activeJobId} onSelect={setActiveJobId} />
              </div>
              <div className="lg:col-span-2">
                {activeJobId ? (
                  <ResultsView jobId={activeJobId} />
                ) : (
                  <div className="bg-white border border-slate-200 rounded-xl p-8 text-center">
                    <p className="text-sm text-slate-400">
                      Submit a job or select one from recent jobs to see results
                    </p>
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <CompareView />
        )}
      </main>
    </div>
  )
}

export default App

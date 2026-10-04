import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'

import { JobProgress } from '../components/analyze/JobProgress'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'
import { ReportView } from '../components/report/ReportView'
import { ApiError, getJob, getResult } from '../lib/api'
import type { JobState, PredictionReport } from '../lib/types'

const POLL_MS = 1500

/** Live analysis: poll the job (INTEGRATION.md 5.2), then show the report. */
export function AnalysisPage() {
  const jobId = useParams().jobId ?? ''
  const [job, setJob] = useState<JobState | null>(null)
  const [report, setReport] = useState<PredictionReport | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    async function poll() {
      try {
        const state = await getJob(jobId)
        if (cancelled) return
        setJob(state)
        if (state.status === 'done') setReport(await getResult(jobId))
        else if (state.status === 'failed') setError(state.error ?? 'The analysis failed.')
        else timer = setTimeout(poll, POLL_MS)
      } catch (caught) {
        if (!cancelled) setError(caught instanceof ApiError && caught.status === 404 ? 'This analysis no longer exists.' : 'Lost contact with the analysis service.')
      }
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [jobId])

  return (
    <div className="flex min-h-dvh flex-col">
      <SiteNav />
      <main id="main" className="flex-1">
        {error ? (
          <div className="mx-auto max-w-2xl px-6 py-24">
            <p className="eyebrow">Analysis failed</p>
            <h1 className="mt-4 font-serif text-3xl text-ink">We could not finish this genome.</h1>
            <p className="mt-3 text-ink-2">{error}</p>
            <Link to="/analyze" className="mt-8 inline-block bg-ink px-5 py-2.5 text-sm font-medium text-paper hover:bg-ink-2">Try another file</Link>
          </div>
        ) : report ? (
          <ReportView report={report} source={{ kind: 'live', jobId }} />
        ) : (
          <JobProgress status={job?.status ?? 'queued'} sampleId={job?.sample_id ?? '…'} />
        )}
      </main>
      <SiteFooter />
    </div>
  )
}

import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router'

import { JobProgress } from '../components/analyze/JobProgress'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'
import { ReportView } from '../components/report/ReportView'
import { findSample, type SampleGenome } from '../samples'

/** Seconds the demo spends on the progress screen before showing the report. */
const DEMO_SECONDS = 3.2
/** Typical step times (JobProgress) end at 16 s; this plays them inside DEMO_SECONDS. */
const TIME_SCALE = 0.18

/**
 * Demo mode: no pipeline runs. The progress screen plays for a few seconds, then the genome's report
 * (stored in src/samples, produced by the same pipeline) is shown.
 */
export function DemoAnalysisPage() {
  const sample = findSample(useParams().key)
  return (
    <div className="flex min-h-dvh flex-col">
      <SiteNav />
      <main id="main" className="flex-1">
        {!sample ? (
          <div className="mx-auto max-w-2xl px-6 py-24">
            <h1 className="font-serif text-3xl text-ink">Demo genome not found</h1>
            <Link to="/analyze" className="mt-6 inline-block text-ink underline underline-offset-4">Back to Analyze</Link>
          </div>
        ) : (
          // Keyed so picking another genome restarts the run with fresh state.
          <DemoRun key={sample.key} sample={sample} />
        )}
      </main>
      <SiteFooter />
    </div>
  )
}

function DemoRun({ sample }: { sample: SampleGenome }) {
  const [ready, setReady] = useState(false)
  useEffect(() => {
    const timer = setTimeout(() => setReady(true), DEMO_SECONDS * 1000)
    return () => clearTimeout(timer)
  }, [])

  if (!ready) {
    return (
      <JobProgress status="running" sampleId={sample.report.sample_id} timeScale={TIME_SCALE} />
    )
  }
  return (
    <ReportView
      report={sample.report}
      source={{
        kind: 'sample',
        label: 'Demo result',
        note: `${sample.title} · ${sample.profile}.`,
      }}
    />
  )
}

import { Link, useParams } from 'react-router'

import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'
import { ReportView } from '../components/report/ReportView'
import { findSample } from '../samples'

/** Output for a precomputed sample genome. */
export function SampleReportPage() {
  const sample = findSample(useParams().key)
  return (
    <div className="flex min-h-dvh flex-col">
      <SiteNav />
      <main id="main" className="flex-1">
        {sample ? (
          <ReportView
            report={sample.report}
            source={{
              kind: 'sample',
              note: `${sample.title} · ${sample.profile}. A held-out test genome, precomputed with the same pipeline that runs on an upload.`,
            }}
          />
        ) : (
          <div className="mx-auto max-w-2xl px-6 py-24">
            <h1 className="font-serif text-3xl text-ink">Sample not found</h1>
            <Link to="/analyze" className="mt-6 inline-block text-ink underline underline-offset-4">Back to Analyze</Link>
          </div>
        )}
      </main>
      <SiteFooter />
    </div>
  )
}

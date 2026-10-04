import { useState } from 'react'
import { useNavigate } from 'react-router'

import { UploadPanel } from '../components/analyze/UploadPanel'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'
import { ApiError, LIVE_ANALYSIS, submitGenome } from '../lib/api'
import { BRAND_NAME } from '../lib/brand'

/** Input: upload one genome (live), or run a demo genome. */
export function AnalyzePage() {
  const navigate = useNavigate()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function analyze(file: File, sampleId: string) {
    setError(null)
    if (!LIVE_ANALYSIS) {
      // Never pretend: without the analysis service there is no prediction to show.
      setError('Live analysis is not connected in demo mode. Choose a demo genome below to see a result.')
      return
    }
    setBusy(true)
    try {
      const { job_id } = await submitGenome(file, sampleId || undefined)
      navigate(`/analysis/${job_id}`)
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'The upload failed. Please try again.')
      setBusy(false)
    }
  }

  return (
    <div className="flex min-h-dvh flex-col">
      <SiteNav />
      <main id="main" className="mx-auto w-full max-w-6xl flex-1 px-4 pb-20 pt-12 sm:px-6 sm:pt-16">
        <p className="eyebrow">Analyze</p>
        <h1 className="mt-4 max-w-3xl font-serif text-[2.6rem] font-normal leading-[1.08] tracking-[-0.02em] text-ink sm:text-[3.4rem]">
          One genome in. A drug-by-drug answer out.
        </h1>
        <p className="mt-5 max-w-2xl text-[17px] leading-relaxed text-ink-2">
          Upload an assembled bacterial genome. {BRAND_NAME} checks it, identifies the species, finds known resistance genes,
          and predicts the MIC for every supported antibiotic.
        </p>
        <div className="mt-12 max-w-2xl">
          <UploadPanel busy={busy} error={error} onSubmit={(file, sampleId) => void analyze(file, sampleId)} />
        </div>
      </main>
      <SiteFooter />
    </div>
  )
}

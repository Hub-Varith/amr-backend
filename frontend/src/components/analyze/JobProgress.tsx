import { motion } from 'motion/react'
import { useEffect, useState } from 'react'

import type { JobStatus } from '../../lib/types'

/** The pipeline steps (docs/PREDICTION_PIPELINE.md) and their typical share of a ~10-40 s run. */
const STEPS = [
  { label: 'Upload received', detail: 'Genome stored for this job', seconds: 0 },
  { label: 'Assembly quality check', detail: 'Contigs, genome size, N50', seconds: 1 },
  { label: 'Species identification', detail: 'Mash distance to the five references', seconds: 2 },
  { label: 'Resistance genes and mutations', detail: 'AMRFinderPlus scan of every contig', seconds: 4 },
  { label: 'MIC prediction', detail: 'One estimate and 90% range per antibiotic', seconds: 14 },
  { label: 'Calls and report', detail: 'Breakpoints, overrides, ranking', seconds: 16 },
]

type Props = {
  status: JobStatus
  sampleId: string
  /** Multiplies the typical step times; demo mode plays the whole timeline in a few seconds. */
  timeScale?: number
  /** Replaces the default footnote. */
  note?: string
}

/**
 * The API reports queued / running / done, not individual steps, so the active step is estimated from
 * elapsed time and never shown as finished before the job is. The note under the list says so.
 */
export function JobProgress({ status, sampleId, timeScale = 1, note }: Props) {
  const [elapsed, setElapsed] = useState(0)
  useEffect(() => {
    const started = Date.now()
    const timer = setInterval(() => setElapsed((Date.now() - started) / 1000), 250)
    return () => clearInterval(timer)
  }, [])

  const done = status === 'done'
  const reached = done ? STEPS.length : Math.max(1, STEPS.filter((step) => step.seconds * timeScale <= elapsed).length)
  const current = done ? -1 : Math.min(reached, STEPS.length) - 1

  return (
    <section aria-live="polite" className="mx-auto w-full max-w-2xl px-4 py-16 sm:px-6 sm:py-24">
      <p className="eyebrow">Analysis in progress</p>
      <h1 className="mt-4 font-serif text-[2.4rem] font-normal leading-tight text-ink">
        Reading <span className="font-mono text-[0.7em]">{sampleId}</span>
      </h1>
      <p className="mt-3 text-[15px] text-ink-2">
        {status === 'queued' ? 'Waiting for a free worker…' : timeScale < 1 ? `Elapsed ${elapsed.toFixed(1)} s` : `Elapsed ${Math.floor(elapsed)} s · typically under a minute`}
      </p>

      <ol className="mt-10 border-t border-rule">
        {STEPS.map((step, index) => {
          const finished = done || index < current
          const active = index === current
          return (
            <li key={step.label} className="flex items-start gap-4 border-b border-rule py-4">
              <span className="mt-1 flex h-4 w-4 shrink-0 items-center justify-center" aria-hidden="true">
                {finished ? (
                  <svg viewBox="0 0 16 16" className="h-4 w-4 text-active" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m3.5 8.5 3 3 6-7" /></svg>
                ) : active ? (
                  <motion.span className="h-2.5 w-2.5 rounded-full bg-ink" animate={{ opacity: [1, 0.25, 1] }} transition={{ duration: 1.2, repeat: Infinity }} />
                ) : (
                  <span className="h-2 w-2 rounded-full border border-rule-strong" />
                )}
              </span>
              <span className="min-w-0">
                <span className={`block text-[15px] ${finished || active ? 'text-ink' : 'text-ink-3'}`}>{step.label}</span>
                <span className="block text-sm text-ink-3">{step.detail}</span>
              </span>
            </li>
          )
        })}
      </ol>
      <p className="mt-4 text-xs text-ink-3">{note ?? 'Step progress is estimated from typical run times; the report appears when the job finishes.'}</p>
    </section>
  )
}

import { motion } from 'motion/react'
import type { ReactNode } from 'react'
import { Link } from 'react-router'

import {
  CALL_LABELS,
  SPECIES_NAMES,
  confidenceLabel,
  drugName,
  formatBand,
  formatMic,
  markerLabel,
  percent,
} from '../../lib/format'
import type { Call, DrugPrediction, PredictionReport } from '../../lib/types'
import { CallBadge, CallDot, ChanceMeter } from './CallBadge'
import { MicAxisLegend, MicBandAxis } from './MicBandAxis'

export type ReportSource = { kind: 'sample'; note: string; label?: string } | { kind: 'live'; jobId: string }

const CALL_ORDER: Call[] = ['likely_active', 'uncertain', 'likely_inactive']
const GROUP_TITLES: Record<Call, string> = {
  likely_active: 'Likely active',
  uncertain: 'Uncertain — wait for lab',
  likely_inactive: 'Likely inactive',
}

/** The full stage 12 report: header, the ranked answer, every drug, the evidence, and (live analyses) the disclaimer. */
export function ReportView({ report, source }: { report: PredictionReport; source: ReportSource }) {
  const byDrug = new Map(report.predictions.map((prediction) => [prediction.drug, prediction]))
  const ranked = report.ranked_active.map((drug) => byDrug.get(drug)).filter((p): p is DrugPrediction => Boolean(p))
  const counts = Object.fromEntries(CALL_ORDER.map((call) => [call, report.predictions.filter((p) => p.call === call).length])) as Record<Call, number>
  const markers = [...new Set(report.predictions.flatMap((p) => p.reasons))].sort((a, b) => a.localeCompare(b))

  function downloadJson() {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = Object.assign(document.createElement('a'), { href: url, download: `${report.sample_id}.dnagen.json` })
    link.click()
    URL.revokeObjectURL(url)
  }

  return (
    <article className="report mx-auto w-full max-w-6xl px-4 pb-20 sm:px-6">
      <ReportHeader report={report} source={source} onDownload={downloadJson} />

      {!report.species ? (
        <Notice tone="inactive" title="Species not covered">
          This genome is not close enough to any of the five supported species (<i>E. coli</i>, <i>K. pneumoniae</i>,{' '}
          <i>S. aureus</i>, <i>P. aeruginosa</i>, <i>A. baumannii</i>), so no predictions were made. Use standard laboratory
          testing.
        </Notice>
      ) : (
        <>
          {!report.qc_pass && (
            <Notice tone="uncertain" title="Assembly quality check failed">
              The genome is fragmented or an unexpected size for its species. Predictions are shown but may be unreliable.
            </Notice>
          )}
          {!report.in_range && (
            <Notice tone="uncertain" title="Low confidence: unlike the training genomes">
              This strain is far from every genome the model learned from. Treat every call below as low confidence.
            </Notice>
          )}

          <Summary counts={counts} total={report.predictions.length} />
          <RankedActive ranked={ranked} uncertain={counts.uncertain} />
          <DrugTable predictions={report.predictions} />
          {markers.length > 0 && <Markers markers={markers} />}
        </>
      )}

      {source.kind === 'live' && (
        <aside role="note" className="mt-12 border border-ink/80 bg-ink px-6 py-5 text-paper print:border-ink print:bg-transparent print:text-ink">
          <p className="font-mono text-[11px] uppercase tracking-[0.14em] text-paper/70 print:text-ink-3">Not prescribing advice</p>
          <p className="mt-2 max-w-4xl text-[15px] leading-relaxed">{report.disclaimer}</p>
        </aside>
      )}

      <div className="mt-8 flex flex-wrap items-center gap-3 print:hidden">
        <Link to="/analyze" className="bg-ink px-5 py-2.5 text-sm font-medium text-paper transition-colors hover:bg-ink-2">
          Analyze another genome
        </Link>
        <button type="button" onClick={() => window.print()}
          className="border border-rule-strong px-5 py-2.5 text-sm text-ink transition-colors hover:border-ink-3 hover:bg-raised">
          Print or save as PDF
        </button>
        <button type="button" onClick={downloadJson}
          className="border border-rule-strong px-5 py-2.5 text-sm text-ink transition-colors hover:border-ink-3 hover:bg-raised">
          Download JSON
        </button>
      </div>
    </article>
  )
}

function ReportHeader({ report, source, onDownload }: { report: PredictionReport; source: ReportSource; onDownload: () => void }) {
  const runName = report.model_version.split('/').pop() ?? report.model_version
  const facts: [string, ReactNode][] = [
    ['Sample', <span key="sample" className="font-mono">{report.sample_id}</span>],
    ['Assembly QC', report.qc_pass ? <span key="qc" className="text-active">Pass</span> : <span key="qc" className="text-inactive">Fail</span>],
    ['Training similarity', report.nearest_training_distance === null
      ? <span key="range" className="text-ink-2">{report.in_range ? 'Species match' : 'Out of range'}</span>
      : <span key="range">{report.in_range ? 'In range' : 'Out of range'} · {report.nearest_training_distance.toFixed(3)}</span>],
    ['Model', <span key="model" className="font-mono">{runName} · {report.run_id}</span>],
    ['Breakpoints', 'CLSI, bloodstream'],
  ]
  return (
    <header className="border-b border-rule pb-8 pt-10 sm:pt-14">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="eyebrow">Susceptibility report · {source.kind === 'sample' ? source.label ?? 'Sample result' : 'Live analysis'}</p>
        <button type="button" onClick={onDownload} className="hidden font-mono text-xs text-ink-3 underline-offset-4 hover:text-ink hover:underline sm:block print:hidden">
          {report.sample_id}.json ↓
        </button>
      </div>
      <h1 className="mt-4 font-serif text-[2.6rem] font-normal italic leading-[1.05] tracking-[-0.02em] text-ink sm:text-[3.4rem]">
        {report.species ? SPECIES_NAMES[report.species] : 'Species not identified'}
      </h1>
      {source.kind === 'sample' && <p className="mt-3 max-w-2xl text-[15px] text-ink-2">{source.note}</p>}
      <dl className="mt-8 grid grid-cols-2 gap-x-8 gap-y-4 sm:grid-cols-3 lg:grid-cols-5">
        {facts.map(([label, value]) => (
          <div key={label} className="min-w-0 border-l border-rule pl-3">
            <dt className="font-mono text-[10px] uppercase tracking-[0.12em] text-ink-3">{label}</dt>
            <dd className="mt-1 truncate text-sm text-ink">{value}</dd>
          </div>
        ))}
      </dl>
    </header>
  )
}

function Notice({ tone, title, children }: { tone: 'uncertain' | 'inactive'; title: string; children: ReactNode }) {
  const color = tone === 'inactive' ? 'border-inactive text-inactive' : 'border-uncertain text-uncertain'
  return (
    <div role="status" className={`mt-8 border-l-2 bg-raised px-5 py-4 ${color}`}>
      <p className="text-sm font-medium">{title}</p>
      <p className="mt-1 text-sm leading-relaxed text-ink-2">{children}</p>
    </div>
  )
}

function Summary({ counts, total }: { counts: Record<Call, number>; total: number }) {
  return (
    <section aria-label="Summary" className="mt-10">
      <div className="flex h-2 w-full overflow-hidden bg-rule" aria-hidden="true">
        {CALL_ORDER.map((call) => (
          <motion.span key={call} initial={{ width: 0 }} animate={{ width: `${(counts[call] / Math.max(total, 1)) * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className={call === 'likely_active' ? 'bg-active' : call === 'uncertain' ? 'bg-uncertain/60' : 'bg-inactive'} />
        ))}
      </div>
      <dl className="mt-4 grid grid-cols-3 gap-4">
        {CALL_ORDER.map((call) => (
          <div key={call}>
            <dt className="flex items-center gap-2 text-sm text-ink-2"><CallDot call={call} />{CALL_LABELS[call]}</dt>
            <dd className="mt-1 font-serif text-3xl text-ink">{counts[call]}<span className="ml-1 text-base text-ink-3">/ {total}</span></dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function RankedActive({ ranked, uncertain }: { ranked: DrugPrediction[]; uncertain: number }) {
  return (
    <section aria-labelledby="ranked" className="mt-14">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b border-ink pb-3">
        <h2 id="ranked" className="font-serif text-[1.9rem] font-normal text-ink">Antibiotics likely to work</h2>
        <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-3">Narrowest spectrum first</p>
      </div>
      {ranked.length === 0 ? (
        <div className="border-b border-rule py-8">
          <p className="font-serif text-xl text-ink">No antibiotic met the bar for “likely active”.</p>
          <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-2">
            The genome carries resistance that rules out the drugs below{uncertain > 0 && `, and ${uncertain} more are too close to call`}.
            Wait for laboratory susceptibility testing before choosing therapy.
          </p>
        </div>
      ) : (
        <ol>
          {ranked.map((prediction, index) => (
            <motion.li key={prediction.drug} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, delay: 0.1 + index * 0.04 }}
              className="grid grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 border-b border-rule py-4 sm:grid-cols-[2.5rem_minmax(0,1.4fr)_8rem_11rem_10rem]">
              <span className="font-mono text-sm text-ink-3">{String(index + 1).padStart(2, '0')}</span>
              <span className="text-[17px] font-medium text-ink">{drugName(prediction.drug)}</span>
              <span className="text-right font-mono text-sm text-ink sm:text-left">
                {formatMic(prediction.pred_mic)} <span className="text-ink-3">mg/L</span>
              </span>
              <span className="col-start-2 text-sm text-ink-2 sm:col-start-auto">
                {prediction.margin_steps ? `${prediction.margin_steps} step${prediction.margin_steps === 1 ? '' : 's'} below S` : 'Near the S breakpoint'}
                <span className="text-ink-3"> (≤{prediction.s_breakpoint})</span>
              </span>
              <span className="col-start-2 text-sm text-ink-2 sm:col-start-auto sm:text-right">
                {prediction.p_active !== null ? `${percent(prediction.p_active)} chance active` : 'Whole range below S'}
              </span>
            </motion.li>
          ))}
        </ol>
      )}
    </section>
  )
}

function DrugTable({ predictions }: { predictions: DrugPrediction[] }) {
  return (
    <section aria-labelledby="all-drugs" className="mt-16">
      <div className="flex flex-wrap items-end justify-between gap-4 border-b border-ink pb-3">
        <h2 id="all-drugs" className="font-serif text-[1.9rem] font-normal text-ink">Every antibiotic</h2>
        <div className="hidden w-[min(42%,26rem)] md:block"><MicAxisLegend /></div>
      </div>
      {CALL_ORDER.map((call) => {
        const rows = predictions.filter((p) => p.call === call)
        if (rows.length === 0) return null
        return (
          <div key={call} className="mt-8">
            <h3 className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.12em] text-ink-2">
              <CallDot call={call} />{GROUP_TITLES[call]}<span className="text-ink-3">· {rows.length}</span>
            </h3>
            <ul className="mt-2 border-t border-rule">
              {rows.map((prediction) => <DrugRow key={prediction.drug} prediction={prediction} />)}
            </ul>
          </div>
        )
      })}
    </section>
  )
}

function DrugRow({ prediction: p }: { prediction: DrugPrediction }) {
  const overrideNote = p.override === 'natural_resistance' ? 'Natural resistance' : p.override === 'strong_marker' ? 'Resistance marker' : null
  return (
    <li className="grid grid-cols-1 gap-x-6 gap-y-2 border-b border-rule py-4 break-inside-avoid md:grid-cols-[minmax(0,13rem)_minmax(0,1fr)_minmax(0,26rem)] md:items-center">
      <div className="min-w-0">
        <p className="font-medium text-ink">{drugName(p.drug)}</p>
        <p className="mt-0.5 font-mono text-xs text-ink-3">
          {p.s_breakpoint !== null ? `S ≤${p.s_breakpoint} · R >${p.r_breakpoint}` : 'No CLSI breakpoint'}
        </p>
      </div>
      <div className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1">
        <CallBadge call={p.call} />
        {overrideNote && <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-inactive">{overrideNote}</span>}
        {p.pred_mic !== null && (
          <span className="font-mono text-sm text-ink">
            {formatMic(p.pred_mic)} <span className="text-ink-3">mg/L · {formatBand(p.band_low, p.band_high)}</span>
          </span>
        )}
        {p.p_active !== null && p.confidence_level && (
          <span className="flex items-center gap-2 text-xs text-ink-2" title={`Confidence level: ${confidenceLabel(p.confidence_level)}`}>
            <ChanceMeter value={p.p_active} call={p.call} />
            {percent(p.p_active)} chance active
          </span>
        )}
        {p.reasons.length > 0 && (
          <span className="flex flex-wrap gap-1.5">
            {p.reasons.map((reason) => (
              <span key={reason} className="border border-rule-strong bg-raised px-1.5 py-px font-mono text-[11px] text-ink-2">{markerLabel(reason)}</span>
            ))}
          </span>
        )}
      </div>
      <div className="min-w-0">
        {p.pred_mic !== null ? (
          <MicBandAxis mic={p.pred_mic} low={p.band_low} high={p.band_high} sBreakpoint={p.s_breakpoint} rBreakpoint={p.r_breakpoint}
            call={p.call} label={`${drugName(p.drug)}: predicted MIC ${formatMic(p.pred_mic)} mg/L, range ${formatBand(p.band_low, p.band_high)}`} />
        ) : (
          <p className="text-sm text-ink-3">Not modelled: the species is intrinsically resistant.</p>
        )}
      </div>
    </li>
  )
}

function Markers({ markers }: { markers: string[] }) {
  const points = markers.filter((marker) => /\s[A-Z*-]?\d/.test(marker))
  const genes = markers.filter((marker) => !points.includes(marker))
  return (
    <section aria-labelledby="markers" className="mt-16">
      <div className="border-b border-ink pb-3">
        <h2 id="markers" className="font-serif text-[1.9rem] font-normal text-ink">Resistance markers found</h2>
      </div>
      <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-ink-2">
        Known resistance genes and mutations detected in the assembly by AMRFinderPlus. They are the evidence behind the calls
        above; the model reads them, not the raw DNA.
      </p>
      <dl className="mt-6 grid gap-6 sm:grid-cols-2">
        {[['Genes', genes], ['Mutations', points]].map(([label, items]) => (items as string[]).length > 0 && (
          <div key={label as string}>
            <dt className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-3">{label as string}</dt>
            <dd className="mt-2 flex flex-wrap gap-2">
              {(items as string[]).map((marker) => (
                <span key={marker} className="border border-rule-strong px-2 py-1 font-mono text-xs text-ink">{markerLabel(marker)}</span>
              ))}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

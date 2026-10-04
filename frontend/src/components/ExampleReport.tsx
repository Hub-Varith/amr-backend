import { Link } from 'react-router'

import { CALL_SHORT, SPECIES_NAMES, drugName, formatMic } from '../lib/format'
import type { Call } from '../lib/types'
import { findSample } from '../samples'

const SHOWN = ['meropenem', 'ceftriaxone', 'ciprofloxacin', 'gentamicin']
const DOT: Record<Call, string> = { likely_active: 'bg-[#7fb8a2]', uncertain: 'bg-[#d6b46a]', likely_inactive: 'bg-[#e08a7e]' }

/** Hero card: four real rows from the KPC Klebsiella sample report. */
export function ExampleReport() {
  const sample = findSample('kpneu-kpc')!
  const rows = SHOWN.map((drug) => sample.report.predictions.find((p) => p.drug === drug)!).filter(Boolean)
  return (
    <section className="specimen" aria-label="Sample report preview">
      <div className="specimen-top"><span>Sample result · {sample.report.sample_id}</span><span>Held-out test genome</span></div>
      <h2>{sample.report.species ? SPECIES_NAMES[sample.report.species] : ''}</h2>
      <p className="specimen-sub">{sample.profile}</p>
      <div className="sequence" aria-hidden="true">ATGTCACTGTATCGCCGTCTAGTTCTG</div>
      <div className="report-row"><small>ANTIBIOTIC</small><small>MIC · mg/L · CALL</small></div>
      {rows.map((p) => (
        <div key={p.drug} className="report-row">
          <span>{drugName(p.drug)}{p.reasons[0] && <small className="ml-2 font-mono">{p.reasons[0]}</small>}</span>
          <span className="flex items-center gap-2 font-mono text-[13px]">
            {formatMic(p.pred_mic)}
            <span className={`h-2 w-2 rounded-full ${DOT[p.call]}`} aria-hidden="true" />
            <span className="sr-only">{CALL_SHORT[p.call]}</span>
          </span>
        </div>
      ))}
      <p className="report-note">
        Real output for a genome the model never saw in training.{' '}
        <Link to="/report/sample/kpneu-kpc" className="underline underline-offset-4 hover:text-[#f5f2e9]">Open the full report →</Link>
      </p>
    </section>
  )
}

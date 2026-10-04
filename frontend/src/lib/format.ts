import type { Call, ConfidenceLevel, SpeciesKey } from './types'

export const SPECIES_NAMES: Record<SpeciesKey, string> = {
  ECOLI: 'Escherichia coli',
  KPNEU: 'Klebsiella pneumoniae',
  SAUR: 'Staphylococcus aureus',
  PAER: 'Pseudomonas aeruginosa',
  ABAU: 'Acinetobacter baumannii',
}

const DRUG_NAMES: Record<string, string> = {
  'amoxicillin-clavulanic-acid': 'Amoxicillin–clavulanate',
  'ampicillin-sulbactam': 'Ampicillin–sulbactam',
  'piperacillin-tazobactam': 'Piperacillin–tazobactam',
  'trimethoprim-sulfamethoxazole': 'Trimethoprim–sulfamethoxazole',
  'ceftazidime-avibactam': 'Ceftazidime–avibactam',
  'ceftolozane-tazobactam': 'Ceftolozane–tazobactam',
  'polymyxin-b': 'Polymyxin B',
}

/** `piperacillin-tazobactam` -> `Piperacillin–tazobactam`. */
export function drugName(drug: string): string {
  return DRUG_NAMES[drug] ?? drug.charAt(0).toUpperCase() + drug.slice(1)
}

export const CALL_LABELS: Record<Call, string> = {
  likely_active: 'Likely active',
  uncertain: 'Uncertain — wait for lab',
  likely_inactive: 'Likely inactive',
}

export const CALL_SHORT: Record<Call, string> = {
  likely_active: 'Likely active',
  uncertain: 'Uncertain',
  likely_inactive: 'Likely inactive',
}

export function confidenceLabel(level: ConfidenceLevel): string {
  const text = level.replace(/_/g, ' ')
  return text.charAt(0).toUpperCase() + text.slice(1)
}

/** Panel range shown on screen. Model values past it are extrapolations and read as a limit. */
export const MIC_FLOOR = 2 ** -6 // 0.015625
export const MIC_CEILING = 2 ** 9 // 512

/** Doubling-step values the way lab panels print them: 0.0625 -> 0.06, 0.125 -> 0.125. */
export function panelValue(mic: number): string {
  if (mic >= 1) return String(Math.round(mic))
  const rounded = Number(mic.toPrecision(mic < 0.1 ? 1 : 3))
  return String(rounded).replace(/^0\./, '0.')
}

/** One MIC for display, clamped to the panel range: 1e-5 -> "≤0.016", 4096 -> ">512". */
export function formatMic(mic: number | null): string {
  if (mic === null) return '—'
  if (mic <= MIC_FLOOR) return `≤${panelValue(MIC_FLOOR)}`
  if (mic > MIC_CEILING) return `>${panelValue(MIC_CEILING)}`
  return panelValue(mic)
}

export function formatBand(low: number | null, high: number | null): string {
  if (low === null || high === null) return ''
  return `${formatMic(low)} – ${formatMic(high)}`
}

export function percent(value: number): string {
  if (value >= 0.995) return '>99%'
  if (value <= 0.005) return '<1%'
  return `${Math.round(value * 100)}%`
}

/** `ompK35_E42RfsTer47` and friends are already readable; keep symbols as AMRFinderPlus writes them. */
export function markerLabel(marker: string): string {
  return marker.replace(/_/g, ' ')
}

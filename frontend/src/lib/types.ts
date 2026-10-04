/** Report shapes from GET /v1/jobs/{id}/result (DATA_CONTRACT.md stage 12, api/schemas/*.py). */

export type SpeciesKey = 'ECOLI' | 'KPNEU' | 'SAUR' | 'PAER' | 'ABAU'
export type Call = 'likely_active' | 'uncertain' | 'likely_inactive'
export type Override = 'natural_resistance' | 'strong_marker'
export type ConfidenceLevel =
  | 'very_likely_inactive'
  | 'probably_inactive'
  | 'leans_inactive'
  | 'unclear'
  | 'leans_active'
  | 'probably_active'
  | 'very_likely_active'

export type DrugPrediction = {
  drug: string
  /** mg/L, already rounded up to a doubling step. Null together with the band when an override skips the model. */
  pred_mic: number | null
  band_low: number | null
  band_high: number | null
  s_breakpoint: number | null
  r_breakpoint: number | null
  /** Calibrated P(MIC <= S breakpoint). */
  p_active: number | null
  confidence_level: ConfidenceLevel | null
  call: Call
  margin_steps: number | null
  reasons: string[]
  override: Override | null
}

export type PredictionReport = {
  sample_id: string
  species: SpeciesKey | null
  qc_pass: boolean
  nearest_training_distance: number | null
  in_range: boolean
  predictions: DrugPrediction[]
  /** Drug names, best first. The frontend never re-ranks. */
  ranked_active: string[]
  model_version: string
  run_id: string
  disclaimer: string
}

export type JobStatus = 'queued' | 'running' | 'done' | 'failed'

export type JobState = {
  job_id: string
  sample_id: string
  status: JobStatus
  created_at: string
  updated_at: string
  error: string | null
}

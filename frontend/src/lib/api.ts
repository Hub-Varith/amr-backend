import type { JobState, PredictionReport } from './types'

/**
 * Live analysis talks to the backend API (INTEGRATION.md section 1). It is off until
 * VITE_API_BASE_URL is set, e.g. in frontend/.env.local:  VITE_API_BASE_URL=http://127.0.0.1:8000
 */
const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export const LIVE_ANALYSIS = API_BASE !== ''

/** Same rules as the backend upload check (api/services/upload_validator.py). */
export const ACCEPTED_EXTENSIONS = ['.fasta', '.fa', '.fna', '.fasta.gz', '.fa.gz', '.fna.gz']
export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024

export class ApiError extends Error {
  readonly status?: number

  constructor(message: string, status?: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new ApiError('The analysis service could not be reached.')
  }
  if (!response.ok) {
    // Errors are application/problem+json with a readable `detail`.
    const problem = (await response.json().catch(() => null)) as { detail?: string } | null
    throw new ApiError(problem?.detail ?? `Request failed (${response.status}).`, response.status)
  }
  return (await response.json()) as T
}

export function submitGenome(file: File, sampleId?: string): Promise<{ job_id: string }> {
  const body = new FormData()
  body.append('file', file)
  if (sampleId) body.append('sample_id', sampleId)
  return request('/v1/predict', { method: 'POST', body })
}

export function getJob(jobId: string): Promise<JobState> {
  return request(`/v1/jobs/${encodeURIComponent(jobId)}`)
}

export function getResult(jobId: string): Promise<PredictionReport> {
  return request(`/v1/jobs/${encodeURIComponent(jobId)}/result`)
}

/** Quick checks before upload, so obvious mistakes never leave the browser. */
export async function checkFile(file: File): Promise<string | null> {
  const name = file.name.toLowerCase()
  if (!ACCEPTED_EXTENSIONS.some((extension) => name.endsWith(extension))) {
    return 'Choose a FASTA file: .fasta, .fa or .fna, optionally gzipped (.gz).'
  }
  if (file.size === 0) return 'This file is empty.'
  if (file.size > MAX_UPLOAD_BYTES) return 'This file is larger than 50 MB. Upload one assembled genome.'
  if (!name.endsWith('.gz')) {
    const head = (await file.slice(0, 4096).text()).trimStart()
    if (!head.startsWith('>')) return 'This does not look like FASTA: the first line should start with “>”.'
  }
  return null
}

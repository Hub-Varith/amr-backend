import { useRef, useState, type DragEvent, type FormEvent } from 'react'

import { LIVE_ANALYSIS, checkFile } from '../../lib/api'
import { DemoPicker } from './DemoPicker'

type Props = {
  busy: boolean
  error: string | null
  onSubmit: (file: File, sampleId: string) => void
}

/** File System Access picker (Chromium). Its `id` makes the browser reopen the folder last used with it. */
type OpenFilePicker = (options: {
  id?: string
  multiple?: boolean
  types?: { description: string; accept: Record<string, string[]> }[]
}) => Promise<{ getFile: () => Promise<File> }[]>

/**
 * Remembered picker folder. Point it at data/raw/genome_database/genome_database_testing once and the
 * browser opens there every time after (browsers cannot open an arbitrary path on the first use).
 */
const PICKER_ID = 'genome-database-testing'

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** Drag-and-drop or browse for one assembled genome, with the same checks the API applies. */
export function UploadPanel({ busy, error, onSubmit }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [file, setFile] = useState<File | null>(null)
  const [sampleId, setSampleId] = useState('')
  const [problem, setProblem] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)

  async function choose(candidate: File | undefined) {
    setProblem(null)
    if (!candidate) return
    const issue = await checkFile(candidate)
    if (issue) {
      setFile(null)
      setProblem(issue)
      return
    }
    setFile(candidate)
  }

  async function browse() {
    const picker = (window as unknown as { showOpenFilePicker?: OpenFilePicker }).showOpenFilePicker
    if (!picker) {
      input.current?.click()
      return
    }
    try {
      const [handle] = await picker({
        id: PICKER_ID,
        multiple: false,
        types: [{ description: 'Genome FASTA', accept: { 'application/octet-stream': ['.fasta', '.fa', '.fna', '.gz'] } }],
      })
      await choose(await handle.getFile())
    } catch (caught) {
      // Closing the picker is not an error.
      if (!(caught instanceof DOMException && caught.name === 'AbortError')) input.current?.click()
    }
  }

  function drop(event: DragEvent) {
    event.preventDefault()
    setDragging(false)
    void choose(event.dataTransfer.files[0])
  }

  function submit(event: FormEvent) {
    event.preventDefault()
    if (file) onSubmit(file, sampleId.trim())
  }

  const message = problem ?? error
  return (
    <form onSubmit={submit} className="border border-rule-strong bg-paper p-6 sm:p-8">
      <div className="flex items-center justify-between gap-4">
        <p className="eyebrow">Upload</p>
        <span className={`font-mono text-[10px] uppercase tracking-[0.12em] ${LIVE_ANALYSIS ? 'text-active' : 'text-ink-3'}`}>
          {LIVE_ANALYSIS ? '● Analysis service connected' : '○ Demo mode'}
        </span>
      </div>
      <h2 className="mt-3 font-serif text-[1.7rem] font-normal text-ink">Your bacterial genome</h2>

      <div
        onDragOver={(event) => { event.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={drop}
        className={`mt-6 flex min-h-48 flex-col items-center justify-center border border-dashed px-6 py-8 text-center transition-colors ${
          dragging ? 'border-ink bg-raised' : file ? 'border-active/60 bg-active/5' : 'border-rule-strong'
        }`}
      >
        {file ? (
          <>
            <p className="font-mono text-sm text-ink">{file.name}</p>
            <p className="mt-1 text-sm text-ink-2">{formatBytes(file.size)} · ready to analyze</p>
            <button type="button" onClick={() => { setFile(null); if (input.current) input.current.value = '' }}
              className="mt-4 text-sm text-ink-2 underline underline-offset-4 hover:text-ink">
              Choose a different file
            </button>
          </>
        ) : (
          <>
            <svg viewBox="0 0 24 24" className="h-7 w-7 text-ink-3" fill="none" stroke="currentColor" strokeWidth="1.2" aria-hidden="true">
              <path d="M12 16V4m0 0-4 4m4-4 4 4M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3" />
            </svg>
            <p className="mt-3 text-[15px] text-ink">Drop a FASTA file here</p>
            <p className="mt-1 text-sm text-ink-2">
              or{' '}
              <button type="button" onClick={() => void browse()} className="text-ink underline underline-offset-4">
                browse your computer
              </button>
            </p>
            <p className="mt-4 font-mono text-[11px] text-ink-3">.fasta · .fa · .fna · gzipped .gz · up to 50 MB</p>
          </>
        )}
        <input ref={input} id="genome" type="file" className="sr-only" aria-label="Genome FASTA file"
          accept=".fasta,.fa,.fna,.gz" onChange={(event) => void choose(event.target.files?.[0])} />
      </div>

      <label htmlFor="sample-id" className="mt-6 block font-mono text-[11px] uppercase tracking-[0.12em] text-ink-3">
        Sample ID <span className="normal-case tracking-normal">(optional)</span>
      </label>
      <input id="sample-id" value={sampleId} onChange={(event) => setSampleId(event.target.value)} maxLength={128}
        placeholder={file ? file.name.replace(/(\.(fasta|fa|fna))?(\.gz)?$/i, '') : 'e.g. BC-0142'}
        className="mt-2 w-full border-b border-rule-strong bg-transparent py-2 font-mono text-sm text-ink placeholder:text-ink-3 focus:border-ink focus:outline-none" />

      <div aria-live="polite">
        {message && <p className="mt-5 border-l-2 border-inactive bg-raised px-4 py-3 text-sm text-ink">{message}</p>}
      </div>

      <button type="submit" disabled={!file || busy}
        className="mt-6 w-full bg-ink px-5 py-3 text-sm font-medium text-paper transition-colors hover:bg-ink-2 disabled:cursor-not-allowed disabled:bg-rule-strong">
        {busy ? 'Uploading…' : 'Analyze genome'}
      </button>
      <p className="mt-4 text-xs leading-relaxed text-ink-3">
        One assembled genome (contigs). Supported species: <i>E. coli</i>, <i>K. pneumoniae</i>, <i>S. aureus</i>,{' '}
        <i>P. aeruginosa</i>, <i>A. baumannii</i>. Results typically take under a minute.
      </p>
      <div id="demo" className="scroll-mt-28"><DemoPicker /></div>
    </form>
  )
}

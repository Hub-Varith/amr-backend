import { useState } from 'react'
import { ExampleReport } from '../components/ExampleReport'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'

export function AnalyzePlaceholderPage() {
  const [demo, setDemo] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState(false)
  const [busy, setBusy] = useState(false)
  async function inspect(file: File | undefined) {
    setDemo(false); setMessage(''); setError(false)
    if (!file) return
    if (file.size > 20 * 1024 * 1024) { setError(true); setMessage('Choose an uncompressed FASTA file smaller than 20 MB.'); return }
    setBusy(true)
    try {
      const text = (await file.text()).trim()
      const lines = text.split(/\r?\n/).filter(line => line.trim())
      const sequences = lines.filter(line => !line.startsWith('>'))
      if (!text.startsWith('>') || !sequences.length || sequences.some(line => !/^[ACGTRYSWKMBDHVN]+$/i.test(line.trim()))) throw new Error('Use a DNA FASTA file with > sequence headers and valid nucleotide letters.')
      const records = lines.filter(line => line.startsWith('>')).length
      const bases = sequences.reduce((sum, line) => sum + line.trim().length, 0)
      setMessage(`${file.name}: ${records.toLocaleString()} sequence records and ${bases.toLocaleString()} bases. Format check passed. No prediction has been run; the analysis service is not connected to this page yet.`)
    } catch (e) { setError(true); setMessage(e instanceof Error ? e.message : 'Unable to read this file. Please try another FASTA.') }
    finally { setBusy(false) }
  }
  return <div className="flex min-h-dvh flex-col"><SiteNav /><main className="workspace flex-1">
    <p className="eyebrow">Research workspace / 01</p><h1>Start with a genome.</h1>
    <p className="max-w-xl text-ink-2">Check a FASTA file locally, or explore an example report to see the workflow. Your file stays in this browser.</p>
    <div className="workspace-grid"><section className="upload-panel"><p className="eyebrow">01 / Input</p><h2 className="mt-4">Your bacterial genome</h2>
      <label htmlFor="genome" className="text-sm text-ink-2">Uncompressed DNA FASTA · .fasta, .fa, .fna · up to 20 MB</label>
      <input id="genome" className="file-input" type="file" accept=".fa,.fna,.fasta" disabled={busy} onChange={e => void inspect(e.target.files?.[0])} />
      <div aria-live="polite">{busy && <p>Checking file format…</p>}{message && <p className={`message ${error ? 'error' : ''}`}>{message}</p>}</div>
      <div className="mt-6 border-t border-rule pt-6"><p className="mb-4 text-sm text-ink-2">Just taking a look? No file needed.</p><button className="action" onClick={() => setDemo(true)}>View example report <span aria-hidden="true">↗</span></button></div>
    </section><div aria-live="polite">{demo ? <ExampleReport /> : <section className="demo-empty"><p className="eyebrow">02 / Report preview</p><h2 className="mt-4">Make the results readable.</h2><p className="text-ink-2">An antibiotic-by-antibiotic view of MIC estimates, with units and limitations in plain sight.</p><p className="mt-6 text-sm text-ink-2">Open the example to explore the presentation. Example values are fictional.</p></section>}</div></div>
  </main><SiteFooter /></div>
}

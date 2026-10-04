import { AnimatePresence, motion } from 'motion/react'
import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'
import { useNavigate } from 'react-router'

import { SPECIES_NAMES } from '../../lib/format'
import { DEMO_GENOMES, type SampleGenome } from '../../samples'

/** 1, 2, ... among the demo genomes of the same species, so two isolates of one species stay distinct. */
function isolateNumber(genome: SampleGenome): number {
  const sameSpecies = DEMO_GENOMES.filter((other) => other.report.species === genome.report.species)
  return sameSpecies.indexOf(genome) + 1
}

/** Demo mode: pick one of the eight demo genomes and open its report after a short load. */
export function DemoPicker() {
  const navigate = useNavigate()
  const listId = useId()
  const root = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [selected, setSelected] = useState<SampleGenome>(DEMO_GENOMES[0])
  const [highlight, setHighlight] = useState(0)

  useEffect(() => {
    if (!open) return
    const close = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  function choose(index: number) {
    setSelected(DEMO_GENOMES[index])
    setOpen(false)
  }

  function onKey(event: KeyboardEvent) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault()
      if (!open) {
        setHighlight(DEMO_GENOMES.indexOf(selected))
        setOpen(true)
        return
      }
      const step = event.key === 'ArrowDown' ? 1 : -1
      setHighlight((current) => (current + step + DEMO_GENOMES.length) % DEMO_GENOMES.length)
    } else if ((event.key === 'Enter' || event.key === ' ') && open) {
      event.preventDefault()
      choose(highlight)
    } else if (event.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div className="mt-8 border-t border-rule pt-6">
      <div className="flex items-center justify-between gap-3">
        <p className="font-mono text-[11px] uppercase tracking-[0.12em] text-ink-3">Demo mode</p>
        <p className="text-xs text-ink-3">8 test genomes</p>
      </div>
      <p className="mt-2 text-sm text-ink-2">No file? Run one of our demo genomes.</p>

      <div ref={root} className="relative mt-4">
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-controls={listId}
          aria-haspopup="listbox"
          aria-label="Demo genome"
          aria-activedescendant={open ? `${listId}-${highlight}` : undefined}
          onClick={() => { setHighlight(DEMO_GENOMES.indexOf(selected)); setOpen((value) => !value) }}
          onKeyDown={onKey}
          className={`flex w-full items-center gap-4 border bg-paper px-4 py-3 text-left transition-colors hover:border-ink-3 ${
            open ? 'border-ink' : 'border-rule-strong'
          }`}
        >
          <GenomeSummary genome={selected} />
          <svg viewBox="0 0 16 16" className={`h-4 w-4 shrink-0 text-ink-3 transition-transform ${open ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m4 6 4 4 4-4" /></svg>
        </button>

        <AnimatePresence>
          {open && (
            <motion.ul
              id={listId}
              role="listbox"
              aria-label="Demo genomes"
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.15 }}
              className="absolute inset-x-0 top-full z-20 mt-1 max-h-[22rem] overflow-y-auto border border-ink bg-paper shadow-[6px_6px_0_var(--color-raised)]"
            >
              {DEMO_GENOMES.map((genome, index) => (
                <li
                  key={genome.key}
                  id={`${listId}-${index}`}
                  role="option"
                  aria-selected={genome.key === selected.key}
                  onMouseEnter={() => setHighlight(index)}
                  onMouseDown={(event) => { event.preventDefault(); choose(index) }}
                  className={`flex cursor-pointer items-center gap-4 border-b border-rule px-4 py-3 last:border-b-0 ${
                    index === highlight ? 'bg-raised' : ''
                  }`}
                >
                  <GenomeSummary genome={genome} />
                  <span className={`h-4 w-4 shrink-0 text-active ${genome.key === selected.key ? '' : 'invisible'}`} aria-hidden="true">
                    <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="m3.5 8.5 3 3 6-7" /></svg>
                  </span>
                </li>
              ))}
            </motion.ul>
          )}
        </AnimatePresence>
      </div>

      <button
        type="button"
        onClick={() => navigate(`/demo/${selected.key}`)}
        className="mt-4 w-full border border-ink px-5 py-3 text-sm font-medium text-ink transition-colors hover:bg-ink hover:text-paper"
      >
        Run demo analysis
      </button>
    </div>
  )
}

function GenomeSummary({ genome }: { genome: SampleGenome }) {
  const species = genome.report.species ? SPECIES_NAMES[genome.report.species] : ''
  return (
    <span className="flex min-w-0 flex-1 items-baseline gap-3">
      <span className="truncate font-serif text-[15px] italic text-ink">{species}</span>
      <span className="shrink-0 text-xs text-ink-3">Isolate {isolateNumber(genome)}</span>
    </span>
  )
}

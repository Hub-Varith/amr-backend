import type { PredictionReport } from '../lib/types'
import abauOxa23 from './abau-oxa23.json'
import abauSusceptible from './abau-susceptible.json'
import demoAbau1 from './demo-abau-1.json'
import demoAbau2 from './demo-abau-2.json'
import demoEcoli1 from './demo-ecoli-1.json'
import demoEcoli2 from './demo-ecoli-2.json'
import demoKpneu1 from './demo-kpneu-1.json'
import demoKpneu2 from './demo-kpneu-2.json'
import demoSaur1 from './demo-saur-1.json'
import demoSaur2 from './demo-saur-2.json'
import ecoliEsbl from './ecoli-esbl.json'
import ecoliSusceptible from './ecoli-susceptible.json'
import kpneuKpc from './kpneu-kpc.json'
import kpneuSusceptible from './kpneu-susceptible.json'
import saurMrsa from './saur-mrsa.json'
import saurMssa from './saur-mssa.json'

/**
 * Real pipeline outputs, not fictional values: each genome is from the held-out test split of release
 * 2026-10-04-hackathon-all5, run through QC, Mash, AMRFinderPlus 4.2.7 and model all5_run1 on 2026-10-03.
 * They are precomputed, so the page labels them as sample results.
 */
export type SampleGenome = {
  key: string
  title: string
  profile: string
  tone: 'resistant' | 'susceptible'
  report: PredictionReport
}

/** Contrasting pairs: a resistant and a susceptible strain per species. */
export const SAMPLES: SampleGenome[] = [
  { key: 'kpneu-kpc', title: 'Carbapenem-resistant Klebsiella', profile: 'blaKPC-2 carbapenemase, blaSHV-12, porin loss', tone: 'resistant', report: kpneuKpc as PredictionReport },
  { key: 'kpneu-susceptible', title: 'Klebsiella, broadly susceptible', profile: 'Only intrinsic genes (blaSHV-11, oqxAB)', tone: 'susceptible', report: kpneuSusceptible as PredictionReport },
  { key: 'ecoli-esbl', title: 'ESBL E. coli', profile: 'blaCTX-M-15 ESBL, quinolone mutations', tone: 'resistant', report: ecoliEsbl as PredictionReport },
  { key: 'ecoli-susceptible', title: 'E. coli, broadly susceptible', profile: 'Only the intrinsic blaEC', tone: 'susceptible', report: ecoliSusceptible as PredictionReport },
  { key: 'saur-mrsa', title: 'MRSA', profile: 'mecA, erm(C)', tone: 'resistant', report: saurMrsa as PredictionReport },
  { key: 'saur-mssa', title: 'MSSA', profile: 'No mecA', tone: 'susceptible', report: saurMssa as PredictionReport },
  { key: 'abau-oxa23', title: 'Carbapenem-resistant Acinetobacter', profile: 'blaOXA-23 carbapenemase, armA', tone: 'resistant', report: abauOxa23 as PredictionReport },
  { key: 'abau-susceptible', title: 'Acinetobacter, broadly susceptible', profile: 'Only intrinsic blaADC and OXA-51-like', tone: 'susceptible', report: abauSusceptible as PredictionReport },
]

/**
 * Demo mode: the eight test-split genomes with the most antibiotics likely to work and no wrong call
 * against their lab results (data/raw/genome_database/genomes.csv), two per species.
 */
export const DEMO_GENOMES: SampleGenome[] = [
  { key: 'demo-ecoli-1', title: 'E. coli isolate', profile: 'Only the intrinsic blaEC', tone: 'susceptible', report: demoEcoli1 as PredictionReport },
  { key: 'demo-ecoli-2', title: 'E. coli isolate', profile: 'Only the intrinsic blaEC', tone: 'susceptible', report: demoEcoli2 as PredictionReport },
  { key: 'demo-kpneu-1', title: 'Klebsiella isolate', profile: 'Only intrinsic genes (blaSHV-11, oqxAB)', tone: 'susceptible', report: demoKpneu1 as PredictionReport },
  { key: 'demo-kpneu-2', title: 'Klebsiella isolate', profile: 'Intrinsic blaSHV-1, tet(D)', tone: 'susceptible', report: demoKpneu2 as PredictionReport },
  { key: 'demo-abau-1', title: 'Acinetobacter isolate', profile: 'Only intrinsic blaADC and OXA-51-like', tone: 'susceptible', report: demoAbau1 as PredictionReport },
  { key: 'demo-abau-2', title: 'Acinetobacter isolate', profile: 'Only intrinsic blaADC and OXA-51-like', tone: 'susceptible', report: demoAbau2 as PredictionReport },
  { key: 'demo-saur-1', title: 'S. aureus isolate (MSSA)', profile: 'No mecA; msr(A)', tone: 'susceptible', report: demoSaur1 as PredictionReport },
  { key: 'demo-saur-2', title: 'S. aureus isolate (MSSA)', profile: 'No mecA', tone: 'susceptible', report: demoSaur2 as PredictionReport },
]

export function findSample(key: string | undefined): SampleGenome | undefined {
  return [...SAMPLES, ...DEMO_GENOMES].find((sample) => sample.key === key)
}

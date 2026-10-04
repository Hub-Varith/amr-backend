import { ButtonLink } from '../Button'
import { RuledList, type RuledItem } from './RuledList'
import { Section } from './Section'

const STEPS: RuledItem[] = [
  {
    key: 'upload',
    title: 'Upload a genome',
    body: 'One assembled genome as FASTA. We check assembly quality and confirm the species.',
    aside: 'QC, Mash',
  },
  {
    key: 'markers',
    title: 'Read resistance markers',
    body: 'Known resistance genes and mutations, plus DNA patterns that no curated database lists yet.',
    aside: 'AMRFinderPlus, unitigs',
  },
  {
    key: 'predict',
    title: 'Predict the MIC',
    body: 'One model per species and drug, with a 90% band showing how sure the prediction is.',
    aside: 'XGBoost AFT, conformal',
  },
  {
    key: 'call',
    title: 'Call and rank',
    body: 'Each band is compared to the clinical breakpoint. Drugs likely to work are ranked, narrowest spectrum first.',
    aside: 'EUCAST / CLSI breakpoints',
  },
]

export function HowItWorks() {
  return (
    <Section
      id="how"
      lead
      number="01"
      label="How it works"
      title="From one bacterial genome to a ranked list of antibiotics, in four steps."
      intro="Breakpoint predicts each antibiotic's MIC: the lowest concentration that stops the bacteria growing in a lab test, in mg/L. Choosing a dose stays with the clinician."
    >
      <RuledList items={STEPS} animateOnLoad />

      <div className="mt-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <dl className="text-[15px]">
          <dt className="font-mono text-xs uppercase tracking-[0.12em] text-ink-3">Species covered</dt>
          <dd className="mt-2 max-w-xl leading-relaxed text-ink-2">
            <i className="text-ink">Klebsiella pneumoniae</i> (first models), <i>Escherichia coli</i>,{' '}
            <i>Staphylococcus aureus</i>, <i>Pseudomonas aeruginosa</i>, <i>Acinetobacter baumannii</i>
          </dd>
        </dl>
        <div className="shrink-0">
          <ButtonLink to="/analyze">Analyze a genome</ButtonLink>
        </div>
      </div>
    </Section>
  )
}

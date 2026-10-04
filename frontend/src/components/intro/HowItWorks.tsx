import { ButtonLink } from '../Button'
import { RuledList, type RuledItem } from './RuledList'
import { Section } from './Section'
import { BRAND_NAME } from '../../lib/brand'

const STEPS: RuledItem[] = [
  {
    key: 'upload',
    title: 'Upload a genome',
    body: 'Start with an assembled bacterial genome in FASTA format. Quality checks and species identification come before prediction.',
    aside: 'QC, Mash',
  },
  {
    key: 'markers',
    title: 'Read resistance markers',
    body: 'Convert known resistance genes and mutations into the features used by the prediction model.',
    aside: 'Genes and mutations',
  },
  {
    key: 'predict',
    title: 'Predict the MIC',
    body: 'Estimate the minimum inhibitory concentration for each supported antibiotic. Evaluate predictions against laboratory measurements.',
    aside: 'MIC estimation',
  },
  {
    key: 'call',
    title: 'Review the evidence',
    body: 'Review the evidence and limitations for each antibiotic. Clinical interpretation requires validated breakpoints and laboratory confirmation.',
    aside: 'EUCAST / CLSI breakpoints',
  },
]

export function HowItWorks() {
  return (
    <Section
      id="how"
      number="01"
      label="How it works"
      title="A sequence becomes a clearer picture."
      intro={`${BRAND_NAME} predicts each antibiotic's MIC: the lowest concentration that stops the bacteria growing in a lab test, in mg/L. Choosing a dose stays with the clinician.`}
    >
      <RuledList items={STEPS} animateOnLoad />

      <div className="mt-10 flex flex-col gap-6 sm:flex-row sm:items-end sm:justify-between">
        <dl className="text-[15px]">
          <dt className="font-mono text-xs uppercase tracking-[0.12em] text-ink-3">Species covered</dt>
          <dd className="mt-2 max-w-xl leading-relaxed text-ink-2">
            <i className="text-ink">Klebsiella pneumoniae</i> , <i>Escherichia coli</i>,{' '}
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

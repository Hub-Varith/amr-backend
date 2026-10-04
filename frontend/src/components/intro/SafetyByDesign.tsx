import { RuledList, type RuledItem } from './RuledList'
import { Section } from './Section'

const PRINCIPLES: RuledItem[] = [
  { key: 'lab', title: 'Grounded in lab measurements', body: 'Model predictions are evaluated against observed laboratory results. A completed run is not proof of accuracy.' },
  { key: 'uncertainty', title: 'Uncertainty stays visible', body: 'A value reported as “greater than” remains a limit, not an invented exact MIC. Confidence estimates require their own validation.' },
  { key: 'scope', title: 'Clear about what is supported', body: 'Results depend on the species, antibiotic, and available resistance features. Performance varies across those groups.' },
  { key: 'confirm', title: 'The laboratory remains essential', body: 'This research prototype does not prescribe treatment. Confirm susceptibility with standard laboratory testing.' },
]

/** The design choices that lean away from very major errors. */
export function SafetyByDesign() {
  return (
    <Section
      id="safety"
      number="01"
      label="Safety"
      title="Evidence first. Limits in view."
      intro="A useful research tool should be as clear about what it cannot establish as what it predicts."
    >
      <RuledList items={PRINCIPLES} />
    </Section>
  )
}

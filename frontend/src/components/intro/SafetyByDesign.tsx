import { RuledList, type RuledItem } from './RuledList'
import { Section } from './Section'

const PRINCIPLES: RuledItem[] = [
  { key: 'round-up', title: 'Errs high', body: 'Predicted MICs round up to the next doubling step. Overstating resistance is the safer mistake.' },
  { key: 'unsure', title: 'Says when it is unsure', body: 'If the band crosses a breakpoint, the call is uncertain and the lab result decides.' },
  { key: 'override', title: 'Known biology overrides', body: 'Natural resistance, and markers such as a carbapenemase for meropenem, force likely inactive.' },
  { key: 'range', title: 'Flags unfamiliar strains', body: 'A genome far from the training data gets every call marked low confidence.' },
]

/** The design choices that lean away from very major errors. */
export function SafetyByDesign() {
  return (
    <Section
      id="safety"
      number="02"
      label="Safety"
      title="Built to fail safe."
      intro="The error that harms patients is calling a drug active when the lab says resistant. Each of these choices makes that error less likely."
    >
      <RuledList items={PRINCIPLES} />
    </Section>
  )
}

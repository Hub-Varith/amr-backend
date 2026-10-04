import { ButtonLink } from '../components/Button'
import { Section } from '../components/intro/Section'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'

/** Stand-in until the upload page is built. */
export function AnalyzePlaceholderPage() {
  return (
    <div className="flex min-h-dvh flex-col">
      <SiteNav />
      <main className="flex-1">
        <Section
          lead
          number="--"
          label="Analyze"
          title="The upload page is being built."
          intro="Soon you will be able to drop in a FASTA file here and follow the analysis as it runs."
        >
          <ButtonLink to="/" variant="secondary">
            Back to the overview
          </ButtonLink>
        </Section>
      </main>
      <SiteFooter />
    </div>
  )
}

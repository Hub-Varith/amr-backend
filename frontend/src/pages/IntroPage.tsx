import { Link } from 'react-router'

import { ButtonLink } from '../components/Button'
import { ExampleReport } from '../components/ExampleReport'
import { SafetyByDesign } from '../components/intro/SafetyByDesign'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'

/** Landing: the product statement, a real sample report, and the safety principles. */
export function IntroPage() {
  return (
    <div className="min-h-dvh overflow-x-clip">
      <SiteNav />
      <a href="#main" className="skip-link">Skip to content</a>
      <main id="main">
        <section className="hero">
          <div><p className="eyebrow">Breakpoint / Genomic resistance research</p>
            <h1>Read the genome.<br /><em>Understand<br />the resistance.</em></h1>
            <p className="hero-copy">Explore how bacterial DNA can inform antibiotic susceptibility. From resistance markers to a clear, drug-by-drug MIC report.</p>
            <div className="hero-actions"><ButtonLink to="/analyze">Analyze a genome <span aria-hidden="true">→</span></ButtonLink><Link className="text-link" to={{ pathname: '/analyze', hash: '#demo' }}>Try a demo genome</Link></div>
            <p className="eyebrow" style={{fontSize:10}}>Research prototype · Not clinically validated</p>
          </div>
          <ExampleReport />
        </section>
        <div className="species-strip" aria-label="Species in the research dataset"><span className="eyebrow">Five species. One research workflow.</span><i>E. coli</i><i>K. pneumoniae</i><i>S. aureus</i><i>P. aeruginosa</i><i>A. baumannii</i></div>
        <SafetyByDesign />
      </main>
      <SiteFooter />
    </div>
  )
}

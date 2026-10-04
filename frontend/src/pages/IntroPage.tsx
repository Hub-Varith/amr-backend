import { HowItWorks } from '../components/intro/HowItWorks'
import { SafetyByDesign } from '../components/intro/SafetyByDesign'
import { SiteFooter } from '../components/intro/SiteFooter'
import { SiteNav } from '../components/intro/SiteNav'

/** Demo order: how it works first, then why it is safe. */
export function IntroPage() {
  return (
    <div className="min-h-dvh overflow-x-clip">
      <SiteNav />
      <main>
        <HowItWorks />
        <SafetyByDesign />
      </main>
      <SiteFooter />
    </div>
  )
}

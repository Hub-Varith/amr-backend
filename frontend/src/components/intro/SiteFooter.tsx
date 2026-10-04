import { BRAND_NAME, DISCLAIMER } from '../../lib/brand'
import { Logo } from '../brand/Logo'

export function SiteFooter() {
  return (
    <footer className="border-t border-rule">
      <div className="mx-auto grid max-w-6xl gap-x-12 gap-y-6 px-4 py-10 sm:px-6 lg:grid-cols-[11rem_minmax(0,1fr)]">
        <Logo />
        <div className="max-w-3xl space-y-3 text-[13px] leading-relaxed text-ink-3">
          <p role="note">
            <span className="text-ink-2">Not prescribing advice. </span>
            {DISCLAIMER}
          </p>
          <p>{BRAND_NAME} is a research prototype built at MHacks. It has not been clinically validated.</p>
        </div>
      </div>
    </footer>
  )
}

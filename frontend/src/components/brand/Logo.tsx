import { BRAND_NAME } from '../../lib/brand'

/** Mark: an MIC band on an axis, crossed by a breakpoint line. */
export function LogoMark({ className = 'h-5 w-5' }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <line x1="1" y1="10" x2="19" y2="10" className="stroke-ink-3" strokeWidth="1" />
      <rect x="3" y="7" width="8" height="6" className="fill-ink" />
      <line x1="14" y1="2" x2="14" y2="18" className="stroke-ink" strokeWidth="1.5" />
    </svg>
  )
}

export function Logo() {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="font-serif text-[19px] leading-none text-ink">{BRAND_NAME}</span>
    </span>
  )
}

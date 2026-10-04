import type { ReactNode } from 'react'
import { Link, type To } from 'react-router'

type ButtonLinkProps = {
  to: To
  children: ReactNode
  variant?: 'primary' | 'secondary'
}

const VARIANTS = {
  primary: 'bg-ink text-paper hover:bg-ink-2',
  secondary: 'border border-rule-strong text-ink hover:border-ink-3 hover:bg-raised',
}

/** Rectangular link-button. Primary is off-white on dark; secondary is outlined. */
export function ButtonLink({ to, children, variant = 'primary' }: ButtonLinkProps) {
  return (
    <Link to={to} className={`inline-block px-5 py-2.5 text-sm font-medium transition-colors ${VARIANTS[variant]}`}>
      {children}
    </Link>
  )
}

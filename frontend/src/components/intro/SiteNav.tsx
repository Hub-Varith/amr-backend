import { Link, useLocation } from 'react-router'

import { Logo } from '../brand/Logo'

type NavItem = { label: string; pathname: string; hash?: string }

const LINKS: NavItem[] = [
  { label: 'How it works', pathname: '/', hash: '#how' },
  { label: 'Safety', pathname: '/', hash: '#safety' },
]

export function SiteNav() {
  const location = useLocation()

  return (
    <header className="sticky top-0 z-50 border-b border-rule bg-paper/95 backdrop-blur-sm">
      <nav className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-6 px-4 sm:px-6">
        <Link to="/" aria-label="Breakpoint home">
          <Logo />
        </Link>
        <div className="hidden items-center gap-7 md:flex">
          {LINKS.map((link) => {
            const isCurrentPage = !link.hash && location.pathname === link.pathname
            return (
              <Link
                key={link.label}
                to={{ pathname: link.pathname, hash: link.hash }}
                aria-current={isCurrentPage ? 'page' : undefined}
                className={`text-sm transition-colors hover:text-ink ${
                  isCurrentPage ? 'text-ink underline decoration-ink-3 underline-offset-[6px]' : 'text-ink-2'
                }`}
              >
                {link.label}
              </Link>
            )
          })}
        </div>
        <Link
          to="/analyze"
          className="border border-rule-strong px-3.5 py-1.5 text-sm text-ink transition-colors hover:border-ink-3 hover:bg-raised"
        >
          Analyze a genome
        </Link>
      </nav>
    </header>
  )
}

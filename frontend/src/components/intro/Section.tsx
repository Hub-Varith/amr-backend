import { motion } from 'motion/react'
import type { ReactNode } from 'react'

type SectionProps = {
  id?: string
  /** Two-digit section number shown in the margin, e.g. "01". */
  number: string
  label: string
  title: ReactNode
  intro?: ReactNode
  /** The first section on the page: gets the h1 and fades in on load instead of on scroll. */
  lead?: boolean
  children?: ReactNode
}

/**
 * A page section laid out like a printed report: the number and label sit in a narrow margin column,
 * the heading and content in the main column. Sections are separated by a hairline rule.
 */
export function Section({ id, number, label, title, intro, lead = false, children }: SectionProps) {
  const Heading = lead ? 'h1' : 'h2'
  const reveal = lead
    ? { animate: { opacity: 1 } }
    : { whileInView: { opacity: 1 }, viewport: { once: true, margin: '-80px' } }

  return (
    <section id={id} className="scroll-mt-16 border-t border-rule first:border-t-0">
      <motion.div
        initial={{ opacity: 0 }}
        {...reveal}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className={`mx-auto grid max-w-6xl gap-x-12 gap-y-4 px-4 sm:px-6 lg:grid-cols-[11rem_minmax(0,1fr)] ${
          lead ? 'pb-16 pt-14 sm:pb-20 sm:pt-20' : 'py-14 sm:py-16'
        }`}
      >
        <p className="flex gap-3 pt-1.5 font-mono text-xs uppercase tracking-[0.12em] text-ink-3 lg:flex-col lg:gap-1">
          <span>{number}</span>
          <span className="text-ink-2">{label}</span>
        </p>
        <div className="min-w-0">
          <Heading
            className={`max-w-3xl text-balance font-serif font-normal tracking-[-0.015em] text-ink ${
              lead ? 'text-[2.4rem] leading-[1.08] sm:text-5xl lg:text-[3.5rem]' : 'text-[1.9rem] leading-[1.15] sm:text-[2.25rem]'
            }`}
          >
            {title}
          </Heading>
          {intro && (
            <p className={`mt-5 max-w-2xl text-pretty text-ink-2 ${lead ? 'text-[17px] leading-relaxed sm:text-lg' : 'leading-relaxed'}`}>
              {intro}
            </p>
          )}
          {children && <div className="mt-10">{children}</div>}
        </div>
      </motion.div>
    </section>
  )
}

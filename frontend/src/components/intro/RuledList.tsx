import { motion } from 'motion/react'
import type { ReactNode } from 'react'

export type RuledItem = {
  key: string
  marker?: string
  title: ReactNode
  body: ReactNode
  aside?: ReactNode
}

/** Items as rows between hairline rules: optional marker, title, body, and a quiet aside on the right. */
export function RuledList({ items, animateOnLoad = false }: { items: RuledItem[]; animateOnLoad?: boolean }) {
  return (
    <ol className="border-t border-rule">
      {items.map((item, index) => (
        <motion.li
          key={item.key}
          initial={{ opacity: 0 }}
          {...(animateOnLoad ? { animate: { opacity: 1 } } : { whileInView: { opacity: 1 }, viewport: { once: true } })}
          transition={{ duration: 0.5, delay: (animateOnLoad ? 0.2 : 0) + index * 0.08 }}
          className="grid grid-cols-[2.25rem_minmax(0,1fr)] gap-x-4 gap-y-1 border-b border-rule py-4 sm:grid-cols-[2.25rem_13rem_minmax(0,1fr)] lg:grid-cols-[2.25rem_13rem_minmax(0,1fr)_auto]"
        >
          <span className="pt-0.5 font-mono text-xs text-ink-3">{item.marker ?? String(index + 1).padStart(2, '0')}</span>
          <h3 className="text-[15px] font-medium text-ink">{item.title}</h3>
          <p className="col-start-2 text-[15px] leading-relaxed text-ink-2 sm:col-start-auto">{item.body}</p>
          {item.aside && (
            <p className="col-start-2 pt-0.5 font-mono text-xs text-ink-3 sm:col-start-3 lg:col-start-auto lg:text-right">
              {item.aside}
            </p>
          )}
        </motion.li>
      ))}
    </ol>
  )
}

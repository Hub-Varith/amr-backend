import { CALL_SHORT } from '../../lib/format'
import type { Call } from '../../lib/types'

const STYLES: Record<Call, string> = {
  likely_active: 'border-active/40 bg-active/10 text-active',
  uncertain: 'border-uncertain/40 bg-uncertain/10 text-uncertain',
  likely_inactive: 'border-inactive/40 bg-inactive/10 text-inactive',
}
const DOTS: Record<Call, string> = {
  likely_active: 'bg-active',
  uncertain: 'bg-uncertain',
  likely_inactive: 'bg-inactive',
}

export function CallBadge({ call }: { call: Call }) {
  return (
    <span className={`inline-flex items-center gap-1.5 whitespace-nowrap border px-2 py-0.5 text-xs font-medium ${STYLES[call]}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${DOTS[call]}`} aria-hidden="true" />
      {CALL_SHORT[call]}
    </span>
  )
}

export function CallDot({ call }: { call: Call }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${DOTS[call]}`} aria-hidden="true" />
}

/** Calibrated chance the drug works, as a thin meter with the number beside it. */
export function ChanceMeter({ value, call }: { value: number; call: Call }) {
  return (
    <span className="flex items-center gap-2">
      <span className="relative h-1 w-14 bg-rule" aria-hidden="true">
        <span className={`absolute inset-y-0 left-0 ${DOTS[call]}`} style={{ width: `${Math.round(value * 100)}%` }} />
      </span>
    </span>
  )
}

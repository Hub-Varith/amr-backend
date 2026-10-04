import { MIC_CEILING, MIC_FLOOR } from '../../lib/format'
import type { Call } from '../../lib/types'

const LOW = Math.log2(MIC_FLOOR)
const HIGH = Math.log2(MIC_CEILING)

/** Position on the axis, 0-100, for an MIC in mg/L. Values past the panel range sit on its edge. */
function x(mic: number): number {
  const step = Math.min(Math.max(Math.log2(mic), LOW), HIGH)
  return ((step - LOW) / (HIGH - LOW)) * 100
}

const BAND_CLASS: Record<Call, string> = {
  likely_active: 'fill-active/25 stroke-active',
  uncertain: 'fill-uncertain/20 stroke-uncertain',
  likely_inactive: 'fill-inactive/20 stroke-inactive',
}
const DOT_CLASS: Record<Call, string> = {
  likely_active: 'fill-active',
  uncertain: 'fill-uncertain',
  likely_inactive: 'fill-inactive',
}

type Props = {
  mic: number | null
  low: number | null
  high: number | null
  sBreakpoint: number | null
  rBreakpoint: number | null
  call: Call
  label: string
}

/** The 90% MIC band on a log2 axis (0.016 to 512 mg/L), with the S and R breakpoints as vertical rules. */
export function MicBandAxis({ mic, low, high, sBreakpoint, rBreakpoint, call, label }: Props) {
  const ticks = Array.from({ length: HIGH - LOW + 1 }, (_, index) => LOW + index)
  return (
    <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="h-6 w-full overflow-visible" role="img" aria-label={label}>
      {sBreakpoint !== null && <rect x={0} y={4} width={x(sBreakpoint)} height={16} className="fill-active/[0.06]" />}
      {rBreakpoint !== null && <rect x={x(rBreakpoint)} y={4} width={100 - x(rBreakpoint)} height={16} className="fill-inactive/[0.06]" />}
      <line x1={0} y1={12} x2={100} y2={12} className="stroke-rule-strong" strokeWidth={0.4} vectorEffect="non-scaling-stroke" />
      {ticks.map((tick) => (
        <line key={tick} x1={x(2 ** tick)} y1={10.5} x2={x(2 ** tick)} y2={13.5} className="stroke-rule-strong" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      ))}
      {low !== null && high !== null && (
        <rect x={x(low)} y={7} width={Math.max(x(high) - x(low), 0.8)} height={10} rx={1}
          className={BAND_CLASS[call]} strokeWidth={1} vectorEffect="non-scaling-stroke" />
      )}
      {sBreakpoint !== null && (
        <line x1={x(sBreakpoint)} y1={2} x2={x(sBreakpoint)} y2={22} className="stroke-active" strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      )}
      {rBreakpoint !== null && rBreakpoint !== sBreakpoint && (
        <line x1={x(rBreakpoint)} y1={2} x2={x(rBreakpoint)} y2={22} className="stroke-inactive" strokeWidth={1.5}
          strokeDasharray="2 2" vectorEffect="non-scaling-stroke" />
      )}
      {mic !== null && <circle cx={x(mic)} cy={12} r={1.6} className={DOT_CLASS[call]} />}
    </svg>
  )
}

/** Axis labels and the key, shown once above the drug table. */
export function MicAxisLegend() {
  const labels = [MIC_FLOOR, 0.125, 1, 8, 64, MIC_CEILING]
  return (
    <div className="space-y-2">
      <div className="relative h-4 font-mono text-[10px] text-ink-3">
        {labels.map((value, index) => (
          <span key={value} className="absolute -translate-x-1/2 whitespace-nowrap"
            style={{ left: `${x(value)}%`, transform: index === 0 ? 'none' : index === labels.length - 1 ? 'translateX(-100%)' : undefined }}>
            {value < 0.1 ? '≤0.016' : value >= 512 ? '>512' : value}
          </span>
        ))}
      </div>
      <p className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-3">
        <span><span className="mr-1.5 inline-block h-2 w-4 rounded-sm border border-ink-3 bg-ink-3/15 align-middle" />90% range</span>
        <span><span className="mr-1.5 inline-block h-3 w-px bg-active align-middle" />S breakpoint</span>
        <span><span className="mr-1.5 inline-block h-3 w-px border-l border-dashed border-inactive align-middle" />R breakpoint</span>
        <span>mg/L, doubling steps</span>
      </p>
    </div>
  )
}

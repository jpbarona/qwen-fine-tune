import { elapsed, gib, num } from '../format'
import type { Dashboard } from '../types'

export default function SystemStrip({ gpu }: { gpu: Dashboard['gpu'] }) {
  const current = gpu.current
  const utilization = gpu.history.flatMap((sample) => typeof sample.utilization === 'number' && Number.isFinite(sample.utilization) ? [sample.utilization] : [])
  const mean = utilization.length ? utilization.reduce((sum, value) => sum + value, 0) / utilization.length : null
  const idle = utilization.length ? utilization.filter((value) => value < 10).length / utilization.length * 100 : null
  const reading = (value: number | null | undefined, digits: number, suffix: string): string =>
    value === null || value === undefined ? '-' : `${num(value, digits)}${suffix}`
  const memory =
    current && (current.memory_used !== null || current.memory_total !== null)
      ? `${gib(current.memory_used)} / ${gib(current.memory_total)} GiB`
      : '-'
  return (
    <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-1 border border-white/[0.07] bg-[#0b0b0d] px-3 py-[7px] font-mono text-[11px] tabular-nums text-[#7b7b84]">
      <span className="text-[10.5px] uppercase tracking-[0.12em] text-[#5f5f68]">system</span>
      <span>
        gpu util <span className="text-[#d9d9de]">{reading(current?.utilization, 0, '%')}</span>
      </span>
      <span title="Mean utilization over the latest 120 samples recorded by the dashboard server">
        mean util <span className="text-[#d9d9de]">{reading(mean, 0, '%')}</span>
      </span>
      <span title="Share of the latest 120 recorded GPU samples with utilization below 10%">
        idle samples <span className="text-[#d9d9de]">{reading(idle, 0, '%')}</span>
      </span>
      <span>
        gpu memory <span className="text-[#d9d9de]">{memory}</span>
      </span>
      <span>
        gpu temp <span className="text-[#d9d9de]">{reading(current?.temperature, 0, ' C')}</span>
      </span>
      <span>
        samples <span className="text-[#d9d9de]">{gpu.history.length}</span>
      </span>
      <span>
        last sample{' '}
        <span className="text-[#d9d9de]">
          {gpu.age_seconds === null || gpu.age_seconds === undefined ? '-' : `${elapsed(gpu.age_seconds)} ago`}
        </span>
      </span>
      {current ? null : <span className="text-[#55555d]">no reading</span>}
    </div>
  )
}

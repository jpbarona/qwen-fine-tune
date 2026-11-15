import { useEffect, useState, type ReactNode } from 'react'
import { ACCENT, BarChart, DIM, FAINT, LineChart, MUTED, Panel, Swatch, TRACE, type ChartSeries } from './charts'
import { age, elapsed, gib, num, perStepRate, rate, whole } from './format'
import type { Dashboard, Evaluation, Run, StagePoint } from './types'

function pairs(steps: (number | null)[], values: (number | null | undefined)[] | undefined): [number, number | null][] {
  if (!values) return []
  const out: [number, number | null][] = []
  const count = Math.min(steps.length, values.length)
  for (let index = 0; index < count; index += 1) {
    const x = steps[index]
    if (x === null || x === undefined) continue
    const y = values[index]
    out.push([x, y === undefined ? null : y])
  }
  return out
}

function scaled(points: [number, number | null][], factor: number): [number, number | null][] {
  return points.map(([x, y]) => [x, y === null ? null : y * factor])
}

function stageSegments(points: StagePoint[]): ChartSeries[] {
  const segments: StagePoint[][] = []
  let current: StagePoint[] = []
  let band: string | null = null
  for (const point of points) {
    if (point.band !== band) {
      if (current.length) segments.push(current)
      current = []
      band = point.band
    }
    current.push(point)
  }
  if (current.length) segments.push(current)
  return segments.map((segment) => ({
    color: ACCENT,
    dots: true,
    points: segment.map((point) => [point.step, point.percent * 100] as [number, number]),
  }))
}

function percentTick(value: number): string {
  return `${value.toFixed(value !== 0 && Math.abs(value) < 1 ? 2 : 1)}%`
}

function Status({ status }: { status: string }) {
  const color =
    status === 'running' ? ACCENT : status === 'finished' ? '#8fae8f' : status === 'stopped' ? '#c76b62' : '#6b6b73'
  return (
    <span className="flex items-center gap-1.5 uppercase tracking-[0.12em]" style={{ color }}>
      <span className="inline-block h-[5px] w-[5px] rounded-full" style={{ background: color }} />
      {status}
    </span>
  )
}

function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="bg-[#0b0b0d] px-3 py-[7px]">
      <div className="text-[9.5px] uppercase leading-[1.5] tracking-[0.09em] text-[#5f5f68]">{label}</div>
      <div className="font-mono text-[12.5px] tabular-nums leading-[1.45] text-[#d9d9de]">{children}</div>
    </div>
  )
}

function Strip({ children }: { children: ReactNode }) {
  return (
    <div className="mt-3 grid grid-cols-2 gap-px border border-white/[0.07] bg-white/[0.07] sm:grid-cols-4 xl:grid-cols-8">
      {children}
    </div>
  )
}

function StepProgress({ run }: { run: Run }) {
  const target = run.target_steps
  const percent = target ? Math.min((run.steps_done / target) * 100, 100) : null
  return (
    <div className="mt-2.5 flex items-center gap-3">
      <div className="whitespace-nowrap font-mono text-[10.5px] tabular-nums text-[#6b6b73]">
        <span className="text-[13px] text-[#e7e7ea]">{whole(run.steps_done)}</span>
        {target ? <span className="text-[#6b6b73]"> / {whole(target)}</span> : null} steps
      </div>
      <div className="h-[3px] flex-1 bg-white/[0.06]">
        {percent !== null ? <div className="h-full" style={{ width: `${percent}%`, background: ACCENT }} /> : null}
      </div>
      {percent !== null ? (
        <div className="w-10 text-right font-mono text-[10px] tabular-nums text-[#6b6b73]">{percent.toFixed(1)}%</div>
      ) : null}
    </div>
  )
}

function EvaluationTable({ evaluations }: { evaluations: Evaluation[] }) {
  return (
    <Panel title="evaluations" className="px-0 py-0">
      <table className="w-full table-fixed border-collapse text-[11px]">
        <thead>
          <tr className="text-[9.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
            <th className="w-14 px-3 py-1.5 text-left font-normal">step</th>
            <th className="px-3 py-1.5 text-left font-normal">kind</th>
            <th className="px-3 py-1.5 text-left font-normal">stage</th>
            <th className="w-16 px-3 py-1.5 text-right font-normal">count</th>
            <th className="w-20 px-3 py-1.5 text-right font-normal">pass@1</th>
          </tr>
        </thead>
        <tbody>
          {evaluations.length === 0 ? (
            <tr>
              <td colSpan={5} className="border-t border-white/[0.06] px-3 py-3 font-mono text-[10.5px] text-[#4f4f57]">
                no evaluation recorded
              </td>
            </tr>
          ) : (
            evaluations.map((item, index) => (
              <tr key={index} className="border-t border-white/[0.06] text-[#b9b9c0]">
                <td className="px-3 py-[5px] font-mono tabular-nums text-[#8b8b94]">{item.step ?? '-'}</td>
                <td className="truncate px-3 py-[5px]">{item.kind || '-'}</td>
                <td className="truncate px-3 py-[5px] text-[#8b8b94]">{item.stage ?? '-'}</td>
                <td className="px-3 py-[5px] text-right font-mono tabular-nums text-[#8b8b94]">{item.count ?? '-'}</td>
                <td className="px-3 py-[5px] text-right font-mono tabular-nums text-[#e7e7ea]">{rate(item.pass_at_1)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </Panel>
  )
}

function BandTable({ run }: { run: Run }) {
  const progress = run.stage_progress
  return (
    <Panel
      title="pass rate per band"
      className="px-0 py-0"
      right={
        progress.band ? (
          <>
            <span className="text-[#8b8b94]">{progress.band}</span>
            <span className="tabular-nums">
              {progress.solved} / {progress.total} solved
            </span>
          </>
        ) : undefined
      }
    >
      <div className="px-3 pb-2">
        {run.bands.length === 0 ? (
          <div className="py-3 font-mono text-[10.5px] text-[#4f4f57]">no stage records</div>
        ) : (
          run.bands.map((band, index) => (
            <div
              key={band.band}
              className={`flex items-center gap-3 py-[6px] ${index === 0 ? '' : 'border-t border-white/[0.05]'}`}
            >
              <span className="w-20 shrink-0 truncate text-[11.5px] text-[#c7c7cc]">{band.band}</span>
              <span className="h-[3px] flex-1 bg-white/[0.06]">
                <span className="block h-full" style={{ width: `${Math.min(band.pass_rate * 100, 100)}%`, background: ACCENT }} />
              </span>
              <span className="w-12 shrink-0 text-right font-mono text-[10.5px] tabular-nums text-[#d9d9de]">
                {rate(band.pass_rate)}
              </span>
              <span className="w-16 shrink-0 text-right font-mono text-[10px] tabular-nums text-[#55555d]">
                {whole(band.rollouts)}
              </span>
            </div>
          ))
        )}
      </div>
    </Panel>
  )
}

function RunSection({ run }: { run: Run }) {
  const latest = run.latest
  const steps = run.series.steps
  const evaluations = run.evaluations
  const lastEval = evaluations.length ? evaluations[evaluations.length - 1] : null
  const ageSeconds = run.metrics_age_seconds
  const slow = run.status === 'running' && ageSeconds !== null && ageSeconds > 300
  const memorySeries: ChartSeries[] = [
    { label: 'live', color: ACCENT, points: scaled(pairs(steps, run.series.gpu_live_mib), 1 / 1024) },
    { label: 'cached', color: MUTED, points: scaled(pairs(steps, run.series.gpu_cached_mib), 1 / 1024) },
    { label: 'fragmented', color: DIM, points: scaled(pairs(steps, run.series.gpu_fragmented_mib), 1 / 1024) },
    { label: 'unaccounted', color: FAINT, points: scaled(pairs(steps, run.series.gpu_unaccounted_mib), 1 / 1024) },
    { label: 'device total', color: TRACE, dash: true, points: scaled(pairs(steps, run.series.gpu_total_mib), 1 / 1024) },
  ]
  const stageDone = run.stage_progress.percent

  return (
    <section className="mb-10">
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-1 border-b border-white/[0.08] pb-2">
        <div className="flex min-w-0 items-baseline gap-2.5">
          <h2 className="text-[15px] font-medium tracking-[-0.01em] text-[#f0f0f2]">{run.arm}</h2>
          <span className="truncate font-mono text-[10.5px] text-[#55555d]">{run.run_name}</span>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-[10.5px] tabular-nums text-[#7b7b84]">
          <Status status={run.status} />
          <span>
            stage <span className="text-[#d9d9de]">{run.stage ?? '-'}</span>
          </span>
          <span className={slow ? 'text-[#e9b153]' : undefined}>last write {age(ageSeconds)}</span>
          <span>started {run.started_utc ?? '-'}</span>
          <span>elapsed {elapsed(run.elapsed_seconds)}</span>
        </div>
      </div>

      <StepProgress run={run} />

      {slow ? (
        <div className="mt-2 border-l-2 border-[#e9b153] px-2 py-1 font-mono text-[10.5px] tabular-nums text-[#e9b153]">
          no metrics write for {elapsed(ageSeconds)}
        </div>
      ) : null}

      <Strip>
        <Stat label="steps">
          {whole(run.steps_done)}
          {run.target_steps ? <span className="text-[#6b6b73]"> / {whole(run.target_steps)}</span> : null}
        </Stat>
        <Stat label="mean reward">{num(latest.mean_reward)}</Stat>
        <Stat label="pass rate">{rate(latest.pass_rate)}</Stat>
        <Stat label="pass@1">{rate(lastEval?.pass_at_1)}</Stat>
        <Stat label="entropy">{num(latest.mean_entropy)}</Stat>
        <Stat label="comp len">{num(latest.mean_completion_tokens, 0)}</Stat>
        <Stat label="truncation">{rate(latest.truncation_rate)}</Stat>
        <Stat label="cache hit rate">{perStepRate(latest.cache_hits, latest.rollouts)}</Stat>
      </Strip>

      <Strip>
        <Stat label="prompts per step">{whole(latest.prompts)}</Stat>
        <Stat label="sequences per step">{whole(latest.sequences)}</Stat>
        <Stat label="stage done">
          {stageDone === null || stageDone === undefined ? '-' : `${(stageDone * 100).toFixed(1)}%`}
        </Stat>
        <Stat label="gpu live GiB">{gib(latest.gpu_live_mib)}</Stat>
        <Stat label="gpu cached GiB">{gib(latest.gpu_cached_mib)}</Stat>
        <Stat label="gpu fragmented GiB">{gib(latest.gpu_fragmented_mib)}</Stat>
        <Stat label="gpu unaccounted GiB">{gib(latest.gpu_unaccounted_mib)}</Stat>
        <Stat label="gpu total GiB">{gib(latest.gpu_total_mib)}</Stat>
      </Strip>

      <div className="mt-3 grid grid-cols-1 gap-px border border-white/[0.07] bg-white/[0.07] sm:grid-cols-2 xl:grid-cols-3">
        <Panel title="reward per step" right={num(latest.mean_reward)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.mean_reward) }]} />
        </Panel>
        <Panel title="pass rate per step" right={rate(latest.pass_rate)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.pass_rate) }]} yMin={0} format={percentTick} />
        </Panel>
        <Panel
          title="pass@1"
          right={
            <>
              <Swatch color={MUTED} label="checkpoint" />
              <Swatch color={ACCENT} label="final" />
            </>
          }
        >
          <LineChart
            series={[
              { color: MUTED, dots: true, points: pairs(run.pass_at_1.checkpoint.steps, run.pass_at_1.checkpoint.values) },
              { color: ACCENT, dots: true, points: pairs(run.pass_at_1.final.steps, run.pass_at_1.final.values) },
            ]}
            yMin={0}
            format={percentTick}
          />
        </Panel>
        <Panel title="entropy per step" right={num(latest.mean_entropy)}>
          <LineChart series={[{ color: ACCENT, dots: true, points: pairs(steps, run.series.mean_entropy) }]} />
        </Panel>
        <Panel title="completion tokens per step" right={num(latest.mean_completion_tokens, 0)}>
          <LineChart
            series={[{ color: ACCENT, points: pairs(steps, run.series.mean_completion_tokens) }]}
            yMin={0}
            format={(value) => String(Math.round(value))}
          />
        </Panel>
        <Panel title="truncation rate per step" right={rate(latest.truncation_rate)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.truncation_rate) }]} yMin={0} format={percentTick} />
        </Panel>
        <Panel
          title="prompts and sequences per step"
          right={
            <>
              <Swatch color={ACCENT} label="prompts" />
              <Swatch color={MUTED} label="sequences" />
            </>
          }
        >
          <LineChart
            series={[
              { color: ACCENT, points: pairs(steps, run.series.prompts) },
              { color: MUTED, points: pairs(steps, run.series.sequences) },
            ]}
            yMin={0}
            format={(value) => String(Math.round(value))}
          />
        </Panel>
        <Panel title="stage completion per step (%)" right={run.stage_progress.band ?? undefined}>
          <LineChart series={stageSegments(run.stage_progress.points)} yMin={0} yMax={100} format={(value) => `${value.toFixed(0)}%`} />
        </Panel>
        <Panel
          title="gpu memory per step (GiB)"
          legend={
            <>
              <Swatch color={ACCENT} label="live" />
              <Swatch color={MUTED} label="cached" />
              <Swatch color={DIM} label="fragmented" />
              <Swatch color={FAINT} label="unaccounted" />
              <Swatch color={TRACE} label="total" />
            </>
          }
        >
          <LineChart series={memorySeries} yMin={0} format={(value) => value.toFixed(1)} />
        </Panel>
        <Panel title="reward distribution">
          <BarChart bins={run.distributions.reward.bins} counts={run.distributions.reward.counts} format={(value) => value.toFixed(2)} />
        </Panel>
        <Panel title="tests passed distribution" className="sm:col-span-2">
          <BarChart bins={run.distributions.tests_passed.bins} counts={run.distributions.tests_passed.counts} format={(value) => String(value)} />
        </Panel>
      </div>

      <div className="mt-3 grid grid-cols-1 gap-px border border-white/[0.07] bg-white/[0.07] lg:grid-cols-2">
        <BandTable run={run} />
        <EvaluationTable evaluations={evaluations} />
      </div>
    </section>
  )
}

function SystemStrip({ gpu }: { gpu: Dashboard['gpu'] }) {
  const current = gpu.current
  if (!current) return null
  return (
    <div className="mb-6 flex flex-wrap items-center gap-x-6 gap-y-1 border border-white/[0.07] bg-[#0b0b0d] px-3 py-[7px] font-mono text-[10.5px] tabular-nums text-[#7b7b84]">
      <span className="text-[9.5px] uppercase tracking-[0.12em] text-[#5f5f68]">system</span>
      <span>
        gpu util <span className="text-[#d9d9de]">{num(current.utilization, 0)}%</span>
      </span>
      <span>
        gpu memory{' '}
        <span className="text-[#d9d9de]">
          {gib(current.memory_used)} / {gib(current.memory_total)} GiB
        </span>
      </span>
      <span>
        gpu temp <span className="text-[#d9d9de]">{num(current.temperature, 0)} C</span>
      </span>
      <span>
        samples <span className="text-[#d9d9de]">{gpu.history.length}</span>
      </span>
    </div>
  )
}

export default function App({ initial }: { initial: Dashboard }) {
  const [data, setData] = useState<Dashboard>(initial)
  const [live, setLive] = useState(true)

  useEffect(() => {
    if (!live) return
    let cancelled = false
    const refresh = () => {
      fetch('/data.json')
        .then((response) => (response.ok ? response.json() : Promise.reject(new Error('bad response'))))
        .then((next: Dashboard) => {
          if (!cancelled) setData(next)
        })
        .catch(() => {})
    }
    const timer = window.setInterval(refresh, 5000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [live])

  const runs = data.runs ?? []

  return (
    <div className="min-h-screen bg-[#08080a] text-[#b9b9c0]">
      <header className="sticky top-0 z-10 border-b border-white/[0.07] bg-[#08080a]/95 backdrop-blur">
        <div className="mx-auto flex h-11 max-w-[1560px] items-center justify-between gap-3 px-5">
          <div className="flex items-baseline gap-3 whitespace-nowrap">
            <span className="text-[13px] font-semibold tracking-[0.01em] text-[#ededf0]">RLVR</span>
            <span className="hidden text-[9.5px] uppercase tracking-[0.14em] text-[#5f5f68] sm:inline">training monitor</span>
          </div>
          <div className="flex items-center gap-4 whitespace-nowrap font-mono text-[10.5px] tabular-nums text-[#6b6b73]">
            <span className="hidden md:inline">updated {data.generated_at || '-'}</span>
            <button
              type="button"
              onClick={() => setLive(!live)}
              aria-pressed={live}
              className="flex items-center gap-1.5 uppercase tracking-[0.1em] text-[#7b7b84] transition-colors hover:text-[#d9d9de]"
            >
              <span
                className="inline-block h-[5px] w-[5px] rounded-full"
                style={{ background: live ? ACCENT : '#4f4f57' }}
              />
              {live ? 'live 5s' : 'paused'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1560px] px-5 pb-16 pt-4">
        <SystemStrip gpu={data.gpu ?? { current: null, history: [] }} />
        {runs.length === 0 ? (
          <div className="flex h-64 items-center justify-center border border-white/[0.07] font-mono text-[11px] text-[#4f4f57]">
            waiting for a run
          </div>
        ) : (
          runs.map((run) =>
            run.status === 'waiting' ? (
              <section key={run.arm} className="mb-10">
                <div className="flex items-baseline justify-between border-b border-white/[0.08] pb-2">
                  <div className="flex items-baseline gap-2.5">
                    <h2 className="text-[15px] font-medium text-[#f0f0f2]">{run.arm}</h2>
                    <span className="font-mono text-[10.5px] text-[#55555d]">no run folder</span>
                  </div>
                  <Status status="waiting" />
                </div>
                <div className="flex h-28 items-center justify-center font-mono text-[11px] text-[#4f4f57]">waiting for a run</div>
              </section>
            ) : (
              <RunSection key={run.run_dir ?? run.arm} run={run} />
            ),
          )
        )}
        <footer className="mt-2 flex flex-wrap gap-x-6 gap-y-1 border-t border-white/[0.06] pt-3 font-mono text-[9.5px] text-[#45454c]">
          {runs.map((run) => (
            <span key={run.arm}>{run.run_dir ?? run.arm}</span>
          ))}
          <span className="ml-auto">refresh 5s</span>
        </footer>
      </main>
    </div>
  )
}

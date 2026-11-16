import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { ACCENT, BarChart, DIM, FAINT, LineChart, MUTED, Panel, Swatch, TRACE, type ChartSeries } from './charts'
import { age, elapsed, gib, num, perStepRate, rate, whole } from './format'
import type { Dashboard, Evaluation, QuestionRow, Run, StagePoint } from './types'

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

function hasValues(points: [number, number | null][]): boolean {
  return points.some(([, y]) => y !== null && Number.isFinite(y))
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
  const percent = value * 100
  if (percent === 0) return '0%'
  const digits = Math.abs(percent) < 1 ? 2 : 1
  return `${percent.toFixed(digits)}%`
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
      <div className="text-[10.5px] uppercase leading-[1.5] tracking-[0.09em] text-[#5f5f68]">{label}</div>
      <div className="font-mono text-[13px] tabular-nums leading-[1.45] text-[#d9d9de]">{children}</div>
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
  // One step samples a growing batch of prompts, so the planned step count
  // depends on the batch size and is only correct at the starting size. The
  // prompts the run must present does not depend on batching, so that is the
  // progress figure. The step plan stays as a muted note naming its batch.
  const samples = run.prompt_samples
  const sampleTarget = run.prompt_sample_target
  const percent = sampleTarget ? Math.min(samples / sampleTarget, 1) : null
  return (
    <div className="mt-2.5 flex flex-wrap items-baseline gap-x-2 font-mono text-[11px] tabular-nums text-[#6b6b73]">
      <span className="text-[15px] text-[#e7e7ea]">{whole(run.steps_done)}</span>
      <span>steps</span>
      {percent !== null && sampleTarget !== null ? (
        <>
          <span className="text-[#3f3f46]">/</span>
          <span>
            {whole(samples)} / {whole(sampleTarget)} prompt samples
          </span>
          <span className="text-[#8b8b94]">{(percent * 100).toFixed(1)}%</span>
        </>
      ) : null}
      {run.target_steps ? (
        <span className="text-[#55555d]">
          planned {whole(run.target_steps)} steps
          {run.start_batch_prompts
            ? ` at ${whole(run.start_batch_prompts)} prompts per step`
            : ' at the starting batch'}
        </span>
      ) : null}
    </div>
  )
}

function EvaluationTable({ evaluations }: { evaluations: Evaluation[] }) {
  return (
    <Panel title="evaluations" flush>
      <table className="w-full table-fixed border-collapse text-[11.5px]">
        <thead>
          <tr className="border-t border-white/[0.06] text-[10.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
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
              <td colSpan={5} className="border-t border-white/[0.06] px-3 py-3 font-mono text-[11px] text-[#4f4f57]">
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
      flush
      right={
        progress.band ? (
          <span>
            <span className="text-[#8b8b94]">{progress.band}</span>
            <span className="ml-2 tabular-nums">
              {progress.solved} / {progress.total} solved
            </span>
            {progress.dropped ? (
              <span className="ml-2 text-[#55555d]">{progress.dropped} dropped</span>
            ) : null}
          </span>
        ) : undefined
      }
    >
      <div className="px-3 pb-2">
        <div className="flex items-center gap-3 border-t border-white/[0.06] pb-1 pt-1.5 text-[10.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
          <span className="w-20 shrink-0">band</span>
          <span className="flex-1">pass rate</span>
          <span className="w-12 shrink-0" />
          <span className="w-16 shrink-0 text-right">rollouts</span>
        </div>
        {run.bands.length === 0 ? (
          <div className="py-3 font-mono text-[11px] text-[#4f4f57]">no stage records</div>
        ) : (
          run.bands.map((band, index) => (
            <div
              key={band.band}
              className={`flex items-center gap-3 py-[6px] ${index === 0 ? '' : 'border-t border-white/[0.05]'}`}
            >
              <span className="w-20 shrink-0 truncate text-[12px] text-[#c7c7cc]">{band.band}</span>
              <span className="h-[3px] flex-1 bg-white/[0.06]">
                <span className="block h-full" style={{ width: `${Math.min(band.pass_rate * 100, 100)}%`, background: ACCENT }} />
              </span>
              <span className="w-12 shrink-0 text-right font-mono text-[11px] tabular-nums text-[#d9d9de]">
                {rate(band.pass_rate)}
              </span>
              <span className="w-16 shrink-0 text-right font-mono text-[11px] tabular-nums text-[#55555d]">
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
  const entropyPoints = pairs(steps, run.series.mean_entropy)
  const checkpointPoints = pairs(run.pass_at_1.checkpoint.steps, run.pass_at_1.checkpoint.values)
  const finalPoints = pairs(run.pass_at_1.final.steps, run.pass_at_1.final.values)
  const hasPassAtOne = hasValues(checkpointPoints) || hasValues(finalPoints)
  const hasEntropy = hasValues(entropyPoints)
  const missingPanels: string[] = []
  if (!hasEntropy) {
    missingPanels.push(
      run.entropy_every_steps ? `entropy per step (measured every ${run.entropy_every_steps} steps)` : 'entropy per step',
    )
  }
  if (!hasPassAtOne) missingPanels.push('pass@1')

  return (
    <section className="mb-10">
      <div className="flex flex-wrap items-end justify-between gap-x-6 gap-y-1 border-b border-white/[0.08] pb-2">
        <div className="flex min-w-0 items-baseline gap-2.5">
          <h2 className="text-[15px] font-medium tracking-[-0.01em] text-[#f0f0f2]">{run.arm}</h2>
          <span className="truncate font-mono text-[11px] text-[#55555d]">{run.run_name}</span>
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-[11px] tabular-nums text-[#7b7b84]">
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
        <div className="mt-2 border-l-2 border-[#e9b153] px-2 py-1 font-mono text-[11px] tabular-nums text-[#e9b153]">
          no metrics write for {elapsed(ageSeconds)}
        </div>
      ) : null}

      <Strip>
        <Stat label="steps">
          {whole(run.steps_done)}
        </Stat>
        <Stat label="mean reward">{num(latest.mean_reward)}</Stat>
        <Stat label="pass rate">{rate(latest.pass_rate)}</Stat>
        <Stat label={lastEval && lastEval.step !== null ? `pass@1 at step ${lastEval.step}` : 'pass@1'}>
          {rate(lastEval?.pass_at_1)}
        </Stat>
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

      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
        <Panel title="reward per step" right={num(latest.mean_reward)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.mean_reward) }]} />
        </Panel>
        <Panel title="pass rate per step" right={rate(latest.pass_rate)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.pass_rate) }]} yMin={0} format={percentTick} />
        </Panel>
        {hasPassAtOne ? (
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
                { color: MUTED, dots: true, points: checkpointPoints },
                { color: ACCENT, dots: true, points: finalPoints },
              ]}
              yMin={0}
              format={percentTick}
            />
          </Panel>
        ) : null}
        {hasEntropy ? (
          <Panel title="entropy per step" right={num(latest.mean_entropy)}>
            <LineChart series={[{ color: ACCENT, dots: true, points: entropyPoints }]} />
          </Panel>
        ) : null}
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
        <Panel
          title="tests passed distribution"
          className={hasEntropy ? 'sm:col-span-2' : 'sm:col-span-2 xl:col-span-3'}
        >
          <BarChart bins={run.distributions.tests_passed.bins} counts={run.distributions.tests_passed.counts} format={(value) => String(value)} />
        </Panel>
      </div>

      {missingPanels.length ? (
        <div className="mt-2 font-mono text-[10.5px] text-[#4f4f57]">not measured yet: {missingPanels.join(', ')}</div>
      ) : null}

      <div className="mt-3 grid grid-cols-1 gap-2 lg:grid-cols-2">
        <BandTable run={run} />
        <EvaluationTable evaluations={evaluations} />
      </div>
    </section>
  )
}

type SortMode = 'never-passed' | 'rate-asc' | 'rate-desc' | 'attempts' | 'last-step' | 'prompt'

const SORT_LABELS: Record<SortMode, string> = {
  'never-passed': 'zero passes first',
  'rate-asc': 'pass rate low to high',
  'rate-desc': 'pass rate high to low',
  attempts: 'attempts',
  'last-step': 'last step',
  prompt: 'prompt id',
}

function sortQuestions(rows: QuestionRow[], sort: SortMode): QuestionRow[] {
  const sorted = [...rows]
  sorted.sort((a, b) => {
    if (sort === 'rate-asc') return a.pass_rate - b.pass_rate || b.rollouts - a.rollouts
    if (sort === 'rate-desc') return b.pass_rate - a.pass_rate || b.rollouts - a.rollouts
    if (sort === 'attempts') return b.rollouts - a.rollouts || a.pass_rate - b.pass_rate
    if (sort === 'last-step') return (b.last_step ?? -1) - (a.last_step ?? -1) || a.pass_rate - b.pass_rate
    if (sort === 'prompt') return a.prompt_id.localeCompare(b.prompt_id)
    const noneA = a.full_pass_steps === 0 ? 1 : 0
    const noneB = b.full_pass_steps === 0 ? 1 : 0
    return noneB - noneA || a.pass_rate - b.pass_rate || b.rollouts - a.rollouts
  })
  return sorted
}

function QuestionTable({ rows }: { rows: QuestionRow[] }) {
  if (rows.length === 0) {
    return <div className="px-3 py-4 font-mono text-[11px] text-[#4f4f57]">no question matches the filter</div>
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[900px] table-fixed border-collapse text-[11.5px]">
        <thead className="sticky top-0 z-[1] bg-[#0b0b0d]">
          <tr className="border-t border-white/[0.06] text-[10.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
            <th className="w-[270px] px-3 py-1.5 text-left font-normal">prompt</th>
            <th className="w-20 px-2 py-1.5 text-left font-normal">stage</th>
            <th className="w-14 px-2 py-1.5 text-right font-normal">steps</th>
            <th className="w-16 px-2 py-1.5 text-right font-normal">rollouts</th>
            <th className="w-16 px-2 py-1.5 text-right font-normal">full pass</th>
            <th className="w-32 px-2 py-1.5 text-left font-normal">pass rate</th>
            <th className="w-16 px-2 py-1.5 text-right font-normal">best</th>
            <th className="w-28 px-2 py-1.5 text-left font-normal">last error</th>
            <th className="w-16 px-2 py-1.5 text-right font-normal">last step</th>
            <th className="w-16 px-2 py-1.5 text-right font-normal">solved</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.prompt_id} className="border-t border-white/[0.055] align-top">
              <td className="px-3 py-[5px]">
                <div className="truncate font-mono text-[11px] text-[#d9d9de]">{row.prompt_id}</div>
                {row.dropped ? (
                  <div className="truncate font-mono text-[10px] uppercase tracking-[0.08em] text-[#8b8b94]">
                    {row.solved ? 'solved, then dropped' : 'dropped'}
                    {row.dropped_step !== null ? ` at step ${row.dropped_step}` : ''}
                  </div>
                ) : null}
                {row.text ? (
                  <div className="truncate text-[10.5px] leading-[1.45] text-[#6b6b73]" title={row.text}>
                    {row.text}
                  </div>
                ) : null}
              </td>
              <td className="truncate px-2 py-[5px] text-[#8b8b94]">{row.stage ?? '-'}</td>
              <td className="px-2 py-[5px] text-right font-mono tabular-nums text-[#8b8b94]">{whole(row.steps)}</td>
              <td className="px-2 py-[5px] text-right font-mono tabular-nums text-[#8b8b94]">{whole(row.rollouts)}</td>
              <td
                className="px-2 py-[5px] text-right font-mono tabular-nums"
                style={{ color: row.full_pass_steps === 0 ? '#c76b62' : '#d9d9de' }}
              >
                {whole(row.full_pass_steps)}
              </td>
              <td className="px-2 py-[5px]">
                <div className="flex items-center gap-2">
                  <span className="h-[3px] w-14 bg-white/[0.06]">
                    <span
                      className="block h-full"
                      style={{
                        width: `${Math.min(row.pass_rate * 100, 100)}%`,
                        background: row.full_pass_steps === 0 ? '#c76b62' : ACCENT,
                      }}
                    />
                  </span>
                  <span className="font-mono tabular-nums text-[#d9d9de]">{rate(row.pass_rate)}</span>
                </div>
              </td>
              <td className="px-2 py-[5px] text-right font-mono tabular-nums text-[#b9b9c0]">
                {row.best_passed ?? '-'}
                <span className="text-[#55555d]">/{row.best_total ?? '-'}</span>
              </td>
              <td className="truncate px-2 py-[5px] font-mono text-[10.5px] text-[#8b8b94]">{row.last_error ?? '-'}</td>
              <td className="px-2 py-[5px] text-right font-mono tabular-nums text-[#8b8b94]">{row.last_step ?? '-'}</td>
              <td className="px-2 py-[5px] text-right">
                <span
                  className="font-mono text-[10.5px] uppercase tracking-[0.08em]"
                  style={{ color: row.solved ? ACCENT : row.full_pass_steps === 0 ? '#c76b62' : '#8b8b94' }}
                >
                  {row.solved ? 'yes' : 'no'}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function QuestionsSection({ run }: { run: Run }) {
  const questions = run.questions ?? { total: 0, rows: [] }
  const [stage, setStage] = useState('all')
  const [solved, setSolved] = useState('all')
  const [sort, setSort] = useState<SortMode>('never-passed')
  const [limit, setLimit] = useState(100)
  const [showDropped, setShowDropped] = useState(false)

  const activeRows = useMemo(() => questions.rows.filter((row) => !row.dropped), [questions.rows])
  const droppedIds = run.dropped?.ids ?? []

  const stages = useMemo(() => {
    const found: string[] = []
    for (const row of questions.rows) {
      if (row.stage && !found.includes(row.stage)) found.push(row.stage)
    }
    return found
  }, [questions.rows])

  const filtered = useMemo(() => {
    const matches = (showDropped ? questions.rows : activeRows).filter((row) => {
      if (stage !== 'all' && row.stage !== stage) return false
      if (solved === 'solved') return row.solved
      if (solved === 'unsolved') return !row.solved
      return true
    })
    return sortQuestions(matches, sort)
  }, [questions.rows, activeRows, showDropped, stage, solved, sort])

  const visible = filtered.slice(0, limit)
  const neverPassed = activeRows.filter((row) => row.full_pass_steps === 0).length

  const selectClass =
    'border border-white/[0.08] bg-[#0b0b0d] px-1.5 py-[3px] font-mono text-[10.5px] text-[#b9b9c0] outline-none focus:border-[#e9b153]/60'

  return (
    <Panel
      title="questions"
      flush
      right={
        <>
          <span>{whole(activeRows.length)} prompts</span>
          {droppedIds.length ? <span>{whole(droppedIds.length)} dropped</span> : null}
          {neverPassed ? <span style={{ color: '#c76b62' }}>{whole(neverPassed)} without a full pass</span> : null}
        </>
      }
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 pb-2 pt-1">
        <label className="flex items-center gap-1.5 text-[10.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
          stage
          <select className={selectClass} value={stage} onChange={(event) => setStage(event.target.value)}>
            <option value="all">all</option>
            {stages.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[10.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
          solved
          <select className={selectClass} value={solved} onChange={(event) => setSolved(event.target.value)}>
            <option value="all">all</option>
            <option value="solved">solved</option>
            <option value="unsolved">not solved</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-[10.5px] uppercase tracking-[0.08em] text-[#5f5f68]">
          sort
          <select className={selectClass} value={sort} onChange={(event) => setSort(event.target.value as SortMode)}>
            {(Object.keys(SORT_LABELS) as SortMode[]).map((mode) => (
              <option key={mode} value={mode}>
                {SORT_LABELS[mode]}
              </option>
            ))}
          </select>
        </label>
        {droppedIds.length ? (
          <button
            type="button"
            onClick={() => setShowDropped(!showDropped)}
            aria-pressed={showDropped}
            className="border border-white/[0.08] px-1.5 py-[3px] font-mono text-[10.5px] uppercase tracking-[0.08em] text-[#8b8b94] transition-colors hover:text-[#e7e7ea]"
          >
            {showDropped ? `hide dropped (${droppedIds.length})` : `show dropped (${droppedIds.length})`}
          </button>
        ) : null}
        <span className="ml-auto font-mono text-[10.5px] tabular-nums text-[#55555d]">
          showing {whole(visible.length)} of {whole(filtered.length)} filtered
        </span>
      </div>
      <div className="max-h-[70vh] overflow-y-auto border-t border-white/[0.06]">
        <QuestionTable rows={visible} />
      </div>
      {visible.length < filtered.length ? (
        <div className="border-t border-white/[0.06] px-3 py-2">
          <button
            type="button"
            onClick={() => setLimit(limit + 150)}
            className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-[#8b8b94] transition-colors hover:text-[#e7e7ea]"
          >
            show 150 more
          </button>
        </div>
      ) : null}
    </Panel>
  )
}

function QuestionsView({ runs }: { runs: Run[] }) {
  if (runs.length === 0) {
    return (
      <div className="flex h-64 items-center justify-center border border-white/[0.07] font-mono text-[11px] text-[#4f4f57]">
        waiting for a run
      </div>
    )
  }
  return (
    <>
      {runs.map((run) =>
        run.status === 'waiting' ? (
          <div
            key={run.arm}
            className="mb-6 flex h-24 items-center justify-center border border-white/[0.07] font-mono text-[11px] text-[#4f4f57]"
          >
            {run.arm}: waiting for a run
          </div>
        ) : (
          <section key={run.run_dir ?? run.arm} className="mb-8">
            <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-white/[0.08] pb-2">
              <div className="flex min-w-0 items-baseline gap-2.5">
                <h2 className="text-[15px] font-medium tracking-[-0.01em] text-[#f0f0f2]">{run.arm}</h2>
                <span className="truncate font-mono text-[11px] text-[#55555d]">{run.run_name}</span>
              </div>
              <span className="font-mono text-[11px] tabular-nums text-[#7b7b84]">
                stage <span className="text-[#d9d9de]">{run.stage ?? '-'}</span>
              </span>
            </div>
            <QuestionsSection run={run} />
          </section>
        ),
      )}
    </>
  )
}

function SystemStrip({ gpu }: { gpu: Dashboard['gpu'] }) {
  const current = gpu.current
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

export default function App({ initial }: { initial: Dashboard }) {
  const [data, setData] = useState<Dashboard>(initial)
  const [live, setLive] = useState(true)
  const [tab, setTab] = useState<'overview' | 'questions'>('overview')

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
            <span className="hidden text-[10.5px] uppercase tracking-[0.14em] text-[#5f5f68] sm:inline">training monitor</span>
            <nav className="ml-2 flex items-center gap-1">
              {(['overview', 'questions'] as const).map((name) => (
                <button
                  key={name}
                  type="button"
                  onClick={() => setTab(name)}
                  aria-pressed={tab === name}
                  className="border-b px-2 py-[3px] text-[10.5px] uppercase tracking-[0.1em] transition-colors"
                  style={{
                    color: tab === name ? '#e7e7ea' : '#6b6b73',
                    borderColor: tab === name ? ACCENT : 'transparent',
                  }}
                >
                  {name}
                </button>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-4 whitespace-nowrap font-mono text-[11px] tabular-nums text-[#6b6b73]">
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
        {tab === 'questions' ? (
          <QuestionsView runs={runs} />
        ) : runs.length === 0 ? (
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
                    <span className="font-mono text-[11px] text-[#55555d]">{run.run_dir ? 'no metrics yet' : 'no run folder'}</span>
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
        <footer className="mt-2 flex flex-wrap gap-x-6 gap-y-1 border-t border-white/[0.06] pt-3 font-mono text-[10.5px] text-[#45454c]">
          {runs.map((run) => (
            <span key={run.arm}>{run.run_dir ?? run.arm}</span>
          ))}
          <span className="ml-auto">refresh 5s</span>
        </footer>
      </main>
    </div>
  )
}

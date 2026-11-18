import { ACCENT, BarChart, DIM, FAINT, LineChart, MUTED, Panel, Swatch, TRACE, type ChartSeries } from '../charts'
import { age, elapsed, gib, num, perStepRate, rate, whole } from '../format'
import type { Evaluation, Run, StagePoint } from '../types'
import { Stat, Status, Strip } from './common'

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

export default function RunSection({ run }: { run: Run }) {
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
  const timingFields = [
    ['generation_seconds', 'generation', ACCENT],
    ['scoring_wait_seconds', 'scoring wait', MUTED],
    ['update_seconds', 'update', DIM],
    ['step_seconds', 'step', TRACE],
  ] as const
  const hasTiming = hasValues(pairs(steps, run.series.step_seconds))
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

      {hasTiming ? (
        <Strip columns={6}>
          <Stat label="step time">{num(latest.step_seconds, 1)} s</Stat>
          <Stat label="generation">{num(latest.generation_seconds, 1)} s</Stat>
          <Stat label="scoring wait">{num(latest.scoring_wait_seconds, 1)} s</Stat>
          <Stat label="update">{num(latest.update_seconds, 1)} s</Stat>
          <Stat label="completion tokens/s">{num(latest.completion_tokens_per_second, 1)}</Stat>
          <Stat label="sequences/s">{num(latest.sequences_per_second, 2)}</Stat>
        </Strip>
      ) : null}

      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
        {hasTiming ? (
          <Panel title="training time per step (s)" legend={timingFields.map(([, label, color]) => <Swatch key={label} color={color} label={label} />)}>
            <LineChart series={timingFields.map(([field, , color]) => ({ color, points: pairs(steps, run.series[field]) }))} yMin={0} format={(value) => value.toFixed(1)} />
          </Panel>
        ) : null}
        <Panel title="reward per step" right={num(latest.mean_reward)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.mean_reward) }]} />
        </Panel>
        <Panel title="pass rate per step" right={rate(latest.pass_rate)}>
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.pass_rate) }]} format={percentTick} />
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
          <LineChart series={[{ color: ACCENT, points: pairs(steps, run.series.truncation_rate) }]} format={percentTick} />
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
          <LineChart series={stageSegments(run.stage_progress.points)} format={(value) => `${value.toFixed(0)}%`} />
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

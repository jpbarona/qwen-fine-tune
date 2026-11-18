import { useMemo, useState } from 'react'
import { ACCENT, Panel } from '../charts'
import { rate, whole } from '../format'
import type { QuestionRow, Run } from '../types'

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

export default function QuestionsView({ runs }: { runs: Run[] }) {
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

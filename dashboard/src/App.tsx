import { useState } from 'react'
import { ACCENT } from './charts'
import type { Dashboard } from './types'
import QuestionsView from './components/QuestionsView'
import RunSection from './components/RunSection'
import SystemStrip from './components/SystemStrip'
import { Status } from './components/common'
import { useDashboard } from './useDashboard'

export default function App({ initial }: { initial: Dashboard }) {
  const [live, setLive] = useState(true)
  const [tab, setTab] = useState<'overview' | 'questions'>('overview')
  const { data, error, refreshing, retry } = useDashboard(initial, live)

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
              {live ? 'live 0.5s' : 'paused'}
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-[1560px] px-5 pb-16 pt-4">
        {error ? (
          <div role="status" className="mb-3 flex flex-wrap items-center justify-between gap-2 border-l-2 border-[#e9b153] bg-[#0b0b0d] px-3 py-2 font-mono text-[11px] text-[#e9b153]">
            <span>{error}. Showing the last received data.</span>
            <button type="button" onClick={() => { setLive(true); retry() }} disabled={refreshing} className="underline disabled:opacity-50">
              {refreshing ? 'retrying' : 'retry now'}
            </button>
          </div>
        ) : null}
        <SystemStrip gpu={data.gpu ?? { current: null, history: [], age_seconds: null }} />
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
            <span key={run.run_dir ?? run.arm}>{run.run_dir ?? run.arm}</span>
          ))}
          <span className="ml-auto">{live ? 'refresh 0.5s' : 'refresh paused'}</span>
        </footer>
      </main>
    </div>
  )
}

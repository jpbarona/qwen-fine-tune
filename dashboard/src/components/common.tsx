import type { ReactNode } from 'react'
import { ACCENT } from '../charts'

export function Status({ status }: { status: string }) {
  const color =
    status === 'running' ? ACCENT : status === 'finished' ? '#8fae8f' : status === 'stopped' ? '#c76b62' : '#6b6b73'
  return (
    <span className="flex items-center gap-1.5 uppercase tracking-[0.12em]" style={{ color }}>
      <span className="inline-block h-[5px] w-[5px] rounded-full" style={{ background: color }} />
      {status}
    </span>
  )
}

export function Stat({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="bg-[#0b0b0d] px-3 py-[7px]">
      <div className="text-[10.5px] uppercase leading-[1.5] tracking-[0.09em] text-[#5f5f68]">{label}</div>
      <div className="font-mono text-[13px] tabular-nums leading-[1.45] text-[#d9d9de]">{children}</div>
    </div>
  )
}

export function Strip({ children, columns = 8 }: { children: ReactNode; columns?: 6 | 8 }) {
  return (
    <div className={`mt-3 grid grid-cols-2 gap-px border border-white/[0.07] bg-white/[0.07] sm:grid-cols-4 ${columns === 6 ? 'xl:grid-cols-6' : 'xl:grid-cols-8'}`}>
      {children}
    </div>
  )
}

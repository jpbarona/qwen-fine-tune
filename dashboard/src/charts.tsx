import { useEffect, useRef, useState, type ReactNode } from 'react'

export const ACCENT = '#e9b153'
export const MUTED = '#9a9aa3'
export const DIM = '#6e6e77'
export const FAINT = '#4f4f57'
export const TRACE = '#3a3a41'
export const GRID = 'rgba(255,255,255,0.055)'
export const AXIS = '#66666f'

export interface ChartSeries {
  label?: string
  color: string
  points: [number, number | null][]
  dash?: boolean
  dots?: boolean
}

function useWidth() {
  const ref = useRef<HTMLDivElement | null>(null)
  const [width, setWidth] = useState(0)
  useEffect(() => {
    const element = ref.current
    if (!element) return
    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) setWidth(entry.contentRect.width)
    })
    observer.observe(element)
    setWidth(element.getBoundingClientRect().width)
    return () => observer.disconnect()
  }, [])
  return [ref, width] as const
}

export function Panel({
  title,
  right,
  legend,
  children,
  flush = false,
  className = '',
}: {
  title: string
  right?: ReactNode
  legend?: ReactNode
  children: ReactNode
  flush?: boolean
  className?: string
}) {
  return (
    <section className={`border border-white/[0.07] bg-[#0b0b0d] ${className}`}>
      <header className="flex items-baseline justify-between gap-3 px-3 pb-1.5 pt-2.5">
        <h3 className="text-[11px] uppercase tracking-[0.09em] text-[#6b6b73]">{title}</h3>
        {right ? <div className="flex items-center gap-3 font-mono text-[11px] text-[#8b8b94]">{right}</div> : null}
      </header>
      {legend ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 px-3 pb-1 font-mono text-[10px] text-[#6b6b73]">{legend}</div>
      ) : null}
      <div className={flush ? '' : 'px-3 pb-2'}>{children}</div>
    </section>
  )
}

function extent(values: number[]): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const value of values) {
    if (value < lo) lo = value
    if (value > hi) hi = value
  }
  return [lo, hi]
}

function linePath(points: [number, number | null][], sx: (x: number) => number, sy: (y: number) => number): string {
  let path = ''
  let open = false
  for (const [x, y] of points) {
    if (y === null || !Number.isFinite(y) || !Number.isFinite(x)) {
      open = false
      continue
    }
    path += `${open ? 'L' : 'M'}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`
    open = true
  }
  return path
}

export function LineChart({
  series,
  height = 118,
  format = (value: number) => value.toFixed(2),
  yMin,
  yMax,
  tickCount = 3,
  empty = 'no data',
}: {
  series: ChartSeries[]
  height?: number
  format?: (value: number) => string
  yMin?: number
  yMax?: number
  tickCount?: number
  empty?: string
}) {
  const [ref, width] = useWidth()
  const pad = { left: 42, right: 10, top: 6, bottom: 16 }
  const flat = series.flatMap((entry) => entry.points)
  const ys = flat.map(([, y]) => y).filter((y): y is number => y !== null && Number.isFinite(y))
  const xs = flat.map(([x]) => x).filter((x) => Number.isFinite(x))

  if (!ys.length || !xs.length) {
    return (
      <div ref={ref} className="flex items-center justify-center font-mono text-[10px] text-[#4f4f57]" style={{ height }}>
        {empty}
      </div>
    )
  }

  const [rawLo, rawHi] = extent(ys)
  let lo = yMin ?? rawLo
  let hi = yMax ?? rawHi
  if (lo === hi) {
    lo -= 0.5
    hi += 0.5
  }
  if (yMin === undefined && yMax === undefined) {
    const span = hi - lo
    lo -= span * 0.06
    hi += span * 0.06
  }
  const xLo = Math.min(...xs)
  const xHi = Math.max(...xs)
  const plotWidth = Math.max(width - pad.left - pad.right, 8)
  const plotHeight = height - pad.top - pad.bottom
  const sx = (x: number) => (xHi === xLo ? pad.left + plotWidth / 2 : pad.left + ((x - xLo) / (xHi - xLo)) * plotWidth)
  const sy = (y: number) => pad.top + (1 - (y - lo) / (hi - lo)) * plotHeight

  const ticks = Array.from({ length: tickCount }, (_, index) =>
    tickCount === 1 ? hi : hi - ((hi - lo) * index) / (tickCount - 1),
  )

  const lastPoints = series.map((entry) => {
    for (let index = entry.points.length - 1; index >= 0; index -= 1) {
      const [x, y] = entry.points[index]
      if (y !== null && Number.isFinite(y)) return { x, y, color: entry.color }
    }
    return null
  })

  return (
    <div ref={ref} className="w-full">
      {width > 0 ? (
        <svg width={width} height={height} role="img" aria-label={series.map((entry) => entry.label ?? 'series').join(', ')}>
          {ticks.map((tick, index) => (
            <g key={index}>
              <line x1={pad.left} x2={width - pad.right} y1={sy(tick)} y2={sy(tick)} stroke={GRID} strokeWidth={1} />
              <text x={pad.left - 6} y={sy(tick) + 3} textAnchor="end" fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS}>
                {format(tick)}
              </text>
            </g>
          ))}
          <text x={pad.left} y={height - 3} fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS} textAnchor="start">
            {xLo}
          </text>
          {xHi !== xLo ? (
            <text x={width - pad.right} y={height - 3} fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS} textAnchor="end">
              {xHi}
            </text>
          ) : null}
          {series.map((entry, index) => (
            <path
              key={index}
              d={linePath(entry.points, sx, sy)}
              fill="none"
              stroke={entry.color}
              strokeWidth={1.4}
              strokeDasharray={entry.dash ? '3 3' : undefined}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          ))}
          {series.map((entry, index) =>
            entry.dots
              ? entry.points.map(([x, y], pointIndex) =>
                  y === null || !Number.isFinite(y) ? null : (
                    <circle key={`${index}-${pointIndex}`} cx={sx(x)} cy={sy(y)} r={1.8} fill={entry.color} />
                  ),
                )
              : null,
          )}
          {lastPoints.map((point, index) =>
            point ? <circle key={`last-${index}`} cx={sx(point.x)} cy={sy(point.y)} r={2.1} fill={point.color} /> : null,
          )}
        </svg>
      ) : (
        <div style={{ height }} />
      )}
    </div>
  )
}

export function BarChart({
  bins,
  counts,
  height = 118,
  format = (value: number) => String(value),
  empty = 'no data',
}: {
  bins: number[]
  counts: number[]
  height?: number
  format?: (value: number) => string
  empty?: string
}) {
  const [ref, width] = useWidth()
  const pad = { left: 42, right: 10, top: 6, bottom: 16 }

  if (!bins.length || !counts.length) {
    return (
      <div ref={ref} className="flex items-center justify-center font-mono text-[10px] text-[#4f4f57]" style={{ height }}>
        {empty}
      </div>
    )
  }

  const hi = Math.max(...counts, 1)
  const plotWidth = Math.max(width - pad.left - pad.right, 8)
  const plotHeight = height - pad.top - pad.bottom
  const slot = plotWidth / counts.length
  const barWidth = Math.max(Math.min(slot - 2, 34), 2)
  const sy = (value: number) => pad.top + (1 - value / hi) * plotHeight

  return (
    <div ref={ref} className="w-full">
      {width > 0 ? (
        <svg width={width} height={height} role="img" aria-label="distribution">
          {[0, 0.5, 1].map((fraction) => {
            const value = Math.round(hi * (1 - fraction))
            return (
              <g key={fraction}>
                <line x1={pad.left} x2={width - pad.right} y1={pad.top + fraction * plotHeight} y2={pad.top + fraction * plotHeight} stroke={GRID} strokeWidth={1} />
                <text x={pad.left - 6} y={pad.top + fraction * plotHeight + 3} textAnchor="end" fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS}>
                  {value}
                </text>
              </g>
            )
          })}
          {counts.map((count, index) => (
            <rect
              key={index}
              x={pad.left + index * slot + (slot - barWidth) / 2}
              y={sy(count)}
              width={barWidth}
              height={Math.max(pad.top + plotHeight - sy(count), count > 0 ? 1 : 0)}
              fill="#4a4a52"
            />
          ))}
          <text x={pad.left} y={height - 3} fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS} textAnchor="start">
            {format(bins[0])}
          </text>
          <text x={width - pad.right} y={height - 3} fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS} textAnchor="end">
            {format(bins[bins.length - 1])}
          </text>
        </svg>
      ) : (
        <div style={{ height }} />
      )}
    </div>
  )
}

export function Swatch({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="inline-block h-[2px] w-3" style={{ background: color }} />
      {label}
    </span>
  )
}

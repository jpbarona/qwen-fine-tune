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
  axis?: 'left' | 'right'
  format?: (value: number) => string
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
    <section className={`min-w-0 border border-white/[0.07] bg-[#0b0b0d] ${className}`}>
      <header className="flex min-w-0 flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 px-3 pb-1.5 pt-2.5">
        <h3 className="min-w-0 break-words text-[11px] uppercase tracking-[0.09em] text-[#6b6b73]">{title}</h3>
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

function axisBand(values: number[], yMin?: number, yMax?: number): [number, number] {
  const [rawLo, rawHi] = extent(values)
  let lo = yMin ?? rawLo
  let hi = yMax ?? rawHi
  if (lo === hi) {
    // A flat series needs a visible band. Respect a pinned lower bound so a
    // zero series draws 0 to 1 instead of -0.5 to 0.5.
    if (yMin !== undefined) {
      hi = lo + 1
    } else if (yMax !== undefined) {
      lo = hi - 1
    } else {
      lo -= 0.5
      hi += 0.5
    }
  }
  if (yMin === undefined && yMax === undefined) {
    const span = hi - lo
    lo -= span * 0.06
    hi += span * 0.06
  }
  return [lo, hi]
}

function ticksFor(hi: number, lo: number, tickCount: number): number[] {
  return Array.from({ length: tickCount }, (_, index) =>
    tickCount === 1 ? hi : hi - ((hi - lo) * index) / (tickCount - 1),
  )
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
  height = 180,
  format = (value: number) => value.toFixed(2),
  yMin,
  yMax,
  tickCount = 6,
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
  const leftSeries = series.filter((entry) => entry.axis !== 'right')
  const rightSeries = series.filter((entry) => entry.axis === 'right')
  const pointsOf = (entries: ChartSeries[]) => entries.flatMap((entry) => entry.points)
  const ysOf = (entries: ChartSeries[]) =>
    pointsOf(entries)
      .map(([, y]) => y)
      .filter((y): y is number => y !== null && Number.isFinite(y))
  const leftYs = ysOf(leftSeries)
  const rightYs = ysOf(rightSeries)
  const ys = [...leftYs, ...rightYs]
  const xs = pointsOf(series)
    .map(([x]) => x)
    .filter((x) => Number.isFinite(x))

  if (!ys.length || !xs.length) {
    return (
      <div ref={ref} className="flex items-center justify-center font-mono text-[10px] text-[#4f4f57]" style={{ height }}>
        {empty}
      </div>
    )
  }

  const [lo, hi] = axisBand(leftYs.length ? leftYs : rightYs, yMin, yMax)
  const hasRight = rightYs.length > 0
  const [rightLo, rightHi] = hasRight ? axisBand(rightYs) : [0, 1]
  const leftFormat = leftSeries.find((entry) => entry.format)?.format ?? format
  const rightFormat = rightSeries.find((entry) => entry.format)?.format ?? format
  const rightTicks = hasRight ? ticksFor(rightHi, rightLo, tickCount) : []
  const rightPad = hasRight
    ? Math.max(10, Math.ceil(Math.max(...rightTicks.map((tick) => rightFormat(tick).length)) * 6.4) + 10)
    : 10
  const pad = { left: 42, right: rightPad, top: 6, bottom: 16 }
  const xLo = Math.min(...xs)
  const xHi = Math.max(...xs)
  const plotWidth = Math.max(width - pad.left - pad.right, 8)
  const plotHeight = height - pad.top - pad.bottom
  const sx = (x: number) => (xHi === xLo ? pad.left + plotWidth / 2 : pad.left + ((x - xLo) / (xHi - xLo)) * plotWidth)
  const syLeft = (y: number) => pad.top + (1 - (Math.min(Math.max(y, lo), hi) - lo) / (hi - lo)) * plotHeight
  const syRight = (y: number) =>
    pad.top + (1 - (Math.min(Math.max(y, rightLo), rightHi) - rightLo) / (rightHi - rightLo)) * plotHeight
  const syOf = (entry: ChartSeries) => (entry.axis === 'right' ? syRight : syLeft)

  const ticks = ticksFor(hi, lo, tickCount)
  const crossesZero = lo < 0 && hi > 0

  const lastPoints = series.map((entry) => {
    for (let index = entry.points.length - 1; index >= 0; index -= 1) {
      const [x, y] = entry.points[index]
      if (y !== null && Number.isFinite(y)) return { x, y, color: entry.color, right: entry.axis === 'right' }
    }
    return null
  })

  return (
    <div ref={ref} className="w-full">
      {width > 0 ? (
        <svg width={width} height={height} role="img" aria-label={series.map((entry) => entry.label ?? 'series').join(', ')}>
          {ticks.map((tick, index) => (
            <g key={index}>
              <line x1={pad.left} x2={width - pad.right} y1={syLeft(tick)} y2={syLeft(tick)} stroke={GRID} strokeWidth={1} strokeDasharray="3 3" />
              <text x={pad.left - 6} y={syLeft(tick) + 3} textAnchor="end" fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS}>
                {leftFormat(tick)}
              </text>
            </g>
          ))}
          {hasRight
            ? rightTicks.map((tick, index) => (
                <text
                  key={`right-${index}`}
                  x={width - pad.right + 6}
                  y={syRight(tick) + 3}
                  textAnchor="start"
                  fontSize={10.5}
                  fontFamily="ui-monospace, monospace"
                  fill={AXIS}
                >
                  {rightFormat(tick)}
                </text>
              ))
            : null}
          {crossesZero ? (
            <line x1={pad.left} x2={width - pad.right} y1={syLeft(0)} y2={syLeft(0)} stroke="rgba(255,255,255,0.24)" strokeWidth={1.2} />
          ) : null}
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
              d={linePath(entry.points, sx, syOf(entry))}
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
                    <circle key={`${index}-${pointIndex}`} cx={sx(x)} cy={syOf(entry)(y)} r={1.8} fill={entry.color} />
                  ),
                )
              : null,
          )}
          {lastPoints.map((point, index) =>
            point ? (
              <circle
                key={`last-${index}`}
                cx={sx(point.x)}
                cy={(point.right ? syRight : syLeft)(point.y)}
                r={2.1}
                fill={point.color}
              />
            ) : null,
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
  height = 180,
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
                <line x1={pad.left} x2={width - pad.right} y1={pad.top + fraction * plotHeight} y2={pad.top + fraction * plotHeight} stroke={GRID} strokeWidth={1} strokeDasharray="3 3" />
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

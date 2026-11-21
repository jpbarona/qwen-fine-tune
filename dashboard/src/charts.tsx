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

function labelsRepeat(values: number[], format: (value: number) => string): boolean {
  const labels = values.map((value) => format(value))
  return new Set(labels).size < labels.length
}

const NICE_STEPS = [1, 2, 2.5, 5]

// Round a preferred step up to the next value from 1, 2, 2.5, 5 times a
// power of ten. An integer axis never uses a step that is not a whole number,
// so every tick stays exact.
function niceStep(raw: number, integerOnly: boolean): number {
  const target = Number.isFinite(raw) && raw > 0 ? raw : 1
  const magnitude = Math.pow(10, Math.floor(Math.log10(target)))
  for (const mantissa of NICE_STEPS) {
    const step = mantissa * magnitude
    if (step >= target && (!integerOnly || Number.isInteger(step))) return step
  }
  const next = magnitude * 10
  return integerOnly ? Math.max(1, Math.round(next)) : next
}

function nextNiceStep(step: number, integerOnly: boolean): number {
  return niceStep(step * (1 + 1e-9), integerOnly)
}

function roundUpTo(value: number, step: number): number {
  return Math.ceil(value / step - 1e-9) * step
}

// Strip floating point noise so a tick is an exact multiple of its step.
function cleanTick(value: number, step: number): number {
  if (!Number.isFinite(value)) return value
  if (Number.isInteger(step)) return Math.round(value)
  const digits = Math.max(0, Math.ceil(-Math.log10(step)) + 2)
  return Number(value.toFixed(Math.min(digits, 12)))
}

function looksInteger(format: (value: number) => string): boolean {
  return /^-?\d+%?$/.test(format(2.5).trim())
}

function makeTicks(
  values: number[],
  yMin: number | undefined,
  yMax: number | undefined,
  tickCount: number,
  format: (value: number) => string,
): { lo: number; hi: number; ticks: number[] } {
  const [lo, hi] = axisBand(values, yMin, yMax)
  const integerOnly = values.length > 0 && values.every((value) => Number.isInteger(value)) && looksInteger(format)
  const intervals = Math.max(tickCount - 1, 1)
  if (yMin !== undefined && yMax !== undefined) {
    const step = (hi - lo) / intervals
    return {
      lo,
      hi,
      ticks: Array.from({ length: tickCount }, (_, index) => cleanTick(hi - index * step, step)),
    }
  }
  let step = niceStep((hi - lo) / intervals, integerOnly)
  let bottom = lo
  let top = hi
  let ticks: number[] = []
  for (let attempt = 0; attempt < 6; attempt += 1) {
    if (yMin !== undefined) {
      bottom = yMin
      top = yMin + step * intervals
    } else if (yMax !== undefined) {
      top = yMax
      bottom = yMax - step * intervals
    } else {
      top = roundUpTo(hi, step)
      bottom = top - step * intervals
      // Aligning the top tick upward on a non-negative series should not add
      // negative grid lines it never had.
      if (bottom < 0 && lo >= 0) {
        bottom = 0
        top = step * intervals
      }
    }
    ticks = Array.from({ length: tickCount }, (_, index) => cleanTick(top - index * step, step))
    if (!labelsRepeat(ticks, format)) break
    step = nextNiceStep(step, integerOnly)
  }
  return { lo: bottom, hi: top, ticks }
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

  const leftFormat = leftSeries.find((entry) => entry.format)?.format ?? format
  const rightFormat = rightSeries.find((entry) => entry.format)?.format ?? format
  const leftValues = leftYs.length ? leftYs : rightYs
  const left = makeTicks(leftValues, yMin, yMax, tickCount, leftFormat)
  const lo = left.lo
  const hi = left.hi
  const ticks = left.ticks
  const hasRight = rightYs.length > 0
  const right = hasRight ? makeTicks(rightYs, undefined, undefined, tickCount, rightFormat) : { lo: 0, hi: 1, ticks: [] }
  const rightLo = right.lo
  const rightHi = right.hi
  const rightTicks = right.ticks
  const rightPad = hasRight
    ? Math.max(10, Math.ceil(Math.max(...rightTicks.map((tick) => rightFormat(tick).length)) * 6.4) + 10)
    : 10
  // The highest tick sits on the top edge of the plot, so it needs enough
  // room above the baseline for the glyph to render without clipping.
  const pad = { left: 42, right: rightPad, top: 12, bottom: 16 }
  const xLo = Math.min(...xs)
  const xHi = Math.max(...xs)
  const plotWidth = Math.max(width - pad.left - pad.right, 8)
  const plotHeight = height - pad.top - pad.bottom
  const sx = (x: number) => (xHi === xLo ? pad.left + plotWidth / 2 : pad.left + ((x - xLo) / (xHi - xLo)) * plotWidth)
  const syLeft = (y: number) => pad.top + (1 - (Math.min(Math.max(y, lo), hi) - lo) / (hi - lo)) * plotHeight
  const syRight = (y: number) =>
    pad.top + (1 - (Math.min(Math.max(y, rightLo), rightHi) - rightLo) / (rightHi - rightLo)) * plotHeight
  const syOf = (entry: ChartSeries) => (entry.axis === 'right' ? syRight : syLeft)

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

  const barTick = (value: number) => String(Math.round(value))
  const axis = makeTicks(counts, 0, undefined, 6, barTick)
  const hi = axis.hi
  const plotWidth = Math.max(width - pad.left - pad.right, 8)
  const plotHeight = height - pad.top - pad.bottom
  const slot = plotWidth / counts.length
  const barWidth = Math.max(Math.min(slot - 2, 34), 2)
  const sy = (value: number) => pad.top + (1 - value / hi) * plotHeight

  return (
    <div ref={ref} className="w-full">
      {width > 0 ? (
        <svg width={width} height={height} role="img" aria-label="distribution">
          {axis.ticks.map((tick) => (
            <g key={tick}>
              <line x1={pad.left} x2={width - pad.right} y1={sy(tick)} y2={sy(tick)} stroke={GRID} strokeWidth={1} strokeDasharray="3 3" />
              <text x={pad.left - 6} y={sy(tick) + 3} textAnchor="end" fontSize={10.5} fontFamily="ui-monospace, monospace" fill={AXIS}>
                {barTick(tick)}
              </text>
            </g>
          ))}
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

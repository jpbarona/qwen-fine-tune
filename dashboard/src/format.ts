const DASH = '-'

export function num(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return value.toFixed(digits)
}

export function whole(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return Math.round(value).toLocaleString('en-US')
}

export function rate(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return DASH
  return `${(value * 100).toFixed(1)}%`
}

export function perStepRate(numerator: number | null | undefined, denominator: number | null | undefined): string {
  if (numerator === null || numerator === undefined || !denominator) return DASH
  return (numerator / denominator).toFixed(3)
}

export function gib(mib: number | null | undefined, digits = 1): string {
  if (mib === null || mib === undefined || !Number.isFinite(mib)) return DASH
  return (mib / 1024).toFixed(digits)
}

export function elapsed(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return DASH
  const total = Math.max(0, Math.floor(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  if (h) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m) return `${m}m ${String(s).padStart(2, '0')}s`
  return `${s}s`
}

export function age(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return DASH
  return `${elapsed(seconds)} ago`
}

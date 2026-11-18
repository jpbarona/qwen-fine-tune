import type { Dashboard } from './types'

type RecordValue = Record<string, unknown>

function isRecord(value: unknown): value is RecordValue {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function isNumber(value: unknown): boolean {
  return typeof value === 'number' && Number.isFinite(value)
}

function isOptionalNumber(value: unknown): boolean {
  return value === null || value === undefined || isNumber(value)
}

function isOptionalString(value: unknown): boolean {
  return value === null || value === undefined || typeof value === 'string'
}

function hasNumbers(record: RecordValue, fields: string[]): boolean {
  return fields.every((field) => isOptionalNumber(record[field]))
}

function hasStrings(record: RecordValue, fields: string[]): boolean {
  return fields.every((field) => isOptionalString(record[field]))
}

function isArrayOf(value: unknown, check: (item: unknown) => boolean): boolean {
  return Array.isArray(value) && value.every(check)
}

function isPointSeries(value: unknown): boolean {
  return isRecord(value) && isArrayOf(value.steps, isOptionalNumber)
    && isArrayOf(value.values, isOptionalNumber)
}

function isHistogram(value: unknown): boolean {
  return isRecord(value) && isArrayOf(value.bins, isNumber) && isArrayOf(value.counts, isNumber)
}

function isGpuSample(value: unknown): boolean {
  return isRecord(value) && hasNumbers(value, ['utilization', 'memory_used', 'memory_total', 'temperature', 't'])
}

function isQuestion(value: unknown): boolean {
  return isRecord(value) && typeof value.prompt_id === 'string'
    && hasStrings(value, ['stage', 'text', 'last_error'])
    && hasNumbers(value, ['steps', 'rollouts', 'full_pass_steps', 'pass_rate', 'best_passed', 'best_total', 'last_step', 'dropped_step'])
    && (value.solved === undefined || typeof value.solved === 'boolean')
    && (value.dropped === undefined || typeof value.dropped === 'boolean')
}

function isRun(value: unknown): boolean {
  if (!isRecord(value) || typeof value.arm !== 'string' || typeof value.status !== 'string') return false
  if (!hasStrings(value, ['run_name', 'run_dir', 'started', 'started_utc', 'stage'])
    || !hasNumbers(value, ['elapsed_seconds', 'steps_done', 'target_steps', 'start_batch_prompts',
      'prompt_samples', 'prompt_sample_target', 'entropy_every_steps', 'metrics_age_seconds'])) return false
  if (!isRecord(value.latest) || !Object.values(value.latest).every(isOptionalNumber)) return false
  if (!isRecord(value.series) || !Array.isArray(value.series.steps)
    || !Object.values(value.series).every((series) => isArrayOf(series, isOptionalNumber))) return false
  const progress = value.stage_progress
  if (!isRecord(progress) || !hasStrings(progress, ['band'])
    || !hasNumbers(progress, ['solved', 'total', 'dropped', 'percent'])
    || !isArrayOf(progress.points, (point) => isRecord(point) && typeof point.band === 'string'
      && ['step', 'percent', 'solved', 'total'].every((field) => isNumber(point[field])))) return false
  if (!isRecord(value.pass_at_1) || !isPointSeries(value.pass_at_1.checkpoint)
    || !isPointSeries(value.pass_at_1.final)) return false
  if (!isRecord(value.distributions) || !isHistogram(value.distributions.reward)
    || !isHistogram(value.distributions.tests_passed)) return false
  if (!isArrayOf(value.evaluations, (item) => isRecord(item)
    && hasStrings(item, ['kind', 'stage']) && hasNumbers(item, ['step', 'count', 'pass_at_1']))) return false
  if (!isArrayOf(value.bands, (band) => isRecord(band) && typeof band.band === 'string'
    && hasNumbers(band, ['rollouts', 'pass_rate']))) return false
  if (value.questions != null && (!isRecord(value.questions)
    || !isArrayOf(value.questions.rows, isQuestion))) return false
  if (value.dropped != null && (!isRecord(value.dropped)
    || !isArrayOf(value.dropped.ids, (id) => typeof id === 'string'))) return false
  return true
}

/** Validate collections before rendering, while allowing absent historical metrics. */
export function isDashboard(value: unknown): value is Dashboard {
  return isRecord(value) && typeof value.generated_at === 'string'
    && isArrayOf(value.runs, isRun) && isRecord(value.gpu)
    && (value.gpu.current == null || isGpuSample(value.gpu.current))
    && isArrayOf(value.gpu.history, isGpuSample) && isOptionalNumber(value.gpu.age_seconds)
}

export type Nullable = number | null | undefined

export interface SeriesMap {
  steps: (number | null)[]
  mean_reward: Nullable[]
  pass_rate: Nullable[]
  mean_entropy: Nullable[]
  grad_norm: Nullable[]
  update_norm: Nullable[]
  mean_completion_tokens: Nullable[]
  truncation_rate: Nullable[]
  cache_hits: Nullable[]
  prompts: Nullable[]
  sequences: Nullable[]
  train_prompts: Nullable[]
  parked_prompts: Nullable[]
  advantage_share: Nullable[]
  group_spread: Nullable[]
  group_sd: Nullable[]
  gpu_live_mib: Nullable[]
  gpu_cached_mib: Nullable[]
  gpu_fragmented_mib: Nullable[]
  gpu_unaccounted_mib: Nullable[]
  gpu_total_mib: Nullable[]
  [key: string]: (number | null | undefined)[] | undefined
}

export interface StagePoint {
  step: number
  band: string
  solved: number
  total: number
  percent: number
}

export interface StageProgress {
  band: string | null
  solved: number
  total: number
  dropped: number
  percent: number | null
  points: StagePoint[]
}

export interface BandRate {
  band: string
  rollouts: number
  pass_rate: number
}

export interface Evaluation {
  step: number | null
  kind: string
  stage: string | null
  count: number | null
  pass_at_1: number | null
}

export interface PointSeries {
  steps: (number | null)[]
  values: Nullable[]
}

export interface Histogram {
  bins: number[]
  counts: number[]
}

export interface QuestionRow {
  prompt_id: string
  stage: string | null
  steps: number
  rollouts: number
  full_pass_steps: number
  pass_rate: number
  best_passed: number | null
  best_total: number | null
  last_error: string | null
  last_step: number | null
  solved: boolean
  dropped: boolean
  dropped_step: number | null
  text: string | null
}

export interface Questions {
  total: number
  rows: QuestionRow[]
}

export interface DroppedPrompt {
  step: number | null
  prompt_ids: string[]
}

export interface Dropped {
  ids: string[]
  events: DroppedPrompt[]
  steps: Record<string, number>
}

export interface Run {
  arm: string
  run_name: string | null
  run_dir: string | null
  started: string | null
  started_utc: string | null
  elapsed_seconds: number | null
  status: string
  steps_done: number
  target_steps: number | null
  start_batch_prompts: number | null
  prompt_samples: number
  prompt_sample_target: number | null
  entropy_every_steps: number | null
  pool_policy: string | null
  stage: string | null
  bands: BandRate[]
  stage_progress: StageProgress
  latest: Record<string, Nullable>
  series: SeriesMap
  pass_at_1: { checkpoint: PointSeries; final: PointSeries }
  evaluations: Evaluation[]
  distributions: { reward: Histogram; tests_passed: Histogram }
  questions: Questions
  dropped: Dropped
  metrics_age_seconds: number | null
}

export interface GpuSample {
  utilization: number | null
  memory_used: number | null
  memory_total: number | null
  temperature: number | null
  t?: number
}

export interface Dashboard {
  generated_at: string
  runs: Run[]
  gpu: { current: GpuSample | null; history: GpuSample[]; age_seconds: number | null }
}

declare global {
  interface Window {
    __RLVR_DATA__?: Dashboard
  }
}

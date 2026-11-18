import assert from 'node:assert/strict'
import { test } from 'node:test'
import { isDashboard } from '../src/validateDashboard.ts'

function historicalRun() {
  return {
    arm: 'code-score', status: 'finished', run_dir: '/runs/code-score/old',
    latest: { mean_reward: 0.5, mean_entropy: null },
    series: { steps: [1, 2], mean_reward: [0.2, 0.5], mean_entropy: [null, null] },
    stage_progress: { band: null, solved: 0, total: 0, percent: null, points: [] },
    pass_at_1: { checkpoint: { steps: [], values: [] }, final: { steps: [2], values: [0.5] } },
    distributions: { reward: { bins: [0], counts: [2] }, tests_passed: { bins: [], counts: [] } },
    evaluations: [{ step: 2, kind: 'final', pass_at_1: 0.5 }],
    bands: [{ band: 'easy', rollouts: 8, pass_rate: 0.5 }],
  }
}

function payload() {
  return {
    generated_at: '2026-11-18T17:18:00Z', runs: [historicalRun()],
    gpu: { current: null, history: [], age_seconds: null },
  }
}

test('accepts historical metrics without timing, question or dropped collections', () => {
  assert.equal(isDashboard(payload()), true)
})

test('accepts an empty dashboard and a waiting run', () => {
  const empty = payload()
  empty.runs = []
  assert.equal(isDashboard(empty), true)
  const waiting = payload()
  waiting.runs[0] = {
    ...historicalRun(), status: 'waiting', latest: {}, series: { steps: [] },
    evaluations: [], bands: [],
  }
  assert.equal(isDashboard(waiting), true)
})

test('accepts sparse GPU readings and newer fields', () => {
  const data: any = payload()
  data.gpu.current = { utilization: null, memory_used: 0 }
  data.gpu.history = [{ utilization: 0 }, { utilization: null }, { utilization: 93, t: 100 }]
  data.runs[0].latest.step_seconds = 4.2
  data.runs[0].series.step_seconds = [null, 4.2]
  data.runs[0].questions = { rows: [{ prompt_id: 'p1', pass_rate: 1, solved: true }] }
  data.runs[0].dropped = { ids: ['p2'] }
  assert.equal(isDashboard(data), true)
})

const corruptions: [string, (data: any) => void][] = [
  ['null GPU history sample', (data) => { data.gpu.history = [null] }],
  ['non-numeric GPU sample', (data) => { data.gpu.history = [{ utilization: {} }] }],
  ['invalid current GPU sample', (data) => { data.gpu.current = [] }],
  ['missing checkpoint series', (data) => { data.runs[0].pass_at_1.checkpoint = {} }],
  ['null histogram', (data) => { data.runs[0].distributions.reward = null }],
  ['invalid histogram bins', (data) => { data.runs[0].distributions.reward.bins = [null] }],
  ['null stage point', (data) => { data.runs[0].stage_progress.points = [null] }],
  ['invalid stage percentage', (data) => { data.runs[0].stage_progress.percent = '50%' }],
  ['null evaluation', (data) => { data.runs[0].evaluations = [null] }],
  ['null band', (data) => { data.runs[0].bands = [null] }],
  ['missing question rows', (data) => { data.runs[0].questions = {} }],
  ['null question', (data) => { data.runs[0].questions = { rows: [null] } }],
  ['non-string prompt ID', (data) => { data.runs[0].questions = { rows: [{ prompt_id: 1 }] } }],
  ['non-string dropped ID', (data) => { data.runs[0].dropped = { ids: [{}] } }],
  ['object metric value', (data) => { data.runs[0].latest.mean_reward = {} }],
  ['non-array metric series', (data) => { data.runs[0].series.mean_reward = {} }],
  ['null run', (data) => { data.runs = [null] }],
]

for (const [description, corrupt] of corruptions) {
  test(`rejects ${description} before replacing the displayed data`, () => {
    const data = payload()
    corrupt(data)
    assert.equal(isDashboard(data), false)
  })
}

test('rejects malformed roots without throwing', () => {
  for (const value of [null, [], true, 12, 'text', {}, { runs: [], gpu: {} }]) {
    assert.equal(isDashboard(value), false)
  }
})

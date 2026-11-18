import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'
import type { Dashboard } from './types'
import { isDashboard } from './validateDashboard'

const empty: Dashboard = {
  generated_at: '',
  runs: [],
  gpu: { current: null, history: [], age_seconds: null },
}

function readInitial(): Dashboard {
  const element = document.getElementById('initial-data')
  if (!element?.textContent) return empty
  try {
    const payload: unknown = JSON.parse(element.textContent)
    return isDashboard(payload) ? payload : empty
  } catch {
    return empty
  }
}

const root = document.getElementById('root')
if (root) {
  createRoot(root).render(
    <StrictMode>
      <App initial={readInitial()} />
    </StrictMode>,
  )
}

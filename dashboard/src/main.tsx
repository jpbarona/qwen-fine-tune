import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './index.css'
import type { Dashboard } from './types'

const empty: Dashboard = { generated_at: '', runs: [], gpu: { current: null, history: [] } }

function readInitial(): Dashboard {
  const element = document.getElementById('initial-data')
  if (!element?.textContent) return empty
  try {
    return JSON.parse(element.textContent) as Dashboard
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

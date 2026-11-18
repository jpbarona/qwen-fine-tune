import { useCallback, useEffect, useState } from 'react'
import type { Dashboard } from './types'
import { isDashboard } from './validateDashboard'

const REFRESH_MS = 5_000
const REQUEST_TIMEOUT_MS = 15_000

/** Keep the last good payload through failures and cancel requests on pause. */
export function useDashboard(initial: Dashboard, live: boolean) {
  const [data, setData] = useState(initial)
  const [error, setError] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const [revision, setRevision] = useState(0)
  const retry = useCallback(() => setRevision((value) => value + 1), [])

  useEffect(() => {
    if (!live) return
    let disposed = false
    let controller: AbortController | null = null
    let timer: ReturnType<typeof setTimeout> | undefined

    const refresh = async () => {
      if (disposed || controller || document.hidden) return
      const request = new AbortController()
      controller = request
      const timeout = window.setTimeout(() => request.abort(), REQUEST_TIMEOUT_MS)
      setRefreshing(true)
      try {
        const response = await fetch('/data.json', { signal: request.signal, cache: 'no-store' })
        if (!response.ok) throw new Error(`Refresh failed (HTTP ${response.status})`)
        const payload: unknown = await response.json()
        if (!isDashboard(payload)) throw new Error('Refresh returned invalid dashboard data')
        if (!disposed) {
          setData(payload)
          setError(null)
        }
      } catch (failure) {
        if (!disposed) {
          setError(request.signal.aborted ? 'Refresh timed out'
            : failure instanceof Error ? failure.message : 'Refresh failed')
        }
      } finally {
        window.clearTimeout(timeout)
        controller = null
        if (!disposed) {
          setRefreshing(false)
          // Schedule after completion so slow responses never overlap.
          if (live) timer = window.setTimeout(refresh, REFRESH_MS)
        }
      }
    }

    const onVisible = () => {
      if (document.hidden) return
      window.clearTimeout(timer)
      void refresh()
    }
    document.addEventListener('visibilitychange', onVisible)
    void refresh()
    return () => {
      disposed = true
      window.clearTimeout(timer)
      controller?.abort()
      document.removeEventListener('visibilitychange', onVisible)
      setRefreshing(false)
    }
  }, [live, revision])

  return { data, error, refreshing, retry }
}

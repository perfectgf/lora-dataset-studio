import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { useLocation } from 'react-router'
import { apiFetch, fetchWithCsrfRetry, getCsrfToken, putJson } from '../api/fetchClient'
import { shouldSendUsageActivity, usageFeature } from './activity.js'
import { USAGE_POLICY_VERSION } from './policy.js'

const UsageStatisticsContext = createContext(null)
const CHANGE_KEY = 'lds:usage-statistics-changed'

export function UsageStatisticsProvider({ children }) {
  const { pathname } = useLocation()
  const [state, setState] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const consent = useRef(null)
  const saving = useRef(false)
  const revision = useRef(0)
  const pending = useRef(new Set())
  const lastSent = useRef(new Map())

  const pause = useCallback(() => {
    ++revision.current // invalidate any consent read still in flight
    consent.current = null
    for (const request of pending.current) request.abort()
    pending.current.clear()
  }, [])

  const accept = useCallback((data) => {
    consent.current = data
    setState(data)
    setError('')
    if (!data?.enabled) lastSent.current.clear()
  }, [])

  const refresh = useCallback(async () => {
    pause()
    if (saving.current) return
    const request = revision.current
    setLoading(true)
    try {
      const data = await apiFetch('/api/usage-statistics', { background: true, cache: 'no-store' })
      if (request === revision.current) accept(data)
    } catch {
      if (request === revision.current) {
        setError('Could not check your choice. Usage activity stays paused. Try again when connected.')
      }
    } finally {
      if (request === revision.current) setLoading(false)
    }
  }, [accept, pause])

  const announceChange = useCallback((mode = 'refresh') => {
    // A change signal only, not a cached consent choice or an installation ID.
    // Other tabs must ask the server again before any new activity is allowed.
    try { localStorage.setItem(CHANGE_KEY, `${mode}:${Date.now()}:${Math.random()}`) } catch { /* storage may be blocked */ }
  }, [])

  const setEnabled = useCallback(async (enabled) => {
    if (saving.current) return
    saving.current = true
    pause()
    if (!enabled) announceChange('pause')
    setBusy(true)
    setError('')
    try {
      const data = await putJson('/api/usage-statistics', { enabled, policy_version: USAGE_POLICY_VERSION }, { background: true })
      accept(data)
      announceChange()
    } catch (failure) {
      setError(failure?.status === 409
        ? 'Usage sharing is unavailable on this installation. Your choice was not saved.'
        : 'Could not save your choice. Usage activity is paused in this tab; retry to save it for the installation.')
    } finally {
      saving.current = false
      setBusy(false)
      setLoading(false)
    }
  }, [accept, announceChange, pause])

  useEffect(() => {
    refresh()
    const onVisibility = () => {
      if (document.visibilityState === 'visible') refresh()
      else pause()
    }
    const onStorage = (event) => {
      if (event.key !== CHANGE_KEY) return
      // Do not GET before an opt-out PUT finishes: that could restore the old
      // enabled state in a second tab while the user is turning sharing off.
      if (event.newValue?.startsWith('pause:')) pause()
      else refresh()
    }
    document.addEventListener('visibilitychange', onVisibility)
    window.addEventListener('storage', onStorage)
    return () => {
      pause()
      document.removeEventListener('visibilitychange', onVisibility)
      window.removeEventListener('storage', onStorage)
    }
  }, [pause, refresh])

  useEffect(() => {
    const feature = usageFeature(pathname)
    const onActivity = (event) => {
      const now = performance.now()
      if (!shouldSendUsageActivity({
        enabled: consent.current?.configured === true && consent.current?.enabled === true,
        trusted: event.isTrusted,
        visible: document.visibilityState === 'visible',
        feature, now, lastSent: lastSent.current.get(feature),
      })) return
      lastSent.current.set(feature, now)
      const controller = new AbortController()
      pending.current.add(controller)
      // A 204 has no JSON body. This optional request must also never interrupt
      // work with a connection toast. The server checks consent again on receipt.
      fetchWithCsrfRetry('/api/usage-statistics/activity', {
        method: 'POST', signal: controller.signal,
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCsrfToken() },
        body: JSON.stringify({ feature }),
      }).catch(() => {}).finally(() => pending.current.delete(controller))
    }
    document.addEventListener('pointerdown', onActivity, { passive: true })
    document.addEventListener('keydown', onActivity)
    return () => {
      document.removeEventListener('pointerdown', onActivity)
      document.removeEventListener('keydown', onActivity)
    }
  }, [pathname])

  const value = useMemo(() => ({ state, loading, busy, error, refresh, setEnabled }),
    [state, loading, busy, error, refresh, setEnabled])
  return <UsageStatisticsContext.Provider value={value}>{children}</UsageStatisticsContext.Provider>
}

export function useUsageStatistics() {
  return useContext(UsageStatisticsContext)
}

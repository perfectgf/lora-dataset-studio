/**
 * Capability probes — what's actually configured/reachable right now
 * (GET /api/capabilities). Drives feature gating (e.g. the Studio nav item)
 * and the onboarding redirect when the app has never been configured.
 */
import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import { apiFetch } from '../api/fetchClient'

const CapabilitiesContext = createContext(null)

const EMPTY_CAPS = {
  configured: false,
  engines: { nanobanana: false, chatgpt: false, openrouter: false, klein: false },
  comfyui: { reachable: false, api_url: '', models: {} },
  ollama: { reachable: false, installed: false, binary_path: '', url: '', vision_model: '', vision_model_ready: false },
  aitoolkit: { configured: false, valid: false },
  captioners: { joycaption: false, ollama: false },
  face_scoring: false,
  // Wheel-range verdict for the optional ML extras (insightface/numpy<2 publish
  // nothing outside 3.10–3.12). Defaults to SUPPORTED on purpose: an unknown
  // probe must not hide an install button that would have worked.
  python: { version: '', ml_supported: true, ml_range: '3.10–3.12' },
  masks: false,
  watermark_inpaint: false,
  watermark_allow_crop: true,
  training_visible: false,
  cloud_training: false,
  studio_visible: false,
}

export function CapabilitiesProvider({ children }) {
  const [caps, setCaps] = useState(EMPTY_CAPS)
  const [loading, setLoading] = useState(true)
  // Whether `caps` is the server's answer, at least once. EMPTY_CAPS reads
  // exactly like a machine nobody configured (`configured: false`), and a
  // request that FAILED must never be read that way: a phone coming back to
  // the app on a reconnecting link dropped the first requests of the page
  // load, and a verified install was sent through Setup (2026-09-04).
  const [known, setKnown] = useState(false)

  // Return the fetched snapshot on success and null on failure. Most callers
  // only need the state update, while managed-runtime polling needs the verdict:
  // it must keep retrying if the lightweight probe turned ready but this fuller
  // refresh failed. `options.background` keeps that automatic retry silent.
  const refresh = useCallback(async (force = false, options = {}) => {
    try {
      const data = await apiFetch(
        `/api/capabilities${force ? '?force=1' : ''}`,
        options,
      )
      setKnown(true)
      setCaps(data)
      return data
    } catch {
      // Keep the last-known caps on a transient network error rather than
      // resetting to EMPTY_CAPS — that would bounce the user into onboarding.
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    let alive = true
    ;(async () => {
      if ((await refresh()) !== null || !alive) return
      // One quiet retry: the first request of a page load is the one a
      // reconnecting link drops, and every gated screen reads this answer.
      await new Promise((r) => setTimeout(r, 2500))
      if (alive) refresh(false, { background: true })
    })()
    return () => { alive = false }
  }, [refresh])

  return (
    <CapabilitiesContext.Provider value={{ caps, loading, known, refresh }}>
      {children}
    </CapabilitiesContext.Provider>
  )
}

export function useCapabilities() {
  const ctx = useContext(CapabilitiesContext)
  if (!ctx) throw new Error('useCapabilities must be used within CapabilitiesProvider')
  return ctx
}

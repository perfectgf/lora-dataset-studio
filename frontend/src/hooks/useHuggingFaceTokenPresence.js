import { useEffect, useState } from 'react'
import { apiFetch } from '../api/fetchClient.js'

const EVENT = 'lds:huggingface-token-presence'
const pending = new Map()
const URLS = {
  HF_TOKEN: '/api/settings',
  HF_CLOUD_TOKEN: '/api/settings?plugin=cloud_training',
}

// Several failed runs share one request. Only presence booleans are read or
// broadcast; token values never enter this state or browser storage.
function readPresence(name) {
  if (!pending.has(name)) {
    const request = apiFetch(URLS[name], { background: true }).then((data) => {
      const present = data.secrets?.[name]
      if (typeof present !== 'boolean') throw new Error('Token presence unavailable')
      return present
    }).finally(() => pending.delete(name))
    pending.set(name, request)
  }
  return pending.get(name)
}

export function notifyHuggingFaceTokenPresence(secrets) {
  for (const name of Object.keys(URLS)) {
    if (typeof secrets?.[name] !== 'boolean') continue
    globalThis.dispatchEvent?.(new CustomEvent(EVENT, { detail: { name, present: secrets[name] } }))
  }
}

export default function useHuggingFaceTokenPresence(name) {
  const [status, setStatus] = useState('loading')
  useEffect(() => {
    if (!name) return
    let alive = true
    let revision = 0
    setStatus('loading')
    const refresh = () => {
      const requested = ++revision
      readPresence(name).then((present) => {
        if (alive && requested === revision) setStatus(present ? 'configured' : 'missing')
      }).catch(() => {
        if (alive && requested === revision) setStatus('unknown')
      })
    }
    const saved = (event) => {
      if (event.detail?.name !== name) return
      // A GET started before a save must not restore its stale answer.
      revision += 1
      setStatus(event.detail.present ? 'configured' : 'missing')
    }
    const visible = () => { if (document.visibilityState === 'visible') refresh() }
    window.addEventListener(EVENT, saved)
    window.addEventListener('focus', refresh)
    document.addEventListener('visibilitychange', visible)
    refresh()
    return () => {
      alive = false
      window.removeEventListener(EVENT, saved)
      window.removeEventListener('focus', refresh)
      document.removeEventListener('visibilitychange', visible)
    }
  }, [name])
  return status
}

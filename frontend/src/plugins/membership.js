import { apiFetch } from '../api/fetchClient.js'
import { setMembership } from './registry.js'

export const MEMBERSHIP_CHANGED = 'lds:membership-changed'
let requestVersion = 0

export async function refreshMembership(fetchStatus = () => apiFetch('/api/plugins/membership', {
  cache: 'no-store', background: true,
})) {
  const version = ++requestVersion
  try {
    const value = await fetchStatus()
    if (version === requestVersion) setMembership(value)
  } catch {
    if (version === requestVersion) setMembership({ allowed: false })
  }
}

export function notifyMembershipChanged(revoked = false, win = globalThis.window) {
  // Drop already mounted member screens before any network refresh finishes.
  requestVersion += 1
  if (revoked) setMembership({ allowed: false })
  win?.dispatchEvent(new CustomEvent(MEMBERSHIP_CHANGED, { detail: { revoked } }))
  try { win?.localStorage.setItem(MEMBERSHIP_CHANGED, JSON.stringify({ revoked, nonce: Math.random() })) }
  catch { /* Polling also covers tabs without shared storage. */ }
}

export function watchMembership(win = globalThis.window) {
  if (!win) return () => {}
  let pending = null
  const startRefresh = (force = false) => {
    if (pending && !force) return
    const next = refreshMembership().finally(() => { if (pending === next) pending = null })
    pending = next
  }
  const refresh = () => startRefresh()
  const changed = (event) => {
    requestVersion += 1
    if (event.detail?.revoked) setMembership({ allowed: false })
    startRefresh(true)
  }
  const storage = (event) => {
    if (event.key !== MEMBERSHIP_CHANGED) return
    try { changed({ detail: JSON.parse(event.newValue) }) } catch { refresh() }
  }
  const timer = win.setInterval(refresh, 15000)
  win.addEventListener(MEMBERSHIP_CHANGED, changed)
  win.addEventListener('storage', storage)
  win.addEventListener('focus', refresh)
  refresh()
  return () => {
    win.clearInterval(timer)
    win.removeEventListener(MEMBERSHIP_CHANGED, changed)
    win.removeEventListener('storage', storage)
    win.removeEventListener('focus', refresh)
    requestVersion += 1
  }
}

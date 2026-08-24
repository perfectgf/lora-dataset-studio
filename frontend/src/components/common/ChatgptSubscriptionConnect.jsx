import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { StatusBadge } from '../settings/primitives'
import { connectFeedback } from './chatgptConnectFeedback'

/* The ChatGPT subscription (Codex OAuth) device-code login, as ONE component.
   It lives in common/ because two screens need it: Settings ▸ Image engines and
   the Setup wizard's image step. Setup used to offer only the pay-per-use API
   key, which read as "the ChatGPT engine costs money" on the very screen where a
   Plus/Pro subscriber decides what this app can do — while the subscription lane
   had shipped months earlier, three clicks away, on a page they had no reason to
   open yet.

   Device-code flow: the user opens the verification URL on ANY device, types the
   one-time code, and we poll the backend until it reports connected.

   `label` picks the shape. With one (Setup) the row reads as a plain lane —
   label + inline state on the left, a compact button on the right — so the
   engine's two doors sit at the same visual level as the neighbouring API-key
   fields instead of inside a box of their own. Without one (Settings, where the
   card title already names the lane) it keeps the status badge. */
export default function ChatgptSubscriptionConnect({ caps, refreshCaps, toast, label, description }) {
  const sub = (caps && caps.chatgpt_subscription) || {}
  const [device, setDevice] = useState(null)     // {verification_url, user_code}
  const [busy, setBusy] = useState(false)
  // {action, message} — never a bare string: see connectFeedback.
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!device) return undefined
    const id = setInterval(async () => {
      try {
        const r = await apiFetch('/api/settings/chatgpt-oauth/poll', { background: true })
        if (r.status === 'connected') {
          setDevice(null)
          toast.success('ChatGPT subscription connected.')
          await refreshCaps(true)
        } else if (r.status === 'error') {
          setDevice(null)
          setError({ action: 'connect', message: r.detail || 'Login failed — try again.' })
        }
      } catch { /* transient — keep polling */ }
    }, 3000)
    return () => clearInterval(id)
  }, [device, refreshCaps, toast])

  const run = async (action, fn, done) => {
    setBusy(true); setError(null)
    try {
      const r = await fn()
      if (done) { setDevice(null); done(r) } else { setDevice(r) }
      if (done) await refreshCaps(true)
    } catch (e) {
      setError({ action, message: e.message || 'Request failed.' })
    } finally {
      setBusy(false)
    }
  }

  const start = () => run('connect', () => postJson('/api/settings/chatgpt-oauth/start', {}))
  const importCodex = () => run(
    'import', () => postJson('/api/settings/chatgpt-oauth/import-codex', {}),
    () => toast.success('Codex CLI session imported.'))
  const disconnect = () => run(
    'disconnect', () => postJson('/api/settings/chatgpt-oauth/logout', {}),
    () => toast.success('ChatGPT subscription disconnected.'))

  const btn = 'rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium ' +
    'text-content hover:bg-surface-raised disabled:opacity-50'
  const feedback = connectFeedback(error)

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        {label ? (
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-2">
            <span className="text-sm font-medium text-content">{label}</span>
            <span className={`text-xs ${sub.connected ? 'text-emerald-400' : 'text-content-subtle'}`}>
              {sub.connected ? `✓ ${subscriptionLabel(sub)}` : 'Not connected'}
            </span>
          </div>
        ) : (
          <StatusBadge
            ok={!!sub.connected}
            okLabel={subscriptionLabel(sub)}
            missingLabel="Not connected"
          />
        )}
        <div className="flex flex-wrap gap-2">
          {!sub.connected && (
            <button type="button" onClick={start} disabled={busy || !!device} className={btn}>
              {device
                ? (label ? 'Waiting for the code…' : 'Waiting for you to enter the code…')
                : (label ? 'Connect' : 'Connect ChatGPT subscription')}
            </button>
          )}
          {!sub.connected && sub.codex_cli_detected && (
            <button type="button" onClick={importCodex} disabled={busy || !!device} className={btn}>
              Import from Codex CLI
            </button>
          )}
          {sub.connected && (
            <button type="button" onClick={disconnect} disabled={busy} className={btn}>
              Disconnect
            </button>
          )}
        </div>
      </div>

      {description && <p className="text-xs text-content-muted">{description}</p>}

      {device && (
        <div role="status" className="rounded-lg border border-primary/40 bg-primary/10 p-3 text-sm text-content">
          <p>1. Open <a href={device.verification_url} target="_blank" rel="noreferrer" className="font-medium underline">{device.verification_url}</a> on any device and sign in.</p>
          <p className="mt-1">2. Enter this one-time code (expires in 15 minutes):</p>
          <p className="mt-1 select-all font-mono text-lg font-semibold tracking-widest">{device.user_code}</p>
        </div>
      )}

      {feedback && <p className="text-xs text-rose-400"><span aria-hidden="true">✗</span> {feedback}</p>}
    </div>
  )
}

/* What a connected subscription is CALLED on screen. The plan is worth naming —
   the lane's limits (reference-image count, daily image cap) follow the plan, not
   the account — but neither field is guaranteed by the token, so each one only
   appears when the backend actually reported it. */
function subscriptionLabel(sub) {
  const s = sub || {}
  const parts = [s.email, s.plan].filter((v) => typeof v === 'string' && v.trim())
  return parts.length ? `Connected — ${parts.join(' · ')}` : 'Connected'
}

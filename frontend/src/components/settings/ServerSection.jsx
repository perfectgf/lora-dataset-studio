import { useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INPUT_CLASS, Card } from './primitives'

const LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1']

/* Server bind (host/port/LAN access). host/port live in config.server and are only
   read by run.py at PROCESS START — Flask can't rebind mid-request — so this card
   contrasts the SAVED config against `runtime` (what's actually bound right now,
   stamped by run.py) and offers a one-click save-then-restart, mirroring
   UpdatesCard's "poll /api/health, then hard-reload" pattern. */
export default function ServerSection({ config, setField, runtime, handleSave }) {
  const toast = useToast()
  const [restarting, setRestarting] = useState(false)
  const [copied, setCopied] = useState(false)
  const lan = !LOOPBACK_HOSTS.includes(config.server.host)
  const knownRuntime = runtime.host != null && runtime.port != null
  const dirty = knownRuntime && (runtime.host !== config.server.host || runtime.port !== config.server.port)

  const waitForHealthAndReload = async () => {
    for (let i = 0; i < 120; i += 1) {
      await new Promise((r) => setTimeout(r, 1000))
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (res.ok) { window.location.reload(); return }
      } catch { /* still restarting — keep waiting */ }
    }
    setRestarting(false)   // gave up after ~2 min
  }

  const restart = async () => {
    setRestarting(true)
    // Save first: "Restart to apply" must apply what's on screen, not whatever
    // was last persisted — otherwise a restart right after editing the port
    // would silently come back on the OLD port with no visible change.
    if (!(await handleSave())) { setRestarting(false); return }
    try {
      await postJson('/api/settings/restart', {})
      waitForHealthAndReload()
    } catch (e) {
      toast.error(e.message || 'Restart failed')
      setRestarting(false)
    }
  }

  const regenerateToken = () => {
    const bytes = crypto.getRandomValues(new Uint8Array(24))
    const token = btoa(String.fromCharCode(...bytes)).replace(/[+/=]/g, '')
    setField('server', 'access_token', token)
  }

  const copyToken = async () => {
    try {
      await navigator.clipboard.writeText(config.server.access_token || '')
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard unavailable (non-HTTPS remote origin) — token stays selectable */ }
  }

  return (
    <Card title="Server"
      help="Where the app listens. Host/port and LAN access need a restart to take effect — edit below, then use “Restart to apply”.">
      <div>
        <label htmlFor="server-port" className="block text-sm font-medium text-content">Port</label>
        <input id="server-port" type="number" min={1} max={65535}
          value={config.server.port ?? ''}
          onChange={(e) => setField('server', 'port', Math.max(1, Math.min(65535, Number(e.target.value) || 1)))}
          className={`${INPUT_CLASS} max-w-[8rem]`} />
      </div>

      <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
        <div>
          <p className="text-sm font-medium text-content">Available on the local network</p>
          <p className="mt-0.5 text-xs text-content-muted">
            Off (default): only this computer can open the app. On: any device on your
            Wi-Fi/LAN can reach it — e.g. from your phone — guarded by the access token below.
          </p>
        </div>
        <button type="button" role="switch" aria-checked={lan}
          onClick={() => setField('server', 'host', lan ? '127.0.0.1' : '0.0.0.0')}
          aria-label="Available on the local network"
          className={`relative h-6 w-11 shrink-0 rounded-full transition-colors ${lan ? 'bg-emerald-500' : 'bg-surface border border-border-strong'}`}>
          <span aria-hidden
            className={`absolute top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${lan ? 'translate-x-5' : 'translate-x-0.5'}`} />
        </button>
      </div>

      {lan && (
        <div>
          <div className="flex items-center justify-between">
            <label htmlFor="server-token" className="block text-sm font-medium text-content">Access token</label>
            <button type="button" onClick={regenerateToken}
              className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
              Generate new token
            </button>
          </div>
          <p className="mb-1 text-xs text-content-muted">
            Remote devices present this once (via the URL below) — a signed session cookie
            takes over from there. Requests from this computer never need it.
          </p>
          <div className="flex gap-2">
            <input id="server-token" type="text" readOnly
              value={config.server.access_token || '(created automatically the first time LAN access is used — or click “Generate new token”)'}
              className={`${INPUT_CLASS} font-mono text-xs`} />
            {config.server.access_token && (
              <button type="button" onClick={copyToken}
                className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised">
                {copied ? 'Copied ✓' : 'Copy'}
              </button>
            )}
          </div>
          {config.server.access_token && (
            <p className="mt-1 break-all text-xs text-content-subtle">
              From another device: <code className="text-content">http://&lt;this-computer&gt;:{config.server.port}/?token={config.server.access_token}</code>
            </p>
          )}
        </div>
      )}

      {knownRuntime && (
        <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2 text-xs ${
          dirty ? 'border-amber-400/50 bg-amber-400/10' : 'border-border bg-surface-raised'}`}>
          <span className="text-content-muted">
            Running: <span className="font-medium text-content">{runtime.host}:{runtime.port}</span>
            {dirty && (
              <> · Saved: <span className="font-medium text-content">{config.server.host}:{config.server.port}</span></>
            )}
          </span>
          {dirty ? (
            <button type="button" onClick={restart} disabled={restarting}
              className="ml-auto shrink-0 rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
              {restarting ? '↻ Restarting…' : 'Save & restart to apply'}
            </button>
          ) : (
            <span className="ml-auto text-emerald-400"><span aria-hidden>✓</span> Running config matches saved config</span>
          )}
        </div>
      )}
    </Card>
  )
}

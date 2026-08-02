import { useState } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INPUT_CLASS, Card } from './primitives'
import ResetToDefault from './ResetToDefault'

/* managed-access:start -- kept dependency-free so its real production logic can
   be exercised by the Node test suite without needing to transpile this JSX file. */
const LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1']

export function managedBrowserAccess(origin) {
  if (typeof origin !== 'string' || !origin.trim() || origin === 'null') return null
  try {
    const parsed = new URL(origin)
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return null
    const hostname = parsed.hostname.replace(/^\[|\]$/g, '').toLowerCase().replace(/\.$/, '')
    // WHATWG URL canonicalises IPv4-mapped IPv6 literals such as
    // ::ffff:127.42.1.2 to ::ffff:7f2a:102. In that form, 127/8 is the
    // 7f00-7fff range in the penultimate hextet.
    const mappedIpv4Loopback = /^::ffff:7f[0-9a-f]{2}:[0-9a-f]{1,4}$/.test(hostname)
    const loopback = hostname === 'localhost'
      || hostname.endsWith('.localhost')
      || hostname === '::1'
      || mappedIpv4Loopback
      || /^127(?:\.\d{1,3}){3}$/.test(hostname)
      || hostname === '0.0.0.0'
      || hostname === '::'
    return {
      origin: parsed.origin,
      hostname,
      port: Number(parsed.port || (parsed.protocol === 'https:' ? 443 : 80)),
      lan: !loopback,
    }
  } catch {
    return null
  }
}

export function resolveServerAccess({ bindManaged, browserOrigin, configHost, configPort, runtimeLanIp }) {
  if (bindManaged) {
    const managed = managedBrowserAccess(browserOrigin)
    return {
      origin: managed?.origin || null,
      host: managed?.hostname || null,
      port: managed?.port ?? null,
      lan: managed?.lan === true,
      lanIp: null,
    }
  }
  return {
    origin: null,
    host: configHost,
    port: configPort,
    lan: !LOOPBACK_HOSTS.includes(configHost),
    lanIp: runtimeLanIp || null,
  }
}

export function shouldShowRemoteControls(bindManaged, lan) {
  return bindManaged || lan
}
/* managed-access:end */

/* Server bind (host/port/LAN access). host/port live in config.server and are only
   read by run.py at PROCESS START — Flask can't rebind mid-request — so this card
   contrasts the SAVED config against `runtime` (what's actually bound right now,
   stamped by run.py) and offers a one-click save-then-restart, mirroring
   UpdatesCard's "poll /api/health, then hard-reload" pattern. */
export default function ServerSection({ config, setField, runtime, handleSave, configDefaults }) {
  const toast = useToast()
  const [restarting, setRestarting] = useState(false)
  const [copied, setCopied] = useState(false)
  const [copiedUrl, setCopiedUrl] = useState(null)   // which reach-URL was just copied (by key)
  const bindManaged = runtime.bind_managed === true
  const access = resolveServerAccess({
    bindManaged,
    browserOrigin: typeof window === 'undefined' ? null : window.location?.origin,
    configHost: config.server.host,
    configPort: config.server.port,
    runtimeLanIp: runtime.lan_ip,
  })
  const lan = access.lan
  const showRemoteControls = shouldShowRemoteControls(bindManaged, lan)
  const requireToken = !!config.server.require_token
  const publicBind = runtime.public === true
  // Real LAN IPv4 of this machine (backend socket probe), so the remote-access
  // URL is copyable as-is instead of a <this-computer> placeholder. null when the
  // backend couldn't determine it (offline / loopback-only) -> keep the placeholder.
  const lanIp = access.lanIp
  const tsIp = runtime.tailscale_ip || null
  const knownRuntime = bindManaged ? !!access.origin : runtime.host != null && runtime.port != null
  const dirty = !bindManaged && knownRuntime
    && (runtime.host !== config.server.host || runtime.port !== config.server.port)

  // The exact URL(s) a phone should open. Token is appended ONLY when the token
  // gate is on (a tokenless URL would 403); when it's on but no token exists yet,
  // reachUrls stays empty and the card asks the user to generate one first.
  const port = access.port
  const token = (requireToken || publicBind) ? (config.server.access_token || '') : ''
  const tokenReady = !(requireToken || publicBind) || !!token
  const tokenQS = token ? `?token=${token}` : ''
  const reachUrls = tokenReady ? (bindManaged ? [
    lan && access.origin && {
      key: 'current',
      label: 'Current browser address',
      url: `${access.origin}/${tokenQS}`,
    },
  ] : [
    lanIp && { key: 'lan', label: 'Same Wi-Fi / LAN', url: `http://${lanIp}:${port}/${tokenQS}` },
    tsIp && { key: 'ts', label: 'From anywhere · Tailscale', url: `http://${tsIp}:${port}/${tokenQS}` },
  ]).filter(Boolean) : []
  const qrUrl = reachUrls[0]?.url || null

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
    if (bindManaged) return
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

  const copyUrl = async (key, url) => {
    try {
      await navigator.clipboard.writeText(url)
      setCopiedUrl(key)
      setTimeout(() => setCopiedUrl(null), 1500)
    } catch { /* clipboard unavailable (non-HTTPS remote origin) — URL stays selectable */ }
  }

  return (
    <Card title="Server"
      help={bindManaged
        ? 'Docker Compose manages the host and port; change the host-side mapping and recreate the container.'
        : 'Where the app listens. Host/port and LAN access need a restart to take effect — edit below, then use “Restart to apply”.'}>
      {bindManaged && (
        <div id="server-bind-managed-note" role="note"
          className="rounded-lg border border-sky-400/30 bg-sky-500/10 px-3 py-2.5 text-sm text-content">
          <p className="font-medium">Host and port are managed by Docker Compose.</p>
          <p className="mt-1 text-xs text-content-muted">
            Set <code className="text-content">LDS_HOST_PORT</code> in the host checkout&apos;s{' '}
            <code className="text-content">.env</code> (for example <code className="text-content">127.0.0.1:5050</code>),
            then recreate the container:
          </p>
          <code className="mt-2 block overflow-x-auto whitespace-nowrap rounded bg-app/70 px-2 py-1.5 text-xs text-content">
            docker compose -f docker-compose.gpu.yml up -d --force-recreate
          </code>
        </div>
      )}
      <div>
        <label htmlFor="server-port" className="block text-sm font-medium text-content">Port</label>
        <input id="server-port" type="number" min={1} max={65535}
          value={port ?? ''}
          disabled={bindManaged}
          aria-describedby={bindManaged ? 'server-bind-managed-note' : undefined}
          onChange={(e) => setField('server', 'port', Math.max(1, Math.min(65535, Number(e.target.value) || 1)))}
          className={`${INPUT_CLASS} max-w-[8rem] disabled:cursor-not-allowed disabled:opacity-60`} />
        {/* The shipped port is 5050 and nothing on this screen said so — a user
            who typed 8080 to test something had no way back to the value
            start.bat binds by default. */}
        {!bindManaged && (
          <ResetToDefault label="Port" section="server" field="port"
            config={config} configDefaults={configDefaults} setField={setField} />
        )}
      </div>

      <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
        <div>
          <p className="text-sm font-medium text-content">
            {bindManaged ? 'Current browser address uses a network host' : 'Available on the local network'}
          </p>
          <p className="mt-0.5 text-xs text-content-muted">
            {bindManaged
              ? (lan
                ? 'On because this browser is using a non-loopback address. Docker and network rules still determine which other devices can reach it.'
                : 'Off because this browser is using a loopback address. That does not reveal whether Docker is exposed; open the app through the host’s LAN address to get a shareable link and QR code.')
              : 'Off (default): only this computer can open the app. On: any device on your Wi-Fi/LAN can reach it — e.g. from your phone — using the plain URL below.'}
          </p>
        </div>
        <button id="server-lan" type="button" role="switch" aria-checked={lan}
          data-focus-gate="server-require-token server-token"
          disabled={bindManaged}
          aria-describedby={bindManaged ? 'server-bind-managed-note' : undefined}
          onClick={() => setField('server', 'host', lan ? '127.0.0.1' : '0.0.0.0')}
          aria-label={bindManaged ? 'Current browser address uses a network host' : 'Available on the local network'}
          className={`relative h-6 w-11 shrink-0 scroll-mt-24 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${lan ? 'bg-emerald-500' : 'bg-surface border border-border-strong'}`}>
          <span aria-hidden
            className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform ${lan ? 'translate-x-5' : 'translate-x-0'}`} />
        </button>
      </div>

      {showRemoteControls && (
        <>
          {/* Trusted-LAN default: no token to type on a phone. The token is an
              opt-in extra layer, off by default (see backend server.require_token). */}
          <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
            <div>
              <p className="text-sm font-medium text-content">Require an access token</p>
              <p className="mt-0.5 text-xs text-content-muted">
                {bindManaged
                  ? (requireToken
                    ? 'On: devices that can reach this Docker service must open a tokenized URL once (a session cookie takes over after).'
                    : 'Off: any device that can reach this Docker service can open it with no password. Docker and network rules determine whether it is exposed.')
                  : (requireToken
                    ? 'On: remote devices must open the URL WITH the token once (a session cookie takes over after). Extra safety on a shared or untrusted network.'
                    : 'Off (default): anyone on your Wi-Fi/LAN can open the app with no password. Fine for a home network; turn on if the network is shared or untrusted.')}
              </p>
              {publicBind && (
                <p className="text-content-subtle text-sm mt-1">
                  Forced on: this app is reachable from the internet
                  (<code>LDS_PUBLIC=1</code>), so the token gate can’t be turned off.
                </p>
              )}
            </div>
            <button id="server-require-token" type="button" role="switch" aria-checked={requireToken}
              data-focus-gate="server-token"
              disabled={publicBind}
              onClick={() => { if (!publicBind) setField('server', 'require_token', !requireToken) }}
              aria-label="Require an access token"
              className={`relative h-6 w-11 shrink-0 scroll-mt-24 rounded-full transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${requireToken ? 'bg-emerald-500' : 'bg-surface border border-border-strong'}`}>
              <span aria-hidden
                className={`absolute top-0.5 left-0.5 h-5 w-5 rounded-full bg-white transition-transform ${requireToken ? 'translate-x-5' : 'translate-x-0'}`} />
            </button>
          </div>

          {(requireToken || publicBind) && (
            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="server-token" className="block text-sm font-medium text-content">Access token</label>
                <button type="button" onClick={regenerateToken}
                  className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
                  Generate new token
                </button>
              </div>
              <p className="mb-1 text-xs text-content-muted">
                Remote devices present this once (baked into the link below) — a signed session
                cookie takes over from there. Requests from this computer never need it.
              </p>
              <div className="flex gap-2">
                <input id="server-token" type="text" readOnly
                  value={config.server.access_token || '(created automatically on the next restart — or click “Generate new token”)'}
                  className={`${INPUT_CLASS} font-mono text-xs`} />
                {config.server.access_token && (
                  <button type="button" onClick={copyToken}
                    className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised">
                    {copied ? 'Copied ✓' : 'Copy'}
                  </button>
                )}
              </div>
            </div>
          )}

          {lan && (
            /* Open it on your phone: scannable QR + copyable URLs, detected from
               the machine's real addresses — no more guessing which IP/port. */
            <div className="rounded-lg border border-border bg-surface-raised px-3 py-3">
              <p className="text-sm font-medium text-content">Open it on your phone</p>
              {reachUrls.length > 0 ? (
                <div className="mt-2 flex items-start gap-4">
                  {qrUrl && (
                    <div className="shrink-0 rounded-md bg-white p-2" title={qrUrl}>
                      <QRCodeSVG value={qrUrl} size={128} level="M" marginSize={2} />
                    </div>
                  )}
                  <div className="min-w-0 flex-1 space-y-2">
                    <p className="text-xs text-content-muted">
                      {bindManaged
                        ? 'Open the same address below from another device that can reach this Docker host.'
                        : 'Point your phone camera at the code — or open a link below. The LAN link needs the phone on the same Wi-Fi; the Tailscale link works from anywhere.'}
                    </p>
                    {reachUrls.map((u) => (
                      <div key={u.key} className="flex items-center gap-2">
                        <div className="min-w-0 flex-1">
                          <p className="text-[11px] uppercase tracking-wide text-content-subtle">{u.label}</p>
                          <code className="block truncate text-xs text-content">{u.url}</code>
                        </div>
                        <button type="button" onClick={() => copyUrl(u.key, u.url)}
                          className="shrink-0 rounded-md border border-border-strong px-2 py-0.5 text-xs font-medium text-content hover:bg-surface-raised">
                          {copiedUrl === u.key ? 'Copied ✓' : 'Copy'}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (requireToken || publicBind) && !token ? (
                <p className="mt-1 text-xs text-content-subtle">
                  Turn the token on, then <span className="text-content">Generate new token</span> (or
                  Save &amp; restart) — the scannable link appears once a token exists.
                </p>
              ) : (
                <p className="mt-1 break-all text-xs text-content-subtle">
                  Couldn’t detect this machine’s address. From another device open{' '}
                  <code className="text-content">http://&lt;this-computer&apos;s LAN IP&gt;:{port}/</code>{' '}
                  (find the IP by running <code className="text-content">ipconfig</code>).
                </p>
              )}
            </div>
          )}
        </>
      )}

      {knownRuntime && (
        <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2 text-xs ${
          dirty ? 'border-amber-400/50 bg-amber-400/10' : 'border-border bg-surface-raised'}`}>
          {bindManaged ? (
            <span className="text-content-muted">
              Opened at: <span className="font-medium text-content">{access.origin}</span>
            </span>
          ) : (
            <span className="text-content-muted">
              Running: <span className="font-medium text-content">{runtime.host}:{runtime.port}</span>
              {runtime.host === '0.0.0.0' && lanIp && (
                <span className="text-content-subtle"> — reachable at http://{lanIp}:{runtime.port}/</span>
              )}
              {dirty && (
                <> · Saved: <span className="font-medium text-content">{config.server.host}:{config.server.port}</span></>
              )}
            </span>
          )}
          {bindManaged ? (
            <span className="ml-auto text-sky-300">Docker-managed bind</span>
          ) : dirty ? (
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

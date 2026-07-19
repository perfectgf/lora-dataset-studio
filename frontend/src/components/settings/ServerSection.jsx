import { useState } from 'react'
import { QRCodeSVG } from 'qrcode.react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INPUT_CLASS, Card } from './primitives'
import { useI18n } from '../../i18n/I18nContext'

const LOOPBACK_HOSTS = ['127.0.0.1', 'localhost', '::1']

/* Server bind (host/port/LAN access). host/port live in config.server and are only
   read by run.py at PROCESS START — Flask can't rebind mid-request — so this card
   contrasts the SAVED config against `runtime` (what's actually bound right now,
   stamped by run.py) and offers a one-click save-then-restart, mirroring
   UpdatesCard's "poll /api/health, then hard-reload" pattern. */
export default function ServerSection({ config, setField, runtime, handleSave }) {
  const { t } = useI18n()
  const toast = useToast()
  const [restarting, setRestarting] = useState(false)
  const [copied, setCopied] = useState(false)
  const [copiedUrl, setCopiedUrl] = useState(null)   // which reach-URL was just copied (by key)
  const lan = !LOOPBACK_HOSTS.includes(config.server.host)
  const requireToken = !!config.server.require_token
  // Real LAN IPv4 of this machine (backend socket probe), so the remote-access
  // URL is copyable as-is instead of a <this-computer> placeholder. null when the
  // backend couldn't determine it (offline / loopback-only) -> keep the placeholder.
  const lanIp = runtime.lan_ip || null
  const tsIp = runtime.tailscale_ip || null
  const knownRuntime = runtime.host != null && runtime.port != null
  const dirty = knownRuntime && (runtime.host !== config.server.host || runtime.port !== config.server.port)

  // The exact URL(s) a phone should open. Token is appended ONLY when the token
  // gate is on (a tokenless URL would 403); when it's on but no token exists yet,
  // reachUrls stays empty and the card asks the user to generate one first.
  const port = config.server.port
  const token = requireToken ? (config.server.access_token || '') : ''
  const tokenReady = !requireToken || !!token
  const tokenQS = token ? `?token=${token}` : ''
  const reachUrls = tokenReady ? [
    lanIp && { key: 'lan', label: t('settings.server.sameLan'), url: `http://${lanIp}:${port}/${tokenQS}` },
    tsIp && { key: 'ts', label: t('settings.server.tailscale'), url: `http://${tsIp}:${port}/${tokenQS}` },
  ].filter(Boolean) : []
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
    setRestarting(true)
    // Save first: "Restart to apply" must apply what's on screen, not whatever
    // was last persisted — otherwise a restart right after editing the port
    // would silently come back on the OLD port with no visible change.
    if (!(await handleSave())) { setRestarting(false); return }
    try {
      await postJson('/api/settings/restart', {})
      waitForHealthAndReload()
    } catch (e) {
      toast.error(e.message || t('settings.server.restartFailed'))
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
    <Card title={t('settings.server.title')} help={t('settings.server.help')}>
      <div>
        <label htmlFor="server-port" className="block text-sm font-medium text-content">
          {t('settings.server.port')}
        </label>
        <input id="server-port" type="number" min={1} max={65535}
          value={config.server.port ?? ''}
          onChange={(e) => setField('server', 'port', Math.max(1, Math.min(65535, Number(e.target.value) || 1)))}
          className={`${INPUT_CLASS} max-w-[8rem]`} />
      </div>

      <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
        <div>
          <p className="text-sm font-medium text-content">{t('settings.server.lanAccess')}</p>
          <p className="mt-0.5 text-xs text-content-muted">
            {t('settings.server.lanAccessHelp')}
          </p>
        </div>
        <button id="server-lan" type="button" role="switch" aria-checked={lan}
          data-focus-gate="server-require-token server-token"
          onClick={() => setField('server', 'host', lan ? '127.0.0.1' : '0.0.0.0')}
          aria-label={t('settings.server.lanAccess')}
          className={`relative h-6 w-11 shrink-0 scroll-mt-24 rounded-full transition-colors ${lan ? 'bg-emerald-500' : 'bg-surface ring-1 ring-inset ring-border-strong'}`}>
          <span aria-hidden
            className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${lan ? 'translate-x-5' : 'translate-x-0'}`} />
        </button>
      </div>

      {lan && (
        <>
          {/* Trusted-LAN default: no token to type on a phone. The token is an
              opt-in extra layer, off by default (see backend server.require_token). */}
          <div className="flex items-start justify-between gap-4 rounded-lg border border-border bg-surface-raised px-3 py-2.5">
            <div>
              <p className="text-sm font-medium text-content">{t('settings.server.requireToken')}</p>
              <p className="mt-0.5 text-xs text-content-muted">
                {requireToken
                  ? t('settings.server.requireTokenOn')
                  : t('settings.server.requireTokenOff')}
              </p>
            </div>
            <button id="server-require-token" type="button" role="switch" aria-checked={requireToken}
              data-focus-gate="server-token"
              onClick={() => setField('server', 'require_token', !requireToken)}
              aria-label={t('settings.server.requireToken')}
              className={`relative h-6 w-11 shrink-0 scroll-mt-24 rounded-full transition-colors ${requireToken ? 'bg-emerald-500' : 'bg-surface ring-1 ring-inset ring-border-strong'}`}>
              <span aria-hidden
                className={`absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white transition-transform ${requireToken ? 'translate-x-5' : 'translate-x-0'}`} />
            </button>
          </div>

          {requireToken && (
            <div>
              <div className="flex items-center justify-between">
                <label htmlFor="server-token" className="block text-sm font-medium text-content">
                  {t('settings.server.accessToken')}
                </label>
                <button type="button" onClick={regenerateToken}
                  className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
                  {t('settings.server.generateToken')}
                </button>
              </div>
              <p className="mb-1 text-xs text-content-muted">
                {t('settings.server.accessTokenHelp')}
              </p>
              <div className="flex gap-2">
                <input id="server-token" type="text" readOnly
                  value={config.server.access_token || t('settings.server.tokenPlaceholder')}
                  className={`${INPUT_CLASS} font-mono text-xs`} />
                {config.server.access_token && (
                  <button type="button" onClick={copyToken}
                    className="shrink-0 rounded-md border border-border-strong px-3 py-1.5 text-xs font-medium text-content hover:bg-surface-raised">
                    {copied ? t('common.copiedCheck') : t('common.copy')}
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Open it on your phone: scannable QR + copyable URLs, detected from
              the machine's real addresses — no more guessing which IP/port. */}
          <div className="rounded-lg border border-border bg-surface-raised px-3 py-3">
            <p className="text-sm font-medium text-content">{t('settings.server.phoneTitle')}</p>
            {reachUrls.length > 0 ? (
              <div className="mt-2 flex items-start gap-4">
                {qrUrl && (
                  <div className="shrink-0 rounded-md bg-white p-2" title={qrUrl}>
                    <QRCodeSVG value={qrUrl} size={128} level="M" marginSize={2} />
                  </div>
                )}
                <div className="min-w-0 flex-1 space-y-2">
                  <p className="text-xs text-content-muted">
                    {t('settings.server.phoneHelp')}
                  </p>
                  {reachUrls.map((u) => (
                    <div key={u.key} className="flex items-center gap-2">
                      <div className="min-w-0 flex-1">
                        <p className="text-[11px] uppercase tracking-wide text-content-subtle">{u.label}</p>
                        <code className="block truncate text-xs text-content">{u.url}</code>
                      </div>
                      <button type="button" onClick={() => copyUrl(u.key, u.url)}
                        className="shrink-0 rounded-md border border-border-strong px-2 py-0.5 text-xs font-medium text-content hover:bg-surface-raised">
                        {copiedUrl === u.key ? t('common.copiedCheck') : t('common.copy')}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : requireToken && !token ? (
              <p className="mt-1 text-xs text-content-subtle">
                {t('settings.server.tokenNeededBefore')}{' '}
                <span className="text-content">{t('settings.server.generateToken')}</span>
                {t('settings.server.tokenNeededAfter')}
              </p>
            ) : (
              <p className="mt-1 break-all text-xs text-content-subtle">
                {t('settings.server.addressMissingBefore')}{' '}
                <code className="text-content">
                  {`http://<${t('settings.server.lanIpPlaceholder')}>:${port}/`}
                </code>{' '}
                {t('settings.server.addressMissingAfter')}{' '}
                <code className="text-content">ipconfig</code>。
              </p>
            )}
          </div>
        </>
      )}

      {knownRuntime && (
        <div className={`flex flex-wrap items-center gap-3 rounded-lg border px-3 py-2 text-xs ${
          dirty ? 'border-amber-400/50 bg-amber-400/10' : 'border-border bg-surface-raised'}`}>
          <span className="text-content-muted">
            {t('settings.server.running')} <span className="font-medium text-content">{runtime.host}:{runtime.port}</span>
            {runtime.host === '0.0.0.0' && lanIp && (
              <span className="text-content-subtle">
                {' '}{t('settings.server.reachableAt', { url: `http://${lanIp}:${runtime.port}/` })}
              </span>
            )}
            {dirty && (
              <> · {t('settings.server.saved')} <span className="font-medium text-content">{config.server.host}:{config.server.port}</span></>
            )}
          </span>
          {dirty ? (
            <button type="button" onClick={restart} disabled={restarting}
              className="ml-auto shrink-0 rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
              {restarting ? t('settings.server.restarting') : t('settings.server.saveRestart')}
            </button>
          ) : (
            <span className="ml-auto text-emerald-400">
              <span aria-hidden>✓</span> {t('settings.server.configMatches')}
            </span>
          )}
        </div>
      )}
    </Card>
  )
}

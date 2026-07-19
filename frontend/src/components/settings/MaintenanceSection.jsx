import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import DiagnosticReport from '../common/DiagnosticReport'
import { Card, TextField } from './primitives'
import { installMode, zipUpdateHeadline, progressLabel, progressPercent } from './updateStatus'
import { useI18n } from '../../i18n/I18nContext'

/* In-app updater: "Check for updates" hits the git-aware check (commits-behind for a
   clone, release tag for a packaged build). "Update & restart" pulls (git) or downloads
   and swaps in the latest release ZIP (packaged install), then restarts the server; we
   poll /api/health until the relaunched process answers and hard-reload the SPA so the
   new frontend/dist loads. */
function UpdatesCard() {
  const { t } = useI18n()
  const [status, setStatus] = useState(null)
  const [checking, setChecking] = useState(false)
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'
  const [progress, setProgress] = useState(null)   // ZIP mode: {phase, downloaded, total}

  // Passive check on mount (cached server-side, no git fetch): the card shows
  // the current build immediately instead of waiting for a manual check.
  useEffect(() => {
    let alive = true
    apiFetch('/api/update/check')
      .then((d) => { if (alive) setStatus((prev) => prev || d) })
      .catch(() => { /* best-effort — the manual button stays available */ })
    return () => { alive = false }
  }, [])

  const check = async () => {
    setChecking(true)
    try {
      setStatus(await apiFetch('/api/update/check?force=1'))
    } catch (e) {
      setStatus({ ok: false, reason: e.message || t('settings.maintenance.checkFailed') })
    } finally {
      setChecking(false)
    }
  }

  const waitForHealthAndReload = async () => {
    // The server is re-execing: /api/health refuses connections for a few seconds,
    // then answers again on the same port. Poll, then hard-reload to pull new dist.
    for (let i = 0; i < 120; i += 1) {
      await new Promise((r) => setTimeout(r, 1000))
      try {
        const res = await fetch('/api/health', { cache: 'no-store' })
        if (res.ok) { window.location.reload(); return }
      } catch { /* still down — keep waiting */ }
    }
    setApplying(false); setPhase('')          // gave up after ~2 min
  }

  // Packaged (ZIP) installs download+swap the release (with a progress bar); a git
  // clone fast-forwards. 'unavailable' = non-git with no downloadable release.
  const mode = installMode(status)

  // ZIP mode: poll the server's progress until it restarts / finishes / fails.
  // A release ZIP is tens of MB, so the user needs to see it advancing.
  const pollProgress = async () => {
    for (let i = 0; i < 1200; i += 1) {       // ~10 min ceiling at 500 ms
      await new Promise((r) => setTimeout(r, 500))
      let p
      try { p = await apiFetch('/api/update/progress') } catch { continue }
      setProgress(p)
      if (p.phase === 'restarting') { setPhase('restarting'); waitForHealthAndReload(); return }
      if (p.phase === 'error') {
        setStatus({ ok: false, reason: p.error || 'Update failed and was rolled back.' })
        setApplying(false); setPhase(''); setProgress(null); return
      }
      if (p.phase === 'done') {               // server decided it was already up to date
        setStatus({ ...status, up_to_date: true }); setApplying(false); setPhase(''); setProgress(null); return
      }
    }
    setApplying(false); setPhase(''); setProgress(null)   // gave up
  }

  const apply = async () => {
    setApplying(true); setPhase('pulling'); setProgress(null)
    try {
      const res = await postJson('/api/update/apply', {})
      if (res.restarting) {                   // git path: synchronous restart
        setPhase('restarting')
        waitForHealthAndReload()              // not awaited: UI shows "restarting…"
      } else if (res.async) {                 // ZIP path: download+swap on the server, poll it
        setProgress({ phase: 'downloading', downloaded: 0, total: res.total || 0 })
        pollProgress()                        // not awaited
      } else {                                // up to date / manual / error, inline
        setStatus(res.ok ? { ...res, up_to_date: true } : res)
        setApplying(false); setPhase('')
      }
    } catch (e) {
      setStatus({ ok: false, reason: e.message || t('settings.maintenance.updateFailed') })
      setApplying(false); setPhase('')
    }
  }

  const s = status
  // In-app update is possible for a git clone (pull) or a packaged install whose
  // latest release ships a ZIP asset (download + swap). Otherwise: link out.
  const canPull = s && s.update_available && (mode === 'git' || mode === 'zip')
  return (
    <Card title={t('settings.maintenance.updatesTitle')} help={t('settings.maintenance.updatesHelp')}>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={check} disabled={checking || applying}
          className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
          {checking ? t('settings.maintenance.checking') : t('settings.maintenance.checkUpdates')}
        </button>
        {s?.current && (
          <span className="text-xs text-content-subtle">
            {t('settings.maintenance.currentBuild')}{' '}
            <span className="font-medium text-content">v{s.current}{s.current_sha ? ` (${s.current_sha})` : ''}</span>
          </span>
        )}
        {s && (
          <span className="text-xs text-content-subtle">
            {t('settings.maintenance.latestBuild')}{' '}
            <span className="font-medium text-content">
              {s.remote_sha
                ? `${s.remote_sha}${typeof s.behind === 'number' && s.behind > 0
                  ? ` (${t('settings.maintenance.commitsAhead', { count: s.behind })})`
                  : ''}`
                : s.latest ? `v${s.latest}`
                : s.update_available ? t('settings.maintenance.updateAvailable')
                : t('settings.maintenance.pressCheck')}
            </span>
          </span>
        )}
        {/* Read WHAT the update contains before pulling: the compare view lists
            exactly the incoming commits; otherwise the branch history. Only
            present after a git-aware "Check for updates" (force). */}
        {s && (s.compare_url || s.commits_url) && (
          <a href={s.compare_url || s.commits_url} target="_blank" rel="noreferrer"
            className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
            {s.compare_url ? t('settings.maintenance.seeUpdate') : t('settings.maintenance.browseCommits')}
          </a>
        )}
      </div>

      {applying && (
        <div className="space-y-1.5" role="status" aria-live="polite">
          <p className="text-sm text-content-muted">
            {phase === 'restarting'
              ? t('settings.maintenance.updatedRestarting')
              : progressLabel(progress, t) || (mode === 'zip'
                ? t('settings.maintenance.downloadingInstalling')
                : t('settings.maintenance.pulling'))}
          </p>
          {/* Real progress bar while downloading a release ZIP (indeterminate when
              the server reported no Content-Length). Git pulls stay text-only. */}
          {phase !== 'restarting' && progress && progress.phase === 'downloading' && (
            <div className="h-1.5 w-full max-w-xs overflow-hidden rounded-full bg-surface-raised">
              <div className="h-full rounded-full bg-gradient-primary transition-[width] duration-300"
                style={{ width: `${progressPercent(progress) ?? 40}%` }} />
            </div>
          )}
        </div>
      )}

      {!applying && s && (
        <div className="text-sm">
          {canPull ? (
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-content">
                <span aria-hidden>⬆</span>{' '}
                {typeof s.behind === 'number'
                  ? t('settings.maintenance.behind', {
                    count: s.behind,
                    refs: s.current_sha && s.remote_sha ? ` (${s.current_sha} → ${s.remote_sha})` : '',
                  })
                  : `${zipUpdateHeadline(s, t)}.`}
              </span>
              <button type="button" onClick={apply}
                className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white transition-transform hover:-translate-y-px">
                {t('settings.maintenance.updateRestart')}
              </button>
            </div>
          ) : s.update_available ? (
            <p className="text-content">
              {t('settings.maintenance.availableVersion', { version: s.latest ? ` — v${s.latest}` : '' })}{' '}
              <a href={s.url} target="_blank" rel="noreferrer" className="font-semibold text-emerald-300 underline">
                {t('settings.maintenance.downloadRelease')}
              </a>{' '}{t('settings.maintenance.replaceFolder')}
            </p>
          ) : s.ok ? (
            <p className="text-emerald-400"><span aria-hidden>✓</span> {t('settings.maintenance.upToDate')}</p>
          ) : (
            <p className="text-content-muted">
              <span aria-hidden>⚠</span> {s.reason || t('settings.maintenance.couldNotCheck')}
            </p>
          )}
        </div>
      )}
    </Card>
  )
}

/* Server-log viewer: tail data/app.log (fallback data/server.log) so an error
   can be copy-pasted into a bug report without hunting for files. Fetches on
   open, auto-refreshes every 5 s while open. */
function LogViewer() {
  const { t } = useI18n()
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState(null)
  const [lines, setLines] = useState([])
  const load = async () => {
    try {
      const d = await apiFetch('/api/logs/tail?n=300')
      setFile(d.file); setLines(d.lines || [])
    } catch { /* viewer is best-effort */ }
  }
  useEffect(() => {
    if (!open) return undefined
    load()
    const id = setInterval(load, 5000)
    return () => clearInterval(id)
  }, [open])
  const copy = () => { try { navigator.clipboard.writeText(lines.join('\n')) } catch { /* ignore */ } }
  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open}
        className="flex w-full items-center gap-2 text-left">
        <h2 className="text-base font-semibold text-content">🪵 {t('settings.maintenance.logTitle')}</h2>
        <span className="text-xs text-content-subtle">
          {open
            ? (file
              ? t('settings.maintenance.logStatus', { file: `data/${file}`, count: lines.length })
              : t('settings.maintenance.noLog'))
            : t('settings.maintenance.logClosedHelp')}
        </span>
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          <div className="flex gap-2">
            <button type="button" onClick={load}
              className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs text-content">
              ↻ {t('settings.maintenance.refresh')}
            </button>
            <button type="button" onClick={copy} disabled={!lines.length}
              className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs text-content disabled:opacity-40">
              📋 {t('settings.maintenance.copyAll')}
            </button>
          </div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-app/60 p-2 text-[11px] leading-snug text-content-muted">
            {lines.length ? lines.join('\n') : t('settings.maintenance.logEmpty')}
          </pre>
        </div>
      )}
    </section>
  )
}

/* App-wide trash: everything the app "deletes" (checkpoints, cloud staging,
   deployed LoRAs) is MOVED here — this card is the only place bytes actually
   die. Size fetched once on mount (no poll). */
function TrashCard() {
  const { t } = useI18n()
  const [size, setSize] = useState(null)
  const [busy, setBusy] = useState(false)
  const [opening, setOpening] = useState(false)
  useEffect(() => {
    let alive = true
    apiFetch('/api/trash')
      .then((d) => { if (alive) setSize(d?.size_bytes ?? null) })
      .catch(() => { /* best-effort */ })
    return () => { alive = false }
  }, [])
  const fmt = (b) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB`
    : b >= 1e6 ? `${Math.round(b / 1e6)} MB`
    : b > 0 ? `${Math.max(1, Math.round(b / 1e3))} KB` : t('settings.maintenance.empty'))
  const openFolder = async () => {
    setOpening(true)
    try {
      const d = await postJson('/api/trash/open', {})
      if (!d?.ok) window.alert(d?.error || t('settings.maintenance.openTrashFailed'))
    } catch {
      window.alert(t('settings.maintenance.openTrashFailed'))
    } finally {
      setOpening(false)
    }
  }
  const empty = async () => {
    if (!window.confirm(t('settings.maintenance.emptyConfirm'))) return
    setBusy(true)
    try {
      const d = await postJson('/api/trash/empty', {})
      if (d?.ok) setSize(0)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card title={t('settings.maintenance.trashTitle')} help={t('settings.maintenance.trashHelp')}>
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-content">
          <span aria-hidden>🗑</span> {t('settings.maintenance.trashSize')}{' '}
          <span className="font-semibold tabular-nums">{size == null ? '…' : fmt(size)}</span>
        </span>
        <button type="button" onClick={openFolder} disabled={opening}
          title={t('settings.maintenance.openTrashTitle')}
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm font-medium text-content disabled:opacity-40">
          {opening ? t('settings.maintenance.opening') : t('settings.maintenance.openFolder')}
        </button>
        <button type="button" onClick={empty} disabled={busy || !size}
          className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 disabled:opacity-40">
          {busy ? t('settings.maintenance.emptying') : t('settings.maintenance.emptyTrash')}
        </button>
      </div>
    </Card>
  )
}

export default function MaintenanceSection({ config, setField }) {
  const { t } = useI18n()
  return (
    <div className="space-y-6">
      <UpdatesCard />
      <TrashCard />
      <Card title={t('settings.maintenance.dataTitle')} help={t('settings.maintenance.dataHelp')}>
        <TextField
          id="dataset-images-root"
          label={t('settings.maintenance.imagesRoot')}
          value={config.paths.dataset_images_root}
          onChange={(v) => setField('paths', 'dataset_images_root', v)}
          placeholder={t('settings.maintenance.imagesRootPlaceholder')}
        />
      </Card>
      <DiagnosticReport />
      <LogViewer />
    </div>
  )
}

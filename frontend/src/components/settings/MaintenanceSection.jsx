import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import DiagnosticReport from '../common/DiagnosticReport'
import DockerUpdateInstructions from '../common/DockerUpdateInstructions'
import PinokioUpdateInstructions from '../common/PinokioUpdateInstructions'
import {
  installMode, patreonCheckFailure, publicReleaseNotice, zipUpdateHeadline, progressLabel, progressPercent,
} from './updateStatus'
import PatreonUpdates from '../patreon/PatreonUpdates'
import { Card } from './primitives'
import { versionLabel } from '../../utils/versionLabel'
import { waitForRelaunchedServer } from '../../utils/serverRestart'
import UsageStatisticsCard from './UsageStatisticsCard'

/* In-app updater: "Check for updates" hits the git-aware check (commits-behind for a
   clone, release tag for a packaged build). "Update & restart" pulls (git) or downloads
   and swaps in the latest release ZIP (packaged install), then restarts the server; we
   poll /api/health until the relaunched process answers and hard-reload the SPA so the
   new frontend/dist loads. */
function UpdatesCard() {
  const [status, setStatus] = useState(null)
  const [checking, setChecking] = useState(false)
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'
  const [progress, setProgress] = useState(null)   // ZIP mode: {phase, downloaded, total}
  // Patreon core channel (GET /api/update/patreon, readable without plugin
  // administration). Null only while the first answer is pending.
  const [patreon, setPatreon] = useState(null)
  const [patreonError, setPatreonError] = useState('')
  const [patreonInstalled, setPatreonInstalled] = useState(null)

  // Same cached Git/release check as the nav badge and banner, so a Git clone
  // never advertises a release ZIP instead of its configured branch.
  useEffect(() => {
    let alive = true
    apiFetch('/api/update/check?auto=1', { background: true })
      .then((d) => { if (alive) setStatus((prev) => prev || d) })
      .catch(() => { /* best-effort — the manual button stays available */ })
    apiFetch('/api/update/patreon', { background: true })
      .then((d) => { if (alive) setPatreon((prev) => prev || d) })
      .catch((e) => { if (alive) setPatreon((prev) => prev || patreonCheckFailure(e)) })
    return () => { alive = false }
  }, [])

  const checkPatreon = async () => {
    try {
      setPatreon(await apiFetch('/api/update/patreon?force=1', { background: true }))
      setPatreonError('')
    } catch (e) {
      // Keep the last answer and say this check failed; without a real answer
      // yet, the block itself shows the failure (never twice).
      const failure = patreonCheckFailure(e)
      setPatreon((prev) => prev || failure)
      setPatreonError(patreon && patreon.status !== 'unavailable' ? failure.message : '')
    }
  }

  const check = async () => {
    setChecking(true)
    try {
      const [publicStatus] = await Promise.all([apiFetch('/api/update/check?force=1'), checkPatreon()])
      setStatus(publicStatus)
    } catch (e) {
      setStatus({ ok: false, reason: e.message || 'Check failed' })
    } finally {
      setChecking(false)
    }
  }

  const waitForHealthAndReload = async () => {
    // Waits for the NEW process (the old one answers until it exits), then
    // hard-reloads to pull the new dist. Changed dependencies can take minutes.
    if (await waitForRelaunchedServer()) { window.location.reload(); return }
    setApplying(false); setPhase('')          // gave up after 15 min
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

  const apply = async ({ forcePublic = false } = {}) => {
    // Defense in depth: Docker owns /app as image content. The button is not
    // rendered in this mode, and a stale callback must not POST an apply anyway.
    if (mode === 'docker') return
    // Same guard for Pinokio: the pull would succeed and the restart would
    // detach the server from the launcher that owns it.
    if (mode === 'pinokio') return
    setApplying(true); setPhase('pulling'); setProgress(null)
    try {
      // force_public: a ZIP install on the Patreon channel leaves it on purpose.
      const res = await postJson('/api/update/apply', forcePublic ? { force_public: true } : {})
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
      setStatus({ ok: false, reason: e.message || 'Update failed' })
      setApplying(false); setPhase('')
    }
  }

  // The Patreon channel installs through the same restart / progress flows:
  // git and Pinokio fast-forward a verified bundle, a ZIP install swaps the
  // verified archive. Pinokio keeps its own Stop / Update / Start.
  const applyPatreon = async () => {
    setApplying(true); setPhase('pulling'); setProgress(null); setPatreonError('')
    try {
      const res = await postJson('/api/update/patreon/apply', {})
      if (res.restarting) {
        setPhase('restarting')
        waitForHealthAndReload()
      } else if (res.async) {
        setProgress({ phase: 'downloading', downloaded: 0, total: res.total || 0 })
        pollProgress()
      } else {
        if (res.restart_required) setPatreonInstalled(res)
        else if (!res.changed) await checkPatreon()
        setApplying(false); setPhase('')
      }
    } catch (e) {
      setPatreonError(e.message || 'The Patreon update could not be installed. Your files were not changed.')
      setApplying(false); setPhase('')
    }
  }

  const s = status
  // In-app update is possible for a git clone (pull) or a packaged install whose
  // latest release ships a ZIP asset (download + swap). Otherwise: link out.
  const dockerMode = mode === 'docker'
  const pinokioMode = mode === 'pinokio'
  const canPull = s && s.update_available && (mode === 'git' || mode === 'zip')
  // A Patreon ZIP or Docker install is never offered the public release; this
  // names it and the explicit way back.
  const notice = publicReleaseNotice(s)
  const installPublic = () => {
    if (!window.confirm('Install the public release over this Patreon release? The Patreon changes are '
      + 'removed; your data and settings are kept, and Patreon updates stay available later.')) return
    apply({ forcePublic: true })
  }
  return (
    <Card title="Updates" help={dockerMode
      ? 'Docker installs are updated from the host: update-docker.bat downloads the new version and '
        + 'rebuilds the image (a Git checkout is pulled by hand).'
      : pinokioMode
        ? 'Pinokio starts and stops this app, so it also owns the update: Stop, Update, Start.'
        : 'Pull the latest version from GitHub and restart — without leaving the app.'}>
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" onClick={check} disabled={checking || applying}
          className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-50">
          {checking ? 'Checking…' : 'Check for updates'}
        </button>
        {s?.current && (
          <span className="text-xs text-content-subtle">
            Current build:{' '}
            <span className="font-medium text-content">v{s.current}{s.current_sha ? ` (${s.current_sha})` : ''}</span>
          </span>
        )}
        {s && (
          <span className="text-xs text-content-subtle">
            Latest build:{' '}
            <span className="font-medium text-content">
              {s.remote_sha
                ? `${s.remote_sha}${typeof s.behind === 'number' && s.behind > 0 ? ` (+${s.behind} commit${s.behind === 1 ? '' : 's'})` : ''}`
                : s.latest ? `v${s.latest}`
                : s.update_available ? 'update available'
                : '— press “Check for updates”'}
            </span>
          </span>
        )}
        {/* Read WHAT the update contains before pulling: the compare view lists
            exactly the incoming commits; otherwise the branch history. Only
            present after a git-aware "Check for updates" (force). */}
        {s && (s.compare_url || s.commits_url) && (
          <a href={s.compare_url || s.commits_url} target="_blank" rel="noreferrer"
            className="text-xs font-medium text-sky-300 underline hover:text-sky-200">
            {s.compare_url ? 'See what’s in this update ↗' : 'Browse recent commits ↗'}
          </a>
        )}
      </div>

      {applying && (
        <div className="space-y-1.5" role="status" aria-live="polite">
          <p className="text-sm text-content-muted">
            {phase === 'restarting'
              ? '↻ Updated — the app is restarting. This page reloads automatically when it’s back…'
              : progressLabel(progress) || (mode === 'zip'
                ? '⬇ Downloading and installing the latest release…'
                : '⬇ Pulling the latest version…')}
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
          {dockerMode && s.update_available ? (
            <div className="space-y-2">
              <p className="text-content">
                <span aria-hidden>⬆</span>{' '}
                Update available{s.latest ? ` — v${s.latest}` : ''}. The running container cannot replace its own image.
              </p>
              <DockerUpdateInstructions status={s} />
            </div>
          ) : pinokioMode && s.update_available ? (
            <div className="space-y-2">
              <p className="text-content">
                <span aria-hidden>⬆</span>{' '}
                {typeof s.behind === 'number' && s.behind > 0
                  ? `${s.behind} commit${s.behind === 1 ? '' : 's'} behind${s.current_sha && s.remote_sha ? ` (${s.current_sha} → ${s.remote_sha})` : ''}.`
                  : `Update available${s.latest ? ` — v${s.latest}` : ''}.`}
              </p>
              <PinokioUpdateInstructions />
            </div>
          ) : canPull ? (
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-content">
                <span aria-hidden>⬆</span>{' '}
                {typeof s.behind === 'number'
                  ? `${s.behind} commit${s.behind === 1 ? '' : 's'} behind${s.current_sha && s.remote_sha ? ` (${s.current_sha} → ${s.remote_sha})` : ''}.`
                  : `${zipUpdateHeadline(s)}.`}
              </span>
              <button type="button" onClick={apply}
                className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 transition-transform hover:-translate-y-px">
                Update &amp; restart
              </button>
            </div>
          ) : s.update_available ? (
            <p className="text-content">
              Update available{s.latest ? ` — v${s.latest}` : ''} —{' '}
              <a href={s.url} target="_blank" rel="noreferrer" className="font-semibold text-emerald-300 underline">
                download the latest release
              </a>{' '}and replace the folder.
            </p>
          ) : s.ok ? (
            <div className="space-y-2">
              <p className="text-emerald-400">
                {/* A members-only release exists: "up to date" would read as "nothing newer". */}
                <span aria-hidden>✓</span> {s.channel === 'patreon' ? 'This installation follows the Patreon channel.'
                  : patreon?.update_available ? 'The free version is up to date.' : 'You’re up to date.'}{' '}
                {/* Name the COMMIT on a git checkout: the release number alone would
                    claim the last release while the tree may be well past it. */}
                <span className="text-content-subtle">{versionLabel(s)}</span>
              </p>
              {notice && (
                <div className="space-y-1 text-content-muted">
                  <p>{notice.text}</p>
                  {notice.canForce && (
                    <button type="button" onClick={installPublic}
                      className="rounded-md border border-border-strong px-3 py-1.5 text-sm font-medium text-content hover:bg-surface-raised">
                      Install the public release instead
                    </button>
                  )}
                  {notice.command && (
                    <p>To return to public releases, run on the host: <code className="text-content">{notice.command}</code></p>
                  )}
                </div>
              )}
            </div>
          ) : (
            <p className="text-content-muted"><span aria-hidden>⚠</span> {s.reason || 'Could not check for updates.'}</p>
          )}
        </div>
      )}

      <PatreonUpdates patreon={patreon} installed={patreonInstalled} error={patreonError}
        applying={applying} onApply={applyPatreon} onChanged={checkPatreon} />
    </Card>
  )
}

/* Server-log viewer: tail data/app.log (fallback data/server.log) so an error
   can be copy-pasted into a bug report without hunting for files. Fetches on
   open, auto-refreshes every 5 s while open. */
function LogViewer() {
  const [open, setOpen] = useState(false)
  const [file, setFile] = useState(null)
  const [lines, setLines] = useState([])
  const load = async () => {
    try {
      const d = await apiFetch('/api/logs/tail?n=300', { background: true })
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
        <h2 className="text-base font-semibold text-content">🪵 Server log</h2>
        <span className="text-xs text-content-subtle">
          {open ? (file ? `data/${file} — last ${lines.length} lines, refreshes every 5 s` : 'no log file yet')
            : 'something failed? open this and copy the log into your bug report'}
        </span>
        <span aria-hidden className="ml-auto text-content-subtle">{open ? '▾' : '▸'}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-2">
          <div className="flex gap-2">
            <button type="button" onClick={load}
              className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs text-content">
              ↻ Refresh
            </button>
            <button type="button" onClick={copy} disabled={!lines.length}
              className="rounded-md border border-border bg-surface-raised px-2.5 py-1 text-xs text-content disabled:opacity-40">
              📋 Copy all
            </button>
          </div>
          <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-md border border-border bg-app/60 p-2 text-[11px] leading-snug text-content-muted">
            {lines.length ? lines.join('\n') : 'Log is empty.'}
          </pre>
        </div>
      )}
    </section>
  )
}

/* Maintenance keeps what is about the APP itself — updating it, reporting a bug,
   reading its log. Everything about the disk (trash, archive, dataset root, where
   the big folders live) moved to Settings › Storage, where those questions are
   answered together instead of one card per screen. */
export default function MaintenanceSection() {
  return (
    <div className="space-y-6">
      <UsageStatisticsCard />
      <UpdatesCard />
      <DiagnosticReport />
      <LogViewer />
    </div>
  )
}

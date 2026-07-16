import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import DiagnosticReport from '../common/DiagnosticReport'
import { Card, TextField } from './primitives'

/* In-app updater: "Check for updates" hits the git-aware check (commits-behind for a
   clone, release tag for a packaged build). "Update & restart" pulls + restarts the
   server; we then poll /api/health until the relaunched process answers and hard-reload
   the SPA so the new frontend/dist loads. */
function UpdatesCard() {
  const [status, setStatus] = useState(null)
  const [checking, setChecking] = useState(false)
  const [applying, setApplying] = useState(false)
  const [phase, setPhase] = useState('')     // '' | 'pulling' | 'restarting'

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
      setStatus({ ok: false, reason: e.message || 'Check failed' })
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

  const apply = async () => {
    setApplying(true); setPhase('pulling')
    try {
      const res = await postJson('/api/update/apply', {})
      if (res.restarting) {
        setPhase('restarting')
        waitForHealthAndReload()              // not awaited: UI shows "restarting…"
      } else {
        setStatus(res.ok ? { ...res, up_to_date: true } : res)
        setApplying(false); setPhase('')
      }
    } catch (e) {
      setStatus({ ok: false, reason: e.message || 'Update failed' })
      setApplying(false); setPhase('')
    }
  }

  const s = status
  const canPull = s && s.update_available && s.is_git
  return (
    <Card title="Updates" help="Pull the latest version from GitHub and restart — without leaving the app.">
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
        <p className="text-sm text-content-muted" role="status">
          {phase === 'restarting'
            ? '↻ Updated — the app is restarting. This page reloads automatically when it’s back…'
            : '⬇ Pulling the latest version…'}
        </p>
      )}

      {!applying && s && (
        <div className="text-sm">
          {canPull ? (
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-content">
                <span aria-hidden>⬆</span>{' '}
                {typeof s.behind === 'number'
                  ? `${s.behind} commit${s.behind === 1 ? '' : 's'} behind${s.current_sha && s.remote_sha ? ` (${s.current_sha} → ${s.remote_sha})` : ''}.`
                  : `Update available${s.latest ? ` — v${s.latest}` : ''}.`}
              </span>
              <button type="button" onClick={apply}
                className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-white transition-transform hover:-translate-y-px">
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
            <p className="text-emerald-400"><span aria-hidden>✓</span> You’re up to date.</p>
          ) : (
            <p className="text-content-muted"><span aria-hidden>⚠</span> {s.reason || 'Could not check for updates.'}</p>
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

/* App-wide trash: everything the app "deletes" (checkpoints, cloud staging,
   deployed LoRAs) is MOVED here — this card is the only place bytes actually
   die. Size fetched once on mount (no poll). */
function TrashCard() {
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
    : b > 0 ? `${Math.max(1, Math.round(b / 1e3))} KB` : 'empty')
  const openFolder = async () => {
    setOpening(true)
    try {
      const d = await postJson('/api/trash/open', {})
      if (!d?.ok) window.alert(d?.error || 'Could not open the trash folder.')
    } catch {
      window.alert('Could not open the trash folder.')
    } finally {
      setOpening(false)
    }
  }
  const empty = async () => {
    if (!window.confirm('Permanently delete everything in the trash?\n\nThis is the ONLY destructive action — deleted checkpoints cannot be recovered afterwards.')) return
    setBusy(true)
    try {
      const d = await postJson('/api/trash/empty', {})
      if (d?.ok) setSize(0)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Card title="Trash" help="Everything the app deletes (checkpoints, cloud staging, deployed LoRAs) is moved here first — emptying it is the only action that actually destroys files.">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-sm text-content">
          <span aria-hidden>🗑</span> Trash size:{' '}
          <span className="font-semibold tabular-nums">{size == null ? '…' : fmt(size)}</span>
        </span>
        <button type="button" onClick={openFolder} disabled={opening}
          title="Open the trash folder in the file explorer"
          className="rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm font-medium text-content disabled:opacity-40">
          {opening ? 'Opening…' : '📂 Open folder'}
        </button>
        <button type="button" onClick={empty} disabled={busy || !size}
          className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-1.5 text-sm font-medium text-red-300 disabled:opacity-40">
          {busy ? 'Emptying…' : 'Empty trash'}
        </button>
      </div>
    </Card>
  )
}

export default function MaintenanceSection({ config, setField }) {
  return (
    <div className="space-y-6">
      <UpdatesCard />
      <TrashCard />
      <Card title="Data" help="Where dataset images live on disk.">
        <TextField
          id="dataset-images-root"
          label="Dataset images root"
          value={config.paths.dataset_images_root}
          onChange={(v) => setField('paths', 'dataset_images_root', v)}
          placeholder="Defaults to data/datasets"
        />
      </Card>
      <DiagnosticReport />
      <LogViewer />
    </div>
  )
}

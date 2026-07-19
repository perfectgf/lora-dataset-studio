import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INSTALL_ALL_ACTION_LABELS, installCatalog } from '../../hooks/useSetupSteps'
import InstallRunner from './InstallRunner'
import { HelpBadge } from '../../help/HelpMode'

const POLL_MS = 1200

function fmtSize(b) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`
  return `${Math.max(0, Math.round(b / 1e3))} KB`
}

const label = (action) => INSTALL_ALL_ACTION_LABELS[action] || action

// Per-action status glyph/colour for the shortcut's progress list. `queued`/`running` are
// live; `success`/`error` are terminal; `idle` shows before its turn (a queued pip install).
const ROW_META = {
  idle: { glyph: '○', cls: 'text-content-subtle', word: 'waiting' },
  queued: { glyph: '○', cls: 'text-content-subtle', word: 'queued' },
  running: { glyph: '⟳', cls: 'text-primary', word: 'installing…' },
  success: { glyph: '✓', cls: 'text-emerald-400', word: 'done' },
  error: { glyph: '✗', cls: 'text-rose-400', word: 'needs attention' },
}

// One tile in the one-by-one menu: a component the app can install itself, its live state,
// and an Install / ↻ Reinstall button (reusing the Setup InstallRunner verbatim — polling,
// live pip log/download %, and the repair-in-place error path all come from it). Items whose
// precondition isn't met yet render their hint (a pointer back to the config step) instead of
// a button. The tile stays even once installed, so a broken venv can be rebuilt at any time.
function InstallItem({ item, onDone }) {
  const { action, label: lbl, present, available, hint } = item
  return (
    <div className="rounded-md border border-border bg-surface-raised p-3 space-y-2">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold text-content">{lbl}</span>
        <span className={`shrink-0 text-xs font-medium ${present ? 'text-emerald-400' : 'text-content-subtle'}`}>
          {present ? '✓ Installed' : '✗ Not installed'}
        </span>
      </div>
      {available ? (
        <InstallRunner action={action}
          buttonLabel={present ? '↻ Reinstall' : 'Install'} onDone={onDone} />
      ) : (
        <p className="text-xs text-content-subtle">{hint}</p>
      )}
    </div>
  )
}

// The Setup "install" step (reached AFTER the API/service config, never on the welcome
// screen — several installs depend on a configured ComfyUI/Ollama). Two paths, one screen:
//
//   1. "Install everything" — one click queues every install the app can run ITSELF right
//      now (the `plan`, derived from live capabilities): the missing ML extras, the Ollama
//      vision model when Ollama is up, the Klein weights when a valid ComfyUI is set. The
//      backend serializes the pip installs (two never race one venv) and downloads models in
//      parallel; this fans out and polls ONE batched status endpoint for a global "X / N" bar.
//
//   2. The one-by-one menu below — ALWAYS shown, listing every app-installable component with
//      an Install / ↻ Reinstall button, so a user can pick and choose or repair a single
//      broken install (a corrupted venv) without redoing everything, even once all is green.
//
// Neither installs ComfyUI/Ollama themselves nor pastes API keys — those stay on the
// step-by-step path (external tools / credentials).
export default function InstallEverything({ plan, caps, onDone }) {
  const toast = useToast()
  const [phase, setPhase] = useState('idle')     // idle | running | done
  const [tracked, setTracked] = useState([])     // the plan captured when the run started
  const [statuses, setStatuses] = useState({})   // action -> live status
  const timer = useRef(null)
  const mounted = useRef(true)

  const isTerminal = (s) => s && (s.state === 'success' || s.state === 'error')
  const allTerminal = (st, actions) => actions.length > 0 && actions.every((a) => isTerminal(st[a]))

  const finish = (st, actions) => {
    setPhase('done')
    onDone?.()
    const failed = actions.filter((a) => (st[a] || {}).state === 'error')
    if (failed.length) {
      toast.warning(`${actions.length - failed.length} of ${actions.length} installed — `
        + `${failed.length} need${failed.length === 1 ? 's' : ''} attention below.`)
    } else {
      toast.success('Everything installed.')
    }
  }

  const poll = (actions) => {
    apiFetch(`/api/setup/install-all/status?actions=${actions.join(',')}`).then((r) => {
      if (!mounted.current) return
      const st = r.statuses || {}
      setStatuses(st)
      if (allTerminal(st, actions)) finish(st, actions)
      else timer.current = setTimeout(() => poll(actions), POLL_MS)
    }).catch(() => {
      if (mounted.current) timer.current = setTimeout(() => poll(actions), POLL_MS)
    })
  }

  // Re-attach on mount to a batch that may already be running (the user left this screen
  // mid-install and came back — the backend kept going). Only resumes when something is
  // actually in flight; otherwise the button stays ready.
  useEffect(() => {
    mounted.current = true
    if (plan && plan.length) {
      apiFetch(`/api/setup/install-all/status?actions=${plan.join(',')}`).then((r) => {
        if (!mounted.current) return
        const st = r.statuses || {}
        const active = plan.some((a) => ['running', 'queued'].includes((st[a] || {}).state))
        if (active) { setTracked(plan); setStatuses(st); setPhase('running'); poll(plan) }
      }).catch(() => { /* not attached — leave idle */ })
    }
    return () => { mounted.current = false; clearTimeout(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = async () => {
    setPhase('running'); setStatuses({})
    try {
      const r = await postJson('/api/setup/install-all', {})
      const actions = r.plan || []
      setTracked(actions); setStatuses(r.statuses || {})
      if (actions.length === 0) { setPhase('done'); onDone?.(); return }
      poll(actions)
    } catch (e) {
      setPhase('idle')
      toast.error(e.message || 'Could not start the install.')
    }
  }

  const catalog = installCatalog(caps)
  const rows = (phase === 'idle' ? plan : tracked) || []
  const doneCount = rows.filter((a) => (statuses[a] || {}).state === 'success').length
  // Nothing left for the one-click shortcut to queue (everything installable is already in).
  // The batch card collapses to a satisfying note; the one-by-one menu below stays for repairs.
  const nothingToInstall = (!plan || plan.length === 0) && phase !== 'running'

  return (
    <div className="space-y-4">
      {/* Path 1 — the one-click shortcut. */}
      <section className="rounded-xl border border-primary/40 bg-primary/5 p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-content">
              ⬇ Install everything
              <HelpBadge topic="page-setup" className="ml-2" />
            </h2>
            <p className="mt-1 text-sm text-content-muted">
              One click sets up every component the app can install for you. Heavy installs run
              one at a time so they never clash; the big model downloads run in parallel.
            </p>
          </div>
          {phase === 'running' && (
            <span className="shrink-0 text-xs font-medium tabular-nums text-content-muted">
              {doneCount} / {rows.length}
            </span>
          )}
        </div>

        {nothingToInstall ? (
          <p className="mt-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
            ✓ Everything the app can install itself is already in place. Use the list below to
            reinstall or repair any component.
          </p>
        ) : (
          <>
            {/* Global progress bar during a run. */}
            {phase === 'running' && rows.length > 0 && (
              <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
                <div className="h-full rounded-full bg-gradient-primary transition-[width] duration-300"
                  style={{ width: `${Math.round((doneCount / rows.length) * 100)}%` }} />
              </div>
            )}

            {/* What will be / is being installed. */}
            <ul className="mt-3 space-y-1.5">
              {rows.map((a) => {
                const s = statuses[a] || {}
                const state = phase === 'idle' ? 'idle' : (s.state || 'idle')
                const m = ROW_META[state] || ROW_META.idle
                const pr = s.progress
                return (
                  <li key={a} className="flex items-center justify-between gap-3 text-sm">
                    <span className="flex items-center gap-2">
                      <span aria-hidden="true" className={m.cls}>{m.glyph}</span>
                      <span className="text-content-muted">{label(a)}</span>
                    </span>
                    <span className="text-xs tabular-nums text-content-subtle">
                      {state === 'running' && pr && pr.total
                        ? `${pr.pct != null ? `${pr.pct}% · ` : ''}${fmtSize(pr.done)} / ${fmtSize(pr.total)}`
                        : (phase === 'idle' ? '' : m.word)}
                    </span>
                  </li>
                )
              })}
            </ul>

            {phase === 'done' ? (
              <p className="mt-4 text-sm font-medium text-content">
                {rows.every((a) => (statuses[a] || {}).state === 'success')
                  ? '✓ All done. You can start building datasets.'
                  : 'Some installs need another look — click Install everything again to retry them, or use the list below.'}
              </p>
            ) : (
              <button type="button" onClick={start} disabled={phase === 'running'}
                className="mt-4 rounded-lg bg-gradient-primary px-5 py-2 text-sm font-semibold text-white disabled:opacity-50">
                {phase === 'running' ? 'Installing…' : `Install everything (${(plan || []).length})`}
              </button>
            )}
            {(phase === 'done' && !rows.every((a) => (statuses[a] || {}).state === 'success')) && (
              <button type="button" onClick={start}
                className="ml-3 mt-4 rounded-lg border border-border-strong px-4 py-2 text-sm font-medium text-content hover:bg-surface-raised">
                Retry
              </button>
            )}
          </>
        )}
      </section>

      {/* Path 2 — the one-by-one menu, always visible (install/repair a single component). */}
      <section className="rounded-xl border border-border bg-surface p-5">
        <h3 className="text-base font-semibold text-content">Install or repair individually</h3>
        <p className="mt-1 text-sm text-content-muted">
          Prefer to pick and choose? Install any component on its own here. Already installed?
          Use <span className="font-medium text-content">↻ Reinstall</span> to repair or update it —
          handy when a dedicated environment breaks.
        </p>
        <div className="mt-3 space-y-3">
          {catalog.map((c) => <InstallItem key={c.action} item={c} onDone={onDone} />)}
        </div>
      </section>
    </div>
  )
}

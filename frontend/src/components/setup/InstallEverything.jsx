import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INSTALL_ALL_ACTION_LABELS } from '../../hooks/useSetupSteps'
import { HelpBadge } from '../../help/HelpMode'

const POLL_MS = 1200

function fmtSize(b) {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`
  if (b >= 1e6) return `${(b / 1e6).toFixed(0)} MB`
  return `${Math.max(0, Math.round(b / 1e3))} KB`
}

const label = (action) => INSTALL_ALL_ACTION_LABELS[action] || action

// Per-action status glyph/colour for the progress list. `queued`/`running` are live;
// `success`/`error` are terminal; `idle` shows before its turn (a queued pip install).
const ROW_META = {
  idle: { glyph: '○', cls: 'text-content-subtle', word: 'waiting' },
  queued: { glyph: '○', cls: 'text-content-subtle', word: 'queued' },
  running: { glyph: '⟳', cls: 'text-primary', word: 'installing…' },
  success: { glyph: '✓', cls: 'text-emerald-400', word: 'done' },
  error: { glyph: '✗', cls: 'text-rose-400', word: 'needs attention' },
}

// The prominent "Install everything" block that leads the Setup welcome screen. One
// click queues every install the app can run itself (the `plan`, derived from live
// capabilities): the missing ML extras, the Ollama vision model when Ollama is up, and
// the Klein weights when a valid ComfyUI is set. The backend serializes the pip installs
// (two never race one venv) and downloads the models in parallel; this component just
// fans out and polls ONE batched status endpoint for a global "X / N" bar. Everything
// else — API keys, installing ComfyUI/Ollama themselves — stays on the step-by-step path.
export default function InstallEverything({ plan, onDone }) {
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

  // Empty plan = nothing the app can install itself is missing. A satisfying, explicit
  // "all set" state (the manual tools it can't install live in the scan above).
  if (phase !== 'running' && (!plan || plan.length === 0)) {
    return (
      <section className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-5 text-center">
        <div className="text-2xl" aria-hidden="true">✓</div>
        <h2 className="mt-1 text-base font-semibold text-content">Everything installable is ready</h2>
        <p className="mt-1 text-sm text-content-muted">
          All the components the app can install itself are in place. Anything still marked
          above (API keys, ComfyUI or Ollama) is set up on its own — use the step-by-step path.
        </p>
      </section>
    )
  }

  const rows = (phase === 'idle' ? plan : tracked) || []
  const doneCount = rows.filter((a) => (statuses[a] || {}).state === 'success').length

  return (
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
            : 'Some installs need another look — click Install everything again to retry them, or open them individually.'}
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
    </section>
  )
}

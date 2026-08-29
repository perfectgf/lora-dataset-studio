import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  INSTALL_ALL_ACTION_LABELS, kreaInstallPlan, kreaNeedsComfyuiRestart,
} from '../../hooks/useSetupSteps'
import { localEngineReadiness } from '../../utils/localEngineReason'
import { HelpBadge } from '../../help/HelpMode'
import { fmtSize } from './fmtSize'
import KreaSamplerInstallRow from './KreaSamplerInstallRow'

const POLL_MS = 1200

const ROW_META = {
  idle: { glyph: '○', cls: 'text-content-subtle', word: 'waiting' },
  queued: { glyph: '○', cls: 'text-content-subtle', word: 'queued' },
  running: { glyph: '⟳', cls: 'text-primary', word: 'installing…' },
  success: { glyph: '✓', cls: 'text-emerald-400', word: 'done' },
  error: { glyph: '✗', cls: 'text-rose-400', word: 'needs attention' },
}


// One click that deploys the SECOND local engine: the comfyui-krea2edit custom-node
// pack (a small clone into this install's custom_nodes) plus its four weights.
//
// WHY IT IS ITS OWN CARD, not part of "Install everything"
// -------------------------------------------------------
// Krea is ~20 GB on top of Klein's ~20. "Install everything" runs unattended, so
// pulling a second engine nobody selected would be hostile on a metered link or a
// small disk. Here the click IS the request. (The other one-click path is picking
// Krea in the workspace and pressing Generate — the 409 fires the same installs.)
//
// The card never claims more than it did: a node pack only registers when ComfyUI
// RESTARTS, so that instruction is shown as its own state instead of being buried
// in a success message.
export default function KreaInstallCard({ caps, onDone }) {
  const toast = useToast()
  const [phase, setPhase] = useState('idle')
  const [tracked, setTracked] = useState([])
  const [statuses, setStatuses] = useState({})
  const timer = useRef(null)
  const mounted = useRef(true)

  const plan = kreaInstallPlan(caps)
  const cu = caps?.comfyui || {}
  const dirValid = !!cu.dir_valid
  const needsRestart = kreaNeedsComfyuiRestart(caps)
  // The engine's REAL verdict, from the backend, identical to the one the
  // generation panel gates its Krea button on.
  const readiness = localEngineReadiness('krea', caps)
  const rows = (phase === 'idle' ? plan : tracked) || []
  const doneCount = rows.filter((a) => (statuses[a] || {}).state === 'success').length
  const isTerminal = (s) => s && (s.state === 'success' || s.state === 'error')

  const poll = (actions) => {
    apiFetch(`/api/setup/install-all/status?actions=${actions.join(',')}`).then((r) => {
      if (!mounted.current) return
      const st = r.statuses || {}
      setStatuses(st)
      if (actions.length && actions.every((a) => isTerminal(st[a]))) {
        setPhase('done')
        onDone?.()
        const failed = actions.filter((a) => (st[a] || {}).state === 'error')
        if (failed.length) {
          toast.warning(`${actions.length - failed.length} of ${actions.length} Krea pieces `
            + 'installed — the rest need a look below.')
        } else {
          toast.success('Krea 2 Edit installed. Restart ComfyUI to load its nodes.')
        }
      } else {
        timer.current = setTimeout(() => poll(actions), POLL_MS)
      }
    }).catch(() => {
      if (mounted.current) timer.current = setTimeout(() => poll(actions), POLL_MS)
    })
  }

  // Re-attach to a run still in flight (the user left this screen and came back —
  // the backend kept downloading).
  useEffect(() => {
    mounted.current = true
    if (plan.length) {
      apiFetch(`/api/setup/install-all/status?actions=${plan.join(',')}`).then((r) => {
        if (!mounted.current) return
        const st = r.statuses || {}
        if (plan.some((a) => ['running', 'queued'].includes((st[a] || {}).state))) {
          setTracked(plan); setStatuses(st); setPhase('running'); poll(plan)
        }
      }).catch(() => { /* not attached — stay idle */ })
    }
    return () => { mounted.current = false; clearTimeout(timer.current) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const start = async () => {
    setPhase('running'); setStatuses({})
    try {
      const r = await postJson('/api/setup/install-group/krea', {})
      const actions = r.plan || []
      setTracked(actions); setStatuses(r.statuses || {})
      if (!actions.length) { setPhase('done'); onDone?.(); return }
      poll(actions)
    } catch (e) {
      setPhase('idle')
      toast.error(e.message || 'Could not start the Krea install.')
    }
  }

  const nothingToInstall = plan.length === 0 && phase !== 'running'

  return (
    <section className="rounded-xl border border-border bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <h3 className="text-base font-semibold text-content">
          Krea 2 Edit — optional second local engine
          <HelpBadge topic="setup-krea-install" className="ml-2" />
        </h3>
        {phase === 'running' && (
          <span className="shrink-0 text-xs font-medium tabular-nums text-content-muted">
            {doneCount} / {rows.length}
          </span>
        )}
      </div>
      <p className="mt-1 text-sm text-content-muted">
        Re-stages one reference photo — new angle, framing, light, background — while keeping
        the face, with no character LoRA. It needs a community node pack and four model files
        (<span className="whitespace-nowrap">~20 GB</span>), so it is installed on request, not
        by &ldquo;Install everything&rdquo;. Klein alone is enough to build datasets.
      </p>

      {!dirValid ? (
        <p className="mt-3 text-xs text-content-subtle">
          Point the app at a valid ComfyUI folder first (the ComfyUI step) — the node pack and
          the weights go inside it.
        </p>
      ) : nothingToInstall ? (
        // "Nothing left to download" is NOT "the engine works". This banner used to
        // say the second while only knowing the first, so a stopped ComfyUI, an
        // unloaded node pack or a corrupted weight all got a green ✓ here while the
        // Generate page refused Krea. It now shows the engine's real verdict — the
        // same one that page reads — and, when it is not ready, the same sentence.
        readiness.ready ? (
          <p className="mt-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
            ✓ Krea 2 Edit is ready — nothing left to install.
          </p>
        ) : (
          <p className="mt-3 break-words rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-content">
            {readiness.reason || '⚠ Krea 2 Edit is not ready yet.'}
            {!readiness.verified && (
              <span className="mt-1 block text-content-muted">
                Every file is downloaded. ComfyUI is not answering, so the app could not
                check that Krea can actually run — start ComfyUI and re-check.
              </span>
            )}
          </p>
        )
      ) : (
        <>
          {phase === 'running' && rows.length > 0 && (
            <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-surface-raised">
              <div className="h-full rounded-full bg-gradient-primary transition-[width] duration-300"
                style={{ width: `${Math.round((doneCount / rows.length) * 100)}%` }} />
            </div>
          )}
          <ul className="mt-3 space-y-1.5">
            {rows.map((a) => {
              const s = statuses[a] || {}
              const state = phase === 'idle' ? 'idle' : (s.state || 'idle')
              const m = ROW_META[state] || ROW_META.idle
              const pr = s.progress
              return (
                <li key={a} className="flex items-center justify-between gap-2 text-sm">
                  <span className="flex min-w-0 items-center gap-2">
                    <span aria-hidden="true" className={m.cls}>{m.glyph}</span>
                    <span className="truncate text-content-muted">
                      {INSTALL_ALL_ACTION_LABELS[a] || a}
                    </span>
                  </span>
                  <span className="shrink-0 text-xs tabular-nums text-content-subtle">
                    {state === 'running' && pr && pr.total
                      ? `${pr.pct != null ? `${pr.pct}% · ` : ''}${fmtSize(pr.done)} / ${fmtSize(pr.total)}`
                      : (phase === 'idle' ? '' : m.word)}
                  </span>
                </li>
              )
            })}
          </ul>
          <button type="button" onClick={start} disabled={phase === 'running'}
            className="mt-4 w-full rounded-lg border border-primary/50 bg-primary/10 px-4 py-2 text-sm font-semibold text-primary disabled:opacity-50 sm:w-auto">
            {phase === 'running' ? 'Installing…' : `Install Krea 2 Edit (${plan.length})`}
          </button>
        </>
      )}

      {/* The one thing no installer can do for the user. Shown whenever the pack is
          on disk but ComfyUI has not loaded it — including long after the install,
          because until the restart the engine card stays red. */}
      {needsRestart && (
        <p className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-content">
          ⚠ The node pack is installed but ComfyUI has not loaded it yet.
          <span className="text-content-muted"> ComfyUI only registers custom nodes at startup —
            restart it, then this page turns green on its own.</span>
        </p>
      )}

      {/* The optional sampler, offered only once there is a ComfyUI to copy it
          into. Below the engine's own state on purpose: it is an addition to a
          working Krea, never a piece of one. */}
      {dirValid && <KreaSamplerInstallRow caps={caps} onDone={onDone} />}
    </section>
  )
}

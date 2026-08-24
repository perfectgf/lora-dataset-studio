import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  INSTALL_ALL_ACTION_LABELS, seedvr2InstallPlan, seedvr2NeedsComfyuiRestart,
} from '../../hooks/useSetupSteps'
import { HelpBadge } from '../../help/HelpMode'
import { ceilingLine, tilingStatus, TTP_PACK, TTP_URL } from './seedvr2Tiling.js'
import { fmtSize } from './fmtSize'

const POLL_MS = 1200

const ROW_META = {
  idle: { glyph: '○', cls: 'text-content-subtle', word: 'waiting' },
  queued: { glyph: '○', cls: 'text-content-subtle', word: 'queued' },
  running: { glyph: '⟳', cls: 'text-primary', word: 'downloading…' },
  success: { glyph: '✓', cls: 'text-emerald-400', word: 'done' },
  error: { glyph: '✗', cls: 'text-rose-400', word: 'needs attention' },
}


/* The three places this feature actually comes from. Verified 2026-08-02 (HTTP
   200 on each), and worth linking rather than naming: the pack URL is what
   someone who prefers cloning by hand needs, the weights repo is what the
   button below downloads (3.9 GB — say where from), and the original project is
   the credit. Apache-2.0 throughout. */
const TILING_TONE = {
  ready: 'border-emerald-500/30 bg-emerald-500/10',
  restart: 'border-amber-500/40 bg-amber-500/10',
  absent: 'border-border bg-surface-raised',
  unknown: 'border-border bg-surface-raised',
}

const PACK_URL = 'https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler'
const WEIGHTS_URL = 'https://huggingface.co/numz/SeedVR2_comfyUI'
const PROJECT_URL = 'https://github.com/ByteDance-Seed/SeedVR'

// SeedVR2 — the FIDELITY upscaler (issue #32, requested by SurpassHR).
//
// TWO HALVES, INSTALLED DIFFERENTLY, AND THE CARD SAYS SO
// -------------------------------------------------------
// The WEIGHTS are two files in a folder: the app downloads them here, on a click,
// like every other model. The NODE PACK is code with thirteen Python
// dependencies that belong in ComfyUI's own interpreter — which this app does not
// own and must never pip into. Cloning it would land a pack that fails to import
// and leave the user reading "install the pack" about a pack that is right there.
// So the pack is a clear instruction (ComfyUI-Manager, which installs the
// requirements properly) and never a fake button.
//
// Like the Krea card, this is deliberately NOT part of "Install everything":
// 3.9 GB for a capability nobody asked for is hostile on a metered link.
export default function SeedVr2InstallCard({ caps, onDone }) {
  const toast = useToast()
  const [phase, setPhase] = useState('idle')
  const [tracked, setTracked] = useState([])
  const [statuses, setStatuses] = useState({})
  const timer = useRef(null)
  const mounted = useRef(true)

  const plan = seedvr2InstallPlan(caps)
  const cu = caps?.comfyui || {}
  const dirValid = !!cu.dir_valid
  const ready = cu.seedvr2_ready === true
  const needsRestart = seedvr2NeedsComfyuiRestart(caps)
  // The optional high-resolution lane. Its own detection, its own wording:
  // it is a SECOND pack, from a different author, and it is not required.
  const tiling = tilingStatus(caps)
  const ceiling = ceilingLine(caps)
  // On disk but not loaded is a RESTART; absent from disk is an install the user
  // has to run in ComfyUI. Two different sentences, so they are two states.
  const packMissing = !cu.seedvr2_nodes_installed
    && Array.isArray(cu.seedvr2_nodes_missing) && cu.seedvr2_nodes_missing.length > 0
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
          toast.warning(`${actions.length - failed.length} of ${actions.length} SeedVR2 files `
            + 'downloaded — the rest need a look below.')
        } else {
          toast.success('SeedVR2 weights downloaded.')
        }
      } else {
        timer.current = setTimeout(() => poll(actions), POLL_MS)
      }
    }).catch(() => {
      if (mounted.current) timer.current = setTimeout(() => poll(actions), POLL_MS)
    })
  }

  // Re-attach to a download still in flight (the user left this screen and came
  // back — the backend kept going).
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
      const r = await postJson('/api/setup/install-group/seedvr2', {})
      const actions = r.plan || []
      setTracked(actions); setStatuses(r.statuses || {})
      if (!actions.length) { setPhase('done'); onDone?.(); return }
      poll(actions)
    } catch (e) {
      setPhase('idle')
      toast.error(e.message || 'Could not start the SeedVR2 download.')
    }
  }

  const nothingToDownload = plan.length === 0 && phase !== 'running'

  return (
    <section className="rounded-xl border border-border bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <h3 className="text-base font-semibold text-content">
          SeedVR2 — optional fidelity upscaler
          <HelpBadge topic="setup-seedvr2-install" className="ml-2" />
        </h3>
        {phase === 'running' && (
          <span className="shrink-0 text-xs font-medium tabular-nums text-content-muted">
            {doneCount} / {rows.length}
          </span>
        )}
      </div>
      <p className="mt-1 text-sm text-content-muted">
        The second way to run ✨ Upscale &amp; improve. Klein re-renders detail from a prompt —
        sharper, but skin and colour can shift; SeedVR2 resolves detail at a higher resolution
        and leaves the original look alone. It needs a community node pack plus two model files
        (<span className="whitespace-nowrap">~3.9 GB</span>), so it is installed on request,
        not by &ldquo;Install everything&rdquo;.
      </p>
      <p className="mt-1 text-xs text-content-subtle">
        <a href={PACK_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Node pack →</a>
        {' · '}
        <a href={WEIGHTS_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Model weights →</a>
        {' · '}
        <a href={PROJECT_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">SeedVR2 by ByteDance-Seed →</a>
        {' — all Apache-2.0.'}
      </p>

      {!dirValid ? (
        <p className="mt-3 text-xs text-content-subtle">
          Point the app at a valid ComfyUI folder first (the ComfyUI step) — the weights go
          inside it, under <code>models/SEEDVR2</code>.
        </p>
      ) : nothingToDownload ? (
        // "Nothing left to download" is NOT "it works" — the node pack is
        // installed outside this app, so the weights can all be there while the
        // capability stays dark. The real verdict comes from the backend.
        ready ? (
          <p className="mt-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
            ✓ SeedVR2 is ready — it appears in the workspace bulk actions.
          </p>
        ) : (
          <p className="mt-3 break-words rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-content">
            ⚠ The weights are in place, but SeedVR2 cannot run yet — see the node pack note below.
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
            {phase === 'running' ? 'Downloading…' : `Download SeedVR2 models (${plan.length})`}
          </button>
        </>
      )}

      {/* THE OPTIONAL HIGH-RESOLUTION LANE — a different pack, a different
          author, and genuinely optional: without it the upscaler still works,
          it is only limited to what the card holds in one pass. Contributed by
          SurpassHR (GitHub #32), who hit that limit as a CUDA out-of-memory and
          shipped the tiled workflow this lane is ported from. */}
      <div className={`mt-3 rounded-md border px-3 py-2 text-sm text-content ${
        TILING_TONE[tiling.state] || TILING_TONE.absent}`}>
        <div className="font-semibold">
          {tiling.state === 'ready' ? '✓' : '○'} High-resolution upscales (tiling)
        </div>
        {ceiling && <p className="mt-1 text-content-muted">{ceiling}</p>}
        <p className="mt-1 text-content-muted">{tiling.text}</p>
        <p className="mt-1 text-xs text-content-subtle">
          <a href={TTP_URL} target="_blank" rel="noreferrer"
            className="text-sky-300 underline hover:text-sky-200">{TTP_PACK} →</a>
          {' — MIT. Tiling workflow contributed by SurpassHR (GitHub #32).'}
        </p>
      </div>

      {/* The half this app deliberately does NOT install. Spelled out rather than
          hidden behind a button that could not work: the pack's Python
          dependencies have to land in ComfyUI's interpreter, which is exactly
          what ComfyUI-Manager does and what a bare clone does not. */}
      {needsRestart ? (
        <p className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-content">
          ⚠ The SeedVR2 node pack is installed but ComfyUI has not loaded it yet.
          <span className="text-content-muted"> ComfyUI only registers custom nodes at
            startup — restart it, and this page turns green on its own. If it still does not,
            the pack&rsquo;s Python dependencies failed to install: ComfyUI&rsquo;s console says
            which one.</span>
        </p>
      ) : packMissing && (
        <p className="mt-3 break-words rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-content">
          ⚠ The SeedVR2 node pack is not installed in ComfyUI.
          <span className="text-content-muted"> Install it from ComfyUI itself — search
            &ldquo;SeedVR2&rdquo; in ComfyUI-Manager — then restart ComfyUI. The app does not
            install this one for you: it pulls thirteen Python packages that have to go into
            ComfyUI&rsquo;s own environment, and a plain copy of the folder would not work.{' '}
            <a href={PACK_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">
              Open the node pack on GitHub →</a></span>
        </p>
      )}
    </section>
  )
}

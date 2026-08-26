import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INSTALL_ALL_ACTION_LABELS, cameraInstallPlan } from '../../hooks/useSetupSteps'
import { HelpBadge } from '../../help/HelpMode'
import { fmtSize } from './fmtSize'

const POLL_MS = 1200

const ROW_META = {
  idle: { glyph: '○', cls: 'text-content-subtle', word: 'waiting' },
  queued: { glyph: '○', cls: 'text-content-subtle', word: 'queued' },
  running: { glyph: '⟳', cls: 'text-primary', word: 'downloading…' },
  success: { glyph: '✓', cls: 'text-emerald-400', word: 'done' },
  error: { glyph: '✗', cls: 'text-rose-400', word: 'needs attention' },
}

/* Where the weights come from, linked rather than merely named — the model repo
   is what the button downloads, the LoRA repo is the credit (trained by fal.ai,
   Apache-2.0), Lightning is the optional speed half. */
const MODEL_URL = 'https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI'
const LORA_URL = 'https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA'
const SPEED_URL = 'https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning'

// 📷 Camera angles — the Gallery's re-shoot lane, installable in ONE click.
//
// Deliberately NOT part of "Install everything", same rule as Krea and SeedVR2:
// the model alone is ~20 GB, and downloading a second base engine for someone
// who never pressed 📷 would be hostile on a metered link. The 409 on the button
// itself already starts these downloads — this card is the same install made
// VISIBLE, so someone setting up a machine can decide "I want this" before the
// first click, and someone whose download died can see which file needs a look.
//
// No node-pack half, and that is the difference from both siblings: the graph
// runs on stock ComfyUI nodes (pinned by test_workflow_portability), so there is
// no restart state and no "install the pack in ComfyUI" instruction — weights on
// disk IS ready.
export default function CameraInstallCard({ caps, onDone }) {
  const toast = useToast()
  const [phase, setPhase] = useState('idle')
  const [tracked, setTracked] = useState([])
  const [statuses, setStatuses] = useState({})
  const timer = useRef(null)
  const mounted = useRef(true)

  const plan = cameraInstallPlan(caps)
  const cu = caps?.comfyui || {}
  const dirValid = !!cu.dir_valid
  const ready = cu.camera_ready === true
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
          toast.warning(`${actions.length - failed.length} of ${actions.length} camera-angle `
            + 'files downloaded — the rest need a look below.')
        } else {
          toast.success('Camera angles is ready — 📷 in the Gallery.')
        }
      } else {
        timer.current = setTimeout(() => poll(actions), POLL_MS)
      }
    }).catch(() => {
      if (mounted.current) timer.current = setTimeout(() => poll(actions), POLL_MS)
    })
  }

  // Re-attach to a download still in flight (pressing 📷 in the Gallery starts
  // these too — the user may arrive here mid-download to watch it).
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
      const r = await postJson('/api/setup/install-group/camera', {})
      const actions = r.plan || []
      setTracked(actions); setStatuses(r.statuses || {})
      if (!actions.length) { setPhase('done'); onDone?.(); return }
      poll(actions)
    } catch (e) {
      setPhase('idle')
      toast.error(e.message || 'Could not start the camera-angles download.')
    }
  }

  const nothingToDownload = plan.length === 0 && phase !== 'running'

  return (
    <section className="rounded-xl border border-border bg-surface p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-x-3 gap-y-1">
        <h3 className="text-base font-semibold text-content">
          📷 Camera angles — re-shoot from another position
          <HelpBadge topic="setup-camera-install" className="ml-2" />
        </h3>
        {phase === 'running' && (
          <span className="shrink-0 text-xs font-medium tabular-nums text-content-muted">
            {doneCount} / {rows.length}
          </span>
        )}
      </div>
      <p className="mt-1 text-sm text-content-muted">
        The Gallery&apos;s 📷 button: re-photograph any picture from another camera
        position — the subject stays put, the background moves with the camera. Runs on
        Qwen-Image-Edit 2511 with a camera-control LoRA
        (<span className="whitespace-nowrap">~21.6 GB</span> all told, shared parts
        skipped when another engine already installed them), so it is installed on
        request, not by &ldquo;Install everything&rdquo;. Pressing 📷 with the weights
        absent starts this same download.
      </p>
      <p className="mt-1 text-xs text-content-subtle">
        <a href={MODEL_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Model weights →</a>
        {' · '}
        <a href={LORA_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Multiple-Angles LoRA (fal.ai) →</a>
        {' · '}
        <a href={SPEED_URL} target="_blank" rel="noreferrer" className="text-sky-300 underline hover:text-sky-200">Lightning speed LoRA →</a>
        {' — LoRAs Apache-2.0; the base model is under the Qwen licence.'}
      </p>

      {!dirValid ? (
        <p className="mt-3 text-xs text-content-subtle">
          Point the app at a valid ComfyUI folder first (the ComfyUI step) — the weights go
          inside it, under <code>models/diffusion_models/qwen</code> and <code>models/loras/qwen</code>.
        </p>
      ) : nothingToDownload ? (
        ready ? (
          <p className="mt-3 rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-sm text-content">
            ✓ Camera angles is ready — open a picture in the 🖼 Gallery and press 📷.
          </p>
        ) : (
          // Weights all present but the verdict is dark: with no node pack in
          // this lane that means ComfyUI itself is not answering — say that,
          // not "install something".
          <p className="mt-3 break-words rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-sm text-content">
            ⚠ The weights are in place, but ComfyUI is not reachable right now — start it and
            this goes green.
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
            {phase === 'running' ? 'Downloading…' : `Download Camera angles models (${plan.length})`}
          </button>
        </>
      )}
    </section>
  )
}

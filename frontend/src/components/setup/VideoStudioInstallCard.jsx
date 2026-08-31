import { useEffect, useRef, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { INSTALL_ALL_ACTION_LABELS, videoStudioInstallPlan } from '../../hooks/useSetupSteps'
import { HelpBadge } from '../../help/HelpMode'

const POLL_MS = 1200

const ROW_META = {
  idle: { glyph: '○', cls: 'text-content-subtle', word: 'waiting' },
  queued: { glyph: '○', cls: 'text-content-subtle', word: 'queued' },
  running: { glyph: '⟳', cls: 'text-primary', word: 'installing…' },
  success: { glyph: '✓', cls: 'text-emerald-400', word: 'done' },
  error: { glyph: '✗', cls: 'text-rose-400', word: 'needs attention' },
}

/* Linked rather than merely named: the weights are Comfy-Org's own conversion of
   MiniMax H3, and the turbo LoRA is somebody else's Apache-2.0 work. */
const MODEL_URL = 'https://huggingface.co/Comfy-Org/MiniMax-H3'
const TURBO_URL = 'https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora'

/* 🎬 Video Test Studio — the video lane's playback engine, installable here.
 *
 * Deliberately NOT part of "Install everything", the same rule as Krea, SeedVR2
 * and Camera angles: the required weights are 39.5 GB, and pulling a video model
 * onto the machine of someone who never opened the Video tab would be hostile on
 * a metered link.
 *
 * WHAT THIS CARD IS CAREFUL ABOUT, AND WHY
 * The lane has three kinds of gap and they are not interchangeable:
 *   * four REQUIRED weights — without them nothing renders, so they are what
 *     "ready" counts;
 *   * one optional LoRA (turbo) and three optional ComfyUI node packs. The
 *     LoRA is a download like the others; the PACKS are links, because this app
 *     downloads model files and does not add code to somebody's ComfyUI
 *     (maintainer's call, 2026-08-31: "we do not take responsibility for
 *     breaking a ComfyUI install"). Each one unlocks a single checkbox;
 *   * two files this app will NOT fetch (the latent upscaler has no verifiable
 *     source; the third-party base is opt-in). A button that cannot exist must
 *     not be drawn, so those get a sentence saying where to put the file.
 * Collapsing the three into one list is how a user ends up believing a 20 GB
 * optional download is mandatory — or worse, that the lane is broken because an
 * option they never wanted is missing.
 */
export default function VideoStudioInstallCard({ caps, onDone }) {
  const toast = useToast()
  const [phase, setPhase] = useState('idle')
  const [tracked, setTracked] = useState([])
  const [statuses, setStatuses] = useState({})
  const timer = useRef(null)
  const mounted = useRef(true)

  const plan = videoStudioInstallPlan(caps)
  const cu = caps?.comfyui || {}
  const dirValid = !!cu.dir_valid
  const ready = cu.video_studio_ready === true
  const missing = Array.isArray(cu.video_studio_missing) ? cu.video_studio_missing : []
  /* The rows the app cannot install: shown as instructions, never as buttons. */
  const byHand = missing.filter((m) => !m.action)
  /* The node packs, as rows to LINK. `present` is tri-state on purpose:
     false = the probe answered and they are absent, true = they are there,
     null/undefined = ComfyUI could not be asked, which must not be drawn as
     a red cross on somebody's working install. */
  const optionPacks = cu.video_studio_options || {}
  const UNLOCKS = {
    turbo: '⚡ Turbo (4-step clips)',
    sparse: 'sparse attention',
    latent_upscale: '🔬 latent upscale ×2',
  }
  const packRows = [
    ...Object.entries(optionPacks)
      .filter(([, o]) => o && o.pack)
      .map(([key, o]) => ({ pack: o.pack, url: o.url, search: o.search,
        present: o.available, unlocks: UNLOCKS[key] || key })),
    ...(cu.video_studio_sage?.pack
      ? [{ pack: cu.video_studio_sage.pack, url: cu.video_studio_sage.url,
        search: cu.video_studio_sage.search,
        present: cu.video_studio_sage.present,
        unlocks: 'SageAttention — a speed patch, used when present' }]
      : []),
  ]
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
        clearInterval(timer.current)
        onDone?.()
      }
    }).catch(() => {})
  }

  useEffect(() => () => { mounted.current = false; clearInterval(timer.current) }, [])

  const start = async () => {
    if (!plan.length) return
    setTracked(plan)
    setPhase('running')
    try {
      await postJson('/api/setup/install-all', { actions: plan })
      timer.current = setInterval(() => poll(plan), POLL_MS)
      poll(plan)
    } catch (e) {
      setPhase('idle')
      toast.error(e?.message || 'Could not start the install.')
    }
  }

  return (
    <section className="rounded-xl border border-border bg-surface p-5">
      <h3 className="flex items-center gap-2 text-base font-semibold text-content">
        🎬 Video Test Studio
        <span className="rounded-md border border-amber-400/40 bg-amber-400/10 px-1.5 py-0.5 text-[0.6875rem] font-semibold text-amber-200">
          beta
        </span>
        <HelpBadge topic="setup-video-studio" />
      </h3>
      <p className="mt-1 text-sm text-content-muted">
        Renders a clip from a trained video LoRA, in the Test Studio’s Video tab.
        The four required files are about <strong>39.5 GB</strong>, and this card
        downloads them — from{' '}
        <a href={MODEL_URL} target="_blank" rel="noreferrer" className="underline">
          Comfy-Org/MiniMax-H3
        </a>. The optional 4-step{' '}
        <a href={TURBO_URL} target="_blank" rel="noreferrer" className="underline">
          turbo LoRA
        </a>{' '}
        is what turns tens of minutes per clip into a few. The optional ComfyUI
        node packs below are NOT installed by this app — a weight is an inert
        file, a custom node is code your ComfyUI imports at startup.
      </p>

      {!dirValid && (
        <p className="mt-3 text-sm text-amber-300">
          Set a valid ComfyUI folder first — the weights are downloaded into it.
        </p>
      )}

      {dirValid && ready && !plan.length && (
        <p className="mt-3 text-sm text-emerald-400">
          ✓ Every weight this lane needs is on disk.
        </p>
      )}

      {dirValid && !!plan.length && (
        <>
          <ul className="mt-3 space-y-1 text-sm">
            {rows.map((a) => {
              const meta = ROW_META[(statuses[a] || {}).state || 'idle'] || ROW_META.idle
              return (
                <li key={a} className="flex items-center gap-2">
                  <span className={meta.cls}>{meta.glyph}</span>
                  <span className="min-w-0 flex-1 truncate text-content">
                    {INSTALL_ALL_ACTION_LABELS[a] || a}
                  </span>
                  <span className="shrink-0 text-content-subtle">{meta.word}</span>
                </li>
              )
            })}
          </ul>
          <button type="button" onClick={start} disabled={phase === 'running'}
            className="mt-4 rounded-lg border border-border-strong px-4 py-2 text-sm font-medium text-content hover:bg-surface-raised disabled:opacity-60">
            {phase === 'running'
              ? `Installing… (${doneCount}/${rows.length})`
              : `Install (${plan.length})`}
          </button>
          {phase === 'done' && (
            <p className="mt-2 text-sm text-content-muted">
              Weights are read on demand — nothing to restart. A node pack you
              install yourself does need a ComfyUI restart.
            </p>
          )}
        </>
      )}

      {/* The ComfyUI half, which this app names but does not install. Four
          packs, each one optional, each one a link. */}
      <div className="mt-4 rounded-lg border border-border bg-app p-3 text-sm">
        <p className="font-medium text-content">Optional — installed on the ComfyUI side</p>
        <p className="mt-1 text-content-subtle">
          These are ComfyUI custom nodes, not model files. The app does not add
          code to your ComfyUI: install them yourself through ComfyUI-Manager
          (search terms below) or by cloning, then restart ComfyUI. Everything
          renders without them — each one unlocks one checkbox.
        </p>
        <ul className="mt-2 space-y-1 text-content-muted">
          {packRows.map((r) => (
            <li key={r.pack} className="flex flex-wrap items-center gap-x-2">
              <span className={r.present === false ? 'text-content-subtle' : 'text-emerald-400'}>
                {r.present === false ? '○' : r.present ? '✓' : '·'}
              </span>
              <a href={r.url} target="_blank" rel="noreferrer" className="underline">{r.pack}</a>
              <span className="text-content-subtle">— {r.unlocks}</span>
              <span className="text-content-subtle">· search “{r.search}”</span>
            </li>
          ))}
        </ul>
      </div>

      {!!byHand.length && (
        <div className="mt-3 rounded-lg border border-border bg-app p-3 text-sm">
          <p className="font-medium text-content">Two files this app will not download</p>
          <ul className="mt-1 space-y-1 text-content-muted">
            {byHand.map((m) => (
              <li key={m.filename}>
                <code className="break-all">{m.filename}</code> → put it in{' '}
                <code>{m.place_in}</code>
                <span className="text-content-subtle">
                  {' '}— only needed for the {m.what === 'eros' ? '10Eros base' : 'latent upscale'} option.
                </span>
              </li>
            ))}
          </ul>
          <p className="mt-1 text-content-subtle">
            The upscaler has no source this app can verify, and the 10Eros base is
            somebody else’s finetune — both stay a deliberate choice, not a
            default download.
          </p>
        </div>
      )}
    </section>
  )
}

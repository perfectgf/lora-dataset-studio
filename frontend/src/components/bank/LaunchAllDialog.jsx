import { useEffect, useMemo, useRef, useState } from 'react'
import { Rocket } from 'lucide-react';
import { attemptModalSubmit } from '../../utils/submitOutcome.js'
import { flagCandidateLabel, launchRejectNote } from './autoRejectReadiness.js'
import {
  defaultPipelineStepKeys, normalizeSemanticEngine, pipelineStepKeys,
  semanticEngineLabel,
} from './bankSemanticEngine.js'

/** 🚀 Launch all — the overnight funnel. The user picks which passes run and how
 * auto-reject behaves, sees a plain "here's what will run" preview, and hits Go.
 * The backend chains the EXISTING passes in this exact order; a pass whose extra
 * isn't installed is skipped (with a reason) at run time, never failing the launch.
 *
 * Defaults: the always-available passes (scan + auto-reject) plus every heavy
 * pass whose tool is actually ready are pre-checked; captioning stays OFF by
 * default — it's the slowest GPU pass and a "clean my bank" run rarely needs a
 * description on every shot, so we make the user opt in rather than silently add
 * hours to an overnight run. Auto-reject defaults to the same flags as the
 * standalone button (blurry + flat) plus duplicate "keep best".
 */
const QUALITY_FLAGS = [
  { key: 'blur', label: 'Blurry' },
  { key: 'noise', label: 'Noisy' },
  { key: 'uniform', label: 'Flat' },
  { key: 'small', label: 'Small' },
]

export default function LaunchAllDialog({
  caps, visionReady, counts, flagsActionable, semanticEngine = 'clip', onClose, onLaunch,
}) {
  const engine = normalizeSemanticEngine(semanticEngine)
  const engineLabel = semanticEngineLabel(engine)
  // A heavy pass is "ready" when its tool is installed; scan/auto-reject always are.
  const ready = useMemo(() => ({
    scan: true,
    auto_reject: true,
    score: !!caps?.bank_scoring,
    semantic_index: engine === 'siglip2' && !!caps?.bank_siglip2,
    // Stage 2 reads whichever semantic engine this Bank selected. The preceding
    // score/index stage supplies it; the backend still skips honestly if that
    // stage did not produce a usable cache.
    semantic_dedup: engine === 'siglip2' ? !!caps?.bank_siglip2 : !!caps?.bank_scoring,
    watermark: !!visionReady,
    faces: !!caps?.face_scoring,
    framing: !!visionReady,
    caption: !!visionReady,
  }), [caps, visionReady, engine])

  const STEP_DEFS = {
    scan:
    { key: 'scan', label: 'Scan quality',
      desc: 'Sharpness, noise, flatness, size + near-duplicate groups (CPU).' },
    auto_reject:
    { key: 'auto_reject', label: 'Auto-reject flagged',
      desc: 'Reject the images carrying the flags below — reversible, nothing deleted.' },
    score:
    { key: 'score', label: 'Score', needs: 'Bank scoring extra',
      desc: 'CLIP aesthetic 1–10, NSFW, visual style and Medium (GPU).' },
    semantic_index:
    { key: 'semantic_index', label: 'Build SigLIP 2 semantic index', needs: 'SigLIP 2 Quality tool',
      desc: 'Build or resume the whole-Bank SigLIP 2 cache for semantic features. Existing CLIP data is preserved.' },
    semantic_dedup:
    { key: 'semantic_dedup', label: '✂ Find crops & variants',
      needs: engine === 'siglip2' ? 'SigLIP 2 Quality tool' : 'Bank scoring extra',
      desc: `Group crops/variants of the same shot from the ${engineLabel} semantic index (needs that index first).` },
    watermark:
    { key: 'watermark', label: 'Find watermarks', needs: 'Vision model',
      desc: 'Detect overlaid watermarks/logos with the Qwen3-VL detector (GPU).' },
    faces:
    { key: 'faces', label: 'Group by person', needs: 'Quality tools',
      desc: 'Face embeddings + person clusters, no reference photo (CPU/GPU).' },
    framing:
    { key: 'framing', label: 'Classify framing', needs: 'Vision model',
      desc: 'Tag each shot face/bust/body/back — powers the framing filter & coverage advice (GPU).' },
    caption:
    { key: 'caption', label: 'Caption', needs: 'Caption engine',
      desc: 'Describe every image so it becomes searchable and rides to the dataset (GPU).' },
  }
  const STEPS = pipelineStepKeys(engine).map((key) => STEP_DEFS[key])

  const [steps, setSteps] = useState(() => new Set(defaultPipelineStepKeys(engine, ready)))
  const [rejectFlags, setRejectFlags] = useState(() => new Set(['blur', 'uniform']))
  const [resolveDups, setResolveDups] = useState(true)

  const toggleStep = (k) => setSteps((prev) => {
    const next = new Set(prev)
    if (next.has(k)) next.delete(k); else next.add(k)
    return next
  })
  const toggleFlag = (k) => setRejectFlags((prev) => {
    const next = new Set(prev)
    if (next.has(k)) next.delete(k); else next.add(k)
    return next
  })

  const autoRejectOn = steps.has('auto_reject')
  // Honest about the ORDER: auto-reject runs after the scan here, so the counts
  // shown next to the flags are a floor, and images nothing has ever measured
  // are invisible to every flag until 🔎 Scan reaches them.
  const rejectNote = launchRejectNote(counts, steps.has('scan'))
  // The honest preview: the steps that will actually RUN, in order, tagged when
  // one will be skipped because its tool isn't ready.
  const plan = STEPS.filter((s) => steps.has(s.key)).map((s) => ({
    ...s, willSkip: !ready[s.key],
  }))
  const nRun = plan.filter((s) => !s.willSkip).length

  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  /* ONE way out, shut only while the launch is being posted. */
  const dismiss = () => { if (!busy) onClose() }
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') dismiss() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])  // eslint-disable-line react-hooks/exhaustive-deps

  /* One bank, one background job: this dialog is refused whenever another pass
     owns the bank — the single most likely answer for an overnight funnel the
     user just spent a minute configuring. Closing first reset all seven
     checkboxes and the reject flags to their defaults. The card scrolls inside
     itself, so scroll to the end where the message and the button that produced
     it live. */
  const cardRef = useRef(null)
  useEffect(() => {
    const card = cardRef.current
    if (error && card) card.scrollTop = card.scrollHeight
  }, [error])

  const launch = async () => {
    if (busy) return
    setBusy(true)
    setError(null)
    let outcome
    try {
      outcome = await attemptModalSubmit(() => onLaunch({
        steps: [...steps],
        reject_flags: autoRejectOn ? [...rejectFlags] : [],
        resolve_dups: autoRejectOn && resolveDups,
      }), { fallback: 'Could not start the run' })
    } finally { setBusy(false) }
    if (outcome.close) onClose()
    else setError(outcome.error)
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Launch all"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget) dismiss() }}>
      <div ref={cardRef}
        className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-border bg-surface-overlay p-5 shadow-2xl space-y-4">
        <div>
          <h2 className="flex items-center gap-2 text-base font-bold text-content"><Rocket aria-hidden="true" className="h-4 w-4" /> Launch all</h2>
          <p className="mt-1 text-sm text-content-muted">
            Chain the whole triage in one go — start it, walk away, come back to a
            cleaned bank. Each pass runs in order; you can Stop it any time, and a
            pass whose tool isn't installed is skipped (never fails the run).
          </p>
        </div>

        <ul className="space-y-1.5">
          {STEPS.map((s) => (
            <li key={s.key}>
              <label className="flex items-start gap-2 rounded-md border border-border bg-surface p-2 text-sm">
                <input type="checkbox" className="mt-0.5" checked={steps.has(s.key)}
                  onChange={() => toggleStep(s.key)} />
                <span className="min-w-0">
                  <span className="font-medium text-content">{s.label}</span>
                  {!ready[s.key] && (
                    <span className="ml-1.5 rounded bg-amber-500/15 px-1.5 py-px text-[10px] font-semibold text-amber-300">
                      {s.needs} not ready — will skip
                    </span>
                  )}
                  <span className="block text-xs text-content-subtle">{s.desc}</span>
                </span>
              </label>
              {s.key === 'auto_reject' && autoRejectOn && (
                <div className="ml-6 mt-1.5 space-y-2 rounded-md border border-border bg-surface p-2">
                  <p className="text-xs text-content-muted">
                    Reject the still-undecided images with these flags (manual ✓/✕ are never touched):
                  </p>
                  {/* The count is what the flag would catch RIGHT NOW — undecided
                      images only, the same pile the pass touches. It is not the
                      outcome: 🔎 Scan runs before auto-reject in this funnel, so
                      the note below says which way the number will move rather
                      than letting a stale figure pass for a promise. */}
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    {QUALITY_FLAGS.map((f) => (
                      <label key={f.key} className="flex items-center gap-1.5 text-sm text-content">
                        <input type="checkbox" checked={rejectFlags.has(f.key)}
                          onChange={() => toggleFlag(f.key)} />
                        {f.label}
                        <span className="text-xs text-content-subtle">
                          ({flagCandidateLabel(f.key, flagsActionable)})
                        </span>
                      </label>
                    ))}
                  </div>
                  {rejectNote && (
                    <p className="m-0 text-[0.6875rem] leading-snug text-amber-200">
                      ⚠ {rejectNote}
                    </p>
                  )}
                  <label className="flex items-center gap-1.5 text-sm text-content">
                    <input type="checkbox" checked={resolveDups}
                      onChange={(e) => setResolveDups(e.target.checked)} />
                    ≈ Duplicates → keep the best, reject the rest
                  </label>
                </div>
              )}
            </li>
          ))}
        </ul>

        <div className="rounded-md border border-border bg-surface-raised p-3 text-sm">
          <p className="font-semibold text-content">What will run</p>
          {nRun === 0 ? (
            <p className="text-content-muted">Nothing selected yet — pick at least one pass.</p>
          ) : (
            <ol className="mt-1 list-decimal pl-5 text-content-muted space-y-0.5">
              {plan.map((s) => (
                <li key={s.key} className={s.willSkip ? 'line-through opacity-60' : ''}>
                  {s.label}
                  {s.key === 'auto_reject' && !s.willSkip && (
                    <span className="text-content-subtle">
                      {' '}({[...rejectFlags].length
                        ? [...rejectFlags].join(', ')
                        : 'no flags'}{resolveDups ? ' + duplicates' : ''})
                    </span>
                  )}
                  {s.willSkip && <span className="text-amber-300"> — skipped</span>}
                </li>
              ))}
            </ol>
          )}
        </div>

        {/* The refusal, right above the button that produced it — the passes you
            ticked are still ticked. shrink-0 keeps it from being squashed to a
            clipped sliver, max-h-24 keeps a long sentence from pushing 🚀 Launch
            off a 400-px screen. */}
        {error && (
          <div role="alert"
            className="shrink-0 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 max-h-24 overflow-y-auto">
            <span className="block whitespace-pre-wrap break-words text-xs leading-relaxed text-red-200">
              {error}
            </span>
            <span className="mt-1 block text-[0.625rem] text-content-subtle">
              Your selection is kept — adjust and try again.
            </span>
          </div>
        )}

        <div className="flex justify-end gap-2">
          <button type="button" onClick={dismiss} disabled={busy}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:text-content hover:bg-surface-raised disabled:opacity-50">
            Cancel
          </button>
          <button type="button" onClick={launch} disabled={busy || nRun === 0}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50">
            <Rocket aria-hidden="true" className="mr-1 inline h-3.5 w-3.5 align-[-2px]" />{busy ? 'Starting…' : `Launch${nRun ? ` ${nRun} pass${nRun > 1 ? 'es' : ''}` : ''}`}
          </button>
        </div>
      </div>
    </div>
  )
}

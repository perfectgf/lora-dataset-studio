import { useMemo, useState } from 'react'

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
  { key: 'blur', label: '🌫 Blurry' },
  { key: 'noise', label: '📺 Noisy' },
  { key: 'uniform', label: '⬜ Flat' },
  { key: 'small', label: '📐 Small' },
]

export default function LaunchAllDialog({ caps, visionReady, onClose, onLaunch }) {
  // A heavy pass is "ready" when its tool is installed; scan/auto-reject always are.
  const ready = useMemo(() => ({
    scan: true,
    auto_reject: true,
    score: !!caps?.bank_scoring,
    // Stage 2 reuses Score's embeddings — ready exactly when Score is (and it's
    // skipped at run time if Score didn't actually produce any).
    semantic_dedup: !!caps?.bank_scoring,
    watermark: !!visionReady,
    faces: !!caps?.face_scoring,
    caption: !!visionReady,
  }), [caps, visionReady])

  const STEPS = [
    { key: 'scan', label: '🔎 Scan quality',
      desc: 'Sharpness, noise, flatness, size + near-duplicate groups (CPU).' },
    { key: 'auto_reject', label: '🧹 Auto-reject flagged',
      desc: 'Reject the images carrying the flags below — reversible, nothing deleted.' },
    { key: 'score', label: '✨ Score', needs: 'Bank scoring extra',
      desc: 'Aesthetic 1–10, NSFW, style groups (GPU).' },
    { key: 'semantic_dedup', label: '✂ Find crops & variants', needs: 'Bank scoring extra',
      desc: 'Group crops/variants of the same shot from Score’s embeddings — no extra GPU (needs Score first).' },
    { key: 'watermark', label: '🚩 Find watermarks', needs: 'Vision model',
      desc: 'Detect overlaid watermarks/logos with the Qwen3-VL detector (GPU).' },
    { key: 'faces', label: '👥 Group by person', needs: 'Quality tools',
      desc: 'Face embeddings + person clusters, no reference photo (CPU/GPU).' },
    { key: 'caption', label: '🏷️ Caption', needs: 'Caption engine',
      desc: 'Describe every image so it becomes searchable and rides to the dataset (GPU).' },
  ]

  const [steps, setSteps] = useState(() => new Set(
    ['scan', 'auto_reject', 'score', 'semantic_dedup', 'watermark', 'faces']
      .filter((k) => ready[k])))
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
  // The honest preview: the steps that will actually RUN, in order, tagged when
  // one will be skipped because its tool isn't ready.
  const plan = STEPS.filter((s) => steps.has(s.key)).map((s) => ({
    ...s, willSkip: !ready[s.key],
  }))
  const nRun = plan.filter((s) => !s.willSkip).length

  const launch = () => {
    onLaunch({
      steps: [...steps],
      reject_flags: autoRejectOn ? [...rejectFlags] : [],
      resolve_dups: autoRejectOn && resolveDups,
    })
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Launch all"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-border bg-surface-overlay p-5 shadow-2xl space-y-4">
        <div>
          <h2 className="text-base font-bold text-content">🚀 Launch all</h2>
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
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    {QUALITY_FLAGS.map((f) => (
                      <label key={f.key} className="flex items-center gap-1.5 text-sm text-content">
                        <input type="checkbox" checked={rejectFlags.has(f.key)}
                          onChange={() => toggleFlag(f.key)} />
                        {f.label}
                      </label>
                    ))}
                  </div>
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

        <div className="rounded-md border border-indigo-400/40 bg-indigo-500/10 p-3 text-sm">
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

        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="button" onClick={launch} disabled={nRun === 0}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
            🚀 Launch{nRun ? ` ${nRun} pass${nRun > 1 ? 'es' : ''}` : ''}
          </button>
        </div>
      </div>
    </div>
  )
}

import { useEffect, useState } from 'react'
import {
  NR_DEFAULTS, NR_PRESETS, TEMPORAL_MODES, STRENGTH_MAX, PASSES_MAX, normalizeNrParams, presetFor,
  temporalOutcome, nrRefusal, costMultiplier,
} from './neuralRenderParams'
import { HelpBadge } from '../../help/HelpMode'

/** ✨ Neural render (DLSS 5) — the dials, asked ONCE, before a render.
 *
 * ONE dialog for the two hosts (the video dataset's clips and the studio's clip
 * history): same dials, same words, same refusal sentence. The host says what
 * is about to be rendered (`subject`) and what happens to it (`consequence`),
 * because that is the one thing that differs — in the dataset the render
 * REPLACES the clip (original kept, restorable), in the studio it is a NEW clip.
 *
 * The capability drives the button, never hides it: a machine without the
 * model reads WHY, in the backend's own sentences, and where to put the file.
 */
export default function NeuralRenderDialog({
  status, subject, consequence, width = null, busy = false, initial = null,
  onRender, onClose,
}) {
  const [params, setParams] = useState(() => normalizeNrParams(initial || NR_DEFAULTS))
  const refusal = nrRefusal(status)
  const preset = presetFor(params)

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose?.() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const set = (patch) => setParams((p) => normalizeNrParams({ ...p, ...patch }))
  const dial = (key, label, hint, max = 2, step = 0.05) => (
    <label className="flex flex-col gap-1 text-xs text-content-muted">
      <span className="flex items-center justify-between">
        <span>{label}</span>
        <span className="font-mono tabular-nums text-content">{params[key].toFixed(2)}</span>
      </span>
      <input type="range" min="0" max={max} step={step} value={params[key]}
        aria-label={label} onChange={(e) => set({ [key]: e.target.value })}
        className="w-full" />
      <span className="text-[0.6875rem] text-content-subtle">{hint}</span>
    </label>
  )

  /* z-[9990], not z-50: this dialog opens over surfaces that carry their own
     fixed chrome — the Studio's action bar is z-[9960] — and a dialog under the
     bar it was opened from puts its own ✨ Render button out of reach. Measured
     at 360 px before the fix: the button was IN the viewport (so a "is it
     visible" check passed) while elementFromPoint at its centre returned a pill
     of the action bar. Same tier as ContinueDialog. */
  return (
    <div role="dialog" aria-modal="true" aria-label="Neural render settings" data-probe-layer
      className="fixed inset-0 z-[9990] flex items-end justify-center bg-black/70 p-2 sm:items-center sm:p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose?.() }}>
      {/* max-h + scroll like every other dialog in the app (PassDialog,
          PromoteDialog, ScoringPythonDialog…): this one had neither, so on a
          phone its content ran past both edges of the viewport with nothing to
          scroll — the title was cut off at the top and the footer at the bottom. */}
      <div className="flex max-h-[90vh] w-full max-w-md flex-col gap-3 overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 shadow-xl">
        <div className="flex items-start justify-between gap-2">
          <h2 className="text-sm font-semibold text-content">
            ✨ Neural render <span className="font-normal text-content-muted">(DLSS 5)</span>
            <HelpBadge topic="video-neural-render" className="ml-2" />
          </h2>
          <button type="button" onClick={onClose} aria-label="Close"
            className="min-h-10 rounded-md border border-border px-2 text-sm text-content-muted hover:text-content lg:min-h-0">✕</button>
        </div>

        <p className="text-xs text-content-muted">{subject} {consequence}</p>

        {refusal ? (
          <p role="status" className="rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-content">
            {refusal}
          </p>
        ) : null}

        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Starting point">
          {NR_PRESETS.map((p) => (
            <button key={p.id} type="button" onClick={() => set(p.params)} aria-pressed={preset === p.id}
              className={`min-h-10 rounded-full border px-3 py-0.5 text-[0.6875rem] font-semibold lg:min-h-0 ${
                preset === p.id ? 'border-border-strong bg-surface-raised text-content'
                  : 'border-border text-content-muted hover:text-content'}`}>
              {p.label}
            </button>
          ))}
        </div>

        {dial('tone', 'Tone', 'How much the model relights. 0 keeps the clip\'s own tones — the setting for flat art, where 1 greys the whites.')}
        {dial('structure', 'Structure', 'How much micro-detail is added to skin, hair and fabric.')}

        <label className="flex items-center gap-2 text-xs text-content-muted">
          <input type="checkbox" checked={params.automask} onChange={(e) => set({ automask: e.target.checked })} />
          Automatic mask <span className="text-content-subtle">(the model decides where it acts; marginal)</span>
        </label>

        {/* The three levers the model does not expose — the ones that make the
            difference visible. Strength past 1 carries the render beyond the
            model's own answer (what the game mod calls Detail strength);
            passes feed the answer back through; 2x works on four times the
            pixels and delivers at the clip's size. Priced on the button. */}
        {dial('strength', 'Strength', `1 is the model's picture. Above it carries on past it — the fastest way to a visible change; 2 roughly doubles the added detail, ${STRENGTH_MAX} triples it.`, STRENGTH_MAX, 0.1)}
        <fieldset className="flex flex-col gap-1">
          <legend className="text-xs text-content-muted">Passes</legend>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Passes">
            {Array.from({ length: PASSES_MAX }, (_, i) => i + 1).map((n) => (
              <button key={n} type="button" onClick={() => set({ passes: n })} aria-pressed={params.passes === n}
                className={`min-h-10 rounded-full border px-3 py-0.5 text-[0.6875rem] font-semibold lg:min-h-0 ${
                  params.passes === n ? 'border-border-strong bg-surface-raised text-content'
                    : 'border-border text-content-muted hover:text-content'}`}>
                {n}
              </button>
            ))}
          </div>
          <p className="text-[0.6875rem] text-content-subtle">Each extra pass feeds the render back through the model. Extra passes run in still mode.</p>
        </fieldset>
        <label className="flex items-center gap-2 text-xs text-content-muted">
          <input type="checkbox" checked={params.scale === 2} onChange={(e) => set({ scale: e.target.checked ? 2 : 1 })} />
          Render at 2× <span className="text-content-subtle">(finer detail, four times the work; the clip keeps its size)</span>
        </label>

        <fieldset className="flex flex-col gap-1">
          <legend className="text-xs text-content-muted">Frames</legend>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Temporal mode">
            {TEMPORAL_MODES.map((m) => (
              <button key={m.id} type="button" onClick={() => set({ temporal: m.id })}
                aria-pressed={params.temporal === m.id} title={m.hint}
                className={`min-h-10 rounded-full border px-3 py-0.5 text-[0.6875rem] font-semibold lg:min-h-0 ${
                  params.temporal === m.id ? 'border-border-strong bg-surface-raised text-content'
                    : 'border-border text-content-muted hover:text-content'}`}>
                {m.label}
              </button>
            ))}
          </div>
          <p className="text-[0.6875rem] text-content-subtle">
            {TEMPORAL_MODES.find((m) => m.id === params.temporal)?.hint} → {temporalOutcome(params.temporal, width, params.passes)}.
          </p>
        </fieldset>

        {/* Pinned to the bottom of the scroll area: the dials above are long
            enough on a phone that a footer scrolling with them is a footer the
            user has to go looking for. The negative margins let the bar span
            the panel's full width under its own padding. */}
        <div className="sticky bottom-0 -mx-4 -mb-4 mt-1 flex items-center justify-end gap-2 border-t border-border bg-surface-overlay px-4 py-3">
          <button type="button" onClick={onClose}
            className="min-h-10 rounded-md border border-border px-3 py-1 text-sm text-content-muted hover:text-content lg:min-h-0">
            Cancel
          </button>
          <button type="button" disabled={!!refusal || busy}
            onClick={() => onRender?.(normalizeNrParams(params))}
            title={refusal || undefined}
            className="min-h-10 rounded-md border border-border-strong bg-surface-raised px-3 py-1 text-sm font-semibold text-content hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50 lg:min-h-0">
            {busy ? '…' : costMultiplier(params) > 1 ? `✨ Render (≈ ×${costMultiplier(params)} time)` : '✨ Render'}
          </button>
        </div>
      </div>
    </div>
  )
}

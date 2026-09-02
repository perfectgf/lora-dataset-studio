import { useEffect, useState } from 'react'
import {
  NR_DEFAULTS, NR_PRESETS, TEMPORAL_MODES, normalizeNrParams, presetFor,
  temporalOutcome, nrRefusal,
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
  const dial = (key, label, hint) => (
    <label className="flex flex-col gap-1 text-xs text-content-muted">
      <span className="flex items-center justify-between">
        <span>{label}</span>
        <span className="font-mono tabular-nums text-content">{params[key].toFixed(2)}</span>
      </span>
      <input type="range" min="0" max="2" step="0.05" value={params[key]}
        aria-label={label} onChange={(e) => set({ [key]: e.target.value })}
        className="w-full" />
      <span className="text-[0.6875rem] text-content-subtle">{hint}</span>
    </label>
  )

  return (
    <div role="dialog" aria-modal="true" aria-label="Neural render settings" data-probe-layer
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 p-2 sm:items-center sm:p-4"
      onClick={(e) => { if (e.target === e.currentTarget) onClose?.() }}>
      <div className="flex w-full max-w-md flex-col gap-3 rounded-xl border border-border bg-surface-overlay p-4 shadow-xl">
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
            {TEMPORAL_MODES.find((m) => m.id === params.temporal)?.hint} → {temporalOutcome(params.temporal, width)}.
          </p>
        </fieldset>

        <div className="flex items-center justify-end gap-2 pt-1">
          <button type="button" onClick={onClose}
            className="min-h-10 rounded-md border border-border px-3 py-1 text-sm text-content-muted hover:text-content lg:min-h-0">
            Cancel
          </button>
          <button type="button" disabled={!!refusal || busy}
            onClick={() => onRender?.(normalizeNrParams(params))}
            title={refusal || undefined}
            className="min-h-10 rounded-md border border-border-strong bg-surface-raised px-3 py-1 text-sm font-semibold text-content hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50 lg:min-h-0">
            {busy ? '…' : '✨ Render'}
          </button>
        </div>
      </div>
    </div>
  )
}

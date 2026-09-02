/**
 * ▶ The launch window of the video pipeline — which preparation passes the
 * chain will run, before it runs them.
 *
 * The button used to chain probe → find shots → thumbnails while its own
 * tooltip promised "measure, embeddings and the rest": a fresh bank still cost
 * four to nine clicks after the one button that claimed to do it all. The chain
 * now offers every preparation pass, and this window is where "everything"
 * stops being a guess — the image lane asks the same question in the same shape
 * (LaunchAllDialog).
 *
 * 🗣 Describe is NOT offered here, deliberately: its wording changes what the
 * captions say (measured), and that choice belongs at the moment of ITS click,
 * not to a chain someone starts before walking away.
 */
import { useEffect, useState } from 'react'
import { PASS_LABELS } from './videoBankStatus'
import { passBlockedBy } from './videoCapability'

/* The order the chain runs them in — inputs first. Each row says what it BUYS,
   because "measure" alone does not tell anyone whether they want it. */
export const PIPELINE_ROWS = [
  { key: 'probe', locked: true,
    why: 'Reads each file — duration, frame rate, size. Everything else needs it.' },
  { key: 'detect', locked: true,
    why: 'Cuts the files into shots. The rest of the lane works on shots.' },
  { key: 'thumbs', locked: true,
    why: 'One image per shot, so the gallery is browsable.' },
  { key: 'measure',
    why: 'Motion, sharpness, darkness, audio — what the 🎚 Quality cuts filter on.' },
  { key: 'embed',
    why: 'The vectors 🔎 search and ✂ Duplicates both read. The slowest step here.' },
  { key: 'dedup',
    why: 'Groups shots that repeat, so you reject a pile in one gesture. Needs the vectors above.' },
  { key: 'camera',
    why: 'Names the camera move of each shot — the 🎥 facet, and the Camera: line of every training prompt.' },
]

export default function RunEverythingDialog({ capability, onClose, onLaunch }) {
  const [picked, setPicked] = useState(
    () => new Set(PIPELINE_ROWS.map((r) => r.key)))
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose?.() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])

  const toggle = (key) => setPicked((prev) => {
    const next = new Set(prev)
    if (next.has(key)) next.delete(key)
    else next.add(key)
    return next
  })

  const rows = PIPELINE_ROWS.map((r) => ({ ...r, blocked: passBlockedBy(capability, r.key) }))
  // A step this install cannot run is never sent: the chain would stop on it
  // and the passes after it would never run at all.
  const steps = rows.filter((r) => picked.has(r.key) && !r.blocked).map((r) => r.key)

  const submit = async (e) => {
    e.preventDefault()
    if (busy || !steps.length) return
    setBusy(true)
    try { await onLaunch(steps) } finally { setBusy(false) }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Run everything" data-probe-layer
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-2 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose?.() }}>
      <form onSubmit={submit}
        className="flex w-full max-w-lg max-h-[92vh] flex-col overflow-hidden rounded-xl border border-border bg-surface-overlay shadow-2xl">
        <header className="shrink-0 space-y-1 border-b border-border p-4">
          <h2 className="text-base font-bold text-content">▶ Run everything</h2>
          <p className="text-sm text-content-muted">
            The preparation passes, chained, in the order each one needs the
            previous. Start it and walk away.
          </p>
          <p className="text-xs text-content-subtle">
            🗣 Describe shots is not in here — its wording changes what the
            captions say, so it stays its own button.
          </p>
        </header>

        <div className="min-h-0 flex-1 space-y-1.5 overflow-y-auto p-3 sm:p-4">
          {rows.map((r) => (
            <label key={r.key}
              className={`flex cursor-pointer items-start gap-2 rounded-md border border-border px-3 py-2 text-sm text-content ${
                r.blocked ? 'opacity-60' : 'bg-surface-raised has-[:checked]:border-sky-400/70 has-[:checked]:bg-sky-500/10'}`}>
              <input type="checkbox" className="mt-0.5"
                checked={picked.has(r.key) && !r.blocked}
                disabled={r.locked || !!r.blocked}
                onChange={() => toggle(r.key)} />
              <span className="min-w-0">
                <span className="font-semibold">{PASS_LABELS[r.key]}</span>
                {r.locked && (
                  <span className="ml-2 text-[0.6875rem] text-content-subtle">always</span>
                )}
                <span className="block text-xs text-content-muted">{r.why}</span>
                {r.blocked && (
                  <span className="block text-xs text-amber-300">{r.blocked.why}</span>
                )}
              </span>
            </label>
          ))}
        </div>

        <div className="shrink-0 border-t border-border p-3">
          <div className="flex flex-wrap items-center justify-end gap-2">
            <span className="mr-auto text-xs text-content-subtle">
              {steps.length} pass(es) will run.
            </span>
            <button type="button" onClick={onClose} disabled={busy}
              className="min-h-10 rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50 lg:min-h-0">
              Cancel
            </button>
            <button type="submit" disabled={busy || !steps.length}
              className="min-h-10 rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50 lg:min-h-0">
              {busy ? 'Starting…' : '▶ Run'}
            </button>
          </div>
        </div>
      </form>
    </div>
  )
}

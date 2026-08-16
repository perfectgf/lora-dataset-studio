/**
 * The Bank's two status bars: what a running pass is doing (⏳ ProgressBar) and
 * the net under the last bulk gesture (↩ UndoBar).
 *
 * Moved out of BankWorkspace.jsx by the Encre redesign. In structure B both of
 * these belong to the TOP BAR — they are bank-wide facts, not part of the
 * triage column — so they now live beside it instead of inside a 3 400-line
 * component that also drew the grid.
 */
import { useState } from 'react'
import { STEP_SHORT } from './bankFacets.js'
import { UNDO_HINT, undoBannerText } from './bankUndo.js'
import { etaPhrase } from './passEta.js'
import { jobKey, stopLabel, stopNote, stopRequested } from './passStop.js'
import {
  PROGRESS_HIDDEN, PROGRESS_STALE, PROGRESS_UNKNOWN, progressPresence,
} from './progressPresence.js'

/* No contact with the server, so no fresh job snapshot. Saying NOTHING here is
   the bug this replaces: a pass that is running perfectly well in the bank's
   server-side thread looked exactly like a pass that had stopped. Plain text,
   NOT a live region — the global ConnectionBanner already announces the outage
   once, and a second live region would double every announcement. */
export function ProgressUnknown({ stale }) {
  return (
    <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm text-content-muted">
      <span aria-hidden>📡</span>{' '}
      {stale
        ? 'Lost contact — the progress above is the last thing we heard. The pass keeps running on the server.'
        : 'Lost contact — can’t read job progress right now. Anything you started keeps running on the server; this comes back when the connection does.'}
    </p>
  )
}

/** ↩ The net under the bank's biggest gesture.
 *
 * Deliberately NOT a toast: a bar that vanishes after four seconds is unusable
 * for anyone who reads slowly, and this is the one control you reach for after
 * realising you just marked 400 images wrong. It stays until it is used,
 * dismissed, or replaced by a newer action — and it is fed by the bank payload,
 * so it is still there after a reload, unlike an undo that only ever lived in
 * this tab.
 *
 * `role="status"` + `aria-live="polite"` so a screen reader is told the net
 * exists at the moment the bulk action lands, without stealing focus from the
 * grid. Both controls are real buttons, in tab order.
 */
export function UndoBar({ offer, busy, onUndo, onDismiss }) {
  if (!offer) return null
  return (
    <div role="status" aria-live="polite"
      className="flex flex-wrap items-center gap-x-3 gap-y-2 rounded-lg border border-sky-400/40 bg-sky-500/10 px-3 py-2 text-sm text-content">
      <p className="m-0 min-w-0 grow basis-full sm:basis-auto">
        <span aria-hidden>↩</span>{' '}
        <span className="font-semibold">{undoBannerText(offer)}</span>
        <span className="block text-xs text-content-muted sm:inline sm:before:content-['_—_']">
          {UNDO_HINT}
        </span>
      </p>
      <button type="button" onClick={onUndo} disabled={busy}
        className="rounded border border-sky-400/60 px-2 py-1 text-xs font-semibold hover:bg-white/10 disabled:opacity-50">
        {busy ? 'Undoing…' : '↩ Undo'}
      </button>
      <button type="button" onClick={onDismiss} disabled={busy}
        title="Keep the change and hide this"
        className="rounded border border-border px-2 py-1 text-xs text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50">
        Dismiss
      </button>
    </div>
  )
}

// Exported for the render test: what this bar says during a step with no
// per-image counter is the whole fix for "the pass looks frozen", and a source
// regex cannot see what the renderer produces.
export function ProgressBar({ activity, onCancel, offline = false }) {
  /* Held HERE and not in the parent on purpose: the click has to be answered
     without waiting for the payload that takes 2.7 s to rebuild on a bank this
     size. Keyed by job, so the next pass gets a live button back with no effect
     and no cleanup — see passStop.js for the measurements behind this. */
  const [stopAskedFor, setStopAskedFor] = useState(null)
  const presence = progressPresence(activity, offline)
  if (presence === PROGRESS_HIDDEN) return null
  if (presence === PROGRESS_UNKNOWN) return <ProgressUnknown />
  const stale = presence === PROGRESS_STALE
  const { kind, done, total, detail } = activity
  const pct = total > 0 ? Math.round((100 * done) / total) : null
  // "12939 / 37800" says where the pass is; it never said how long that leaves.
  // On a bank this size the difference between twenty minutes and four hours is
  // the difference between waiting and going to do something else.
  const eta = etaPhrase(activity)
  const asked = stopRequested(activity, stopAskedFor)
  const note = stopNote(activity, asked)
  const pipe = kind === 'pipeline' ? activity.pipeline : null
  return (
    <div className="space-y-2 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-sm">
      {/* flex-wrap: at 400 px the label, the bar and Stop cannot share one row —
          they used to squash the label to a sliver. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span aria-hidden>⏳</span>
        <span className={`text-content ${stale ? 'opacity-60' : ''}`}>
          {pipe
            ? `🚀 Launch all — step ${(pipe.index ?? 0) + 1}/${pipe.total_steps} · ${STEP_SHORT[pipe.current] || pipe.current}`
            : ({ scan: 'Quality scan', faces: 'Face pass', score: 'Scoring pass',
              semantic_index: 'Semantic index',
              semantic_dedup: 'Crops & variants', watermark: 'Watermark scan',
              framing: 'Framing pass', caption: 'Captioning', promote: 'Promotion',
              medium: 'Medium pass', angles: 'Measuring head angles',
              // The longest pass in the app (one GPU render per image). A run
              // that costs an hour must not sit under the anonymous fallback.
              improve: 'Upscale & improve',
              bank_promote: 'Copying into the new bank',
              // The one destructive pass: it must NAME itself in the bar, not
              // ride under the anonymous "Job running" fallback.
              delete_rejected: 'Deleting rejected files' }[kind] || 'Job') + ' running'}
          {/* A step with no per-image counter publishes done=total=0 and says
              what it is doing in words (the cache write, the style grouping).
              Printing a bare "0" next to that sentence would read as "0 done"
              on work that is running — so the figure only appears when there
              is one. */}
          {(done || total) ? <>{' — '}{done}{total ? ` / ${total}` : ''}</> : null}
          {eta ? `${(done || total) ? ' · ' : ' — '}${eta}` : ''}
          {detail ? `${(done || total || eta) ? ' · ' : ' — '}${detail}` : ''}
        </span>
        {pct != null && (
          <div className="h-1.5 w-40 overflow-hidden rounded bg-surface-raised" role="progressbar"
            aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full bg-amber-400" style={{ width: `${pct}%` }} />
          </div>
        )}
        {/* `disabled` is the whole of requirement one: seven POSTs in 20 ms is
            what an enabled button that looks identical after the click buys.
            `title` carries the promise for the pointer that hovers before it
            presses; the line below carries it for everyone else. */}
        <button type="button" disabled={asked} title={note || undefined}
          onClick={() => { setStopAskedFor(jobKey(activity)); onCancel?.() }}
          className="ml-auto rounded-md border border-border px-2 py-0.5 text-xs text-content hover:bg-surface-raised disabled:cursor-default disabled:opacity-60 disabled:hover:bg-transparent">
          {stopLabel(asked)}
        </button>
      </div>
      {/* One line, xs, muted — and it REPLACES the Stop sentence that used to
          ride inside the style-grouping phase label, where everyone read it all
          the time and it wrapped the counter off its row at 400 px.
          aria-live so the answer to the click reaches a screen reader too. */}
      {note && (
        <p className="m-0 pl-6 text-xs text-content-muted" aria-live="polite">{note}</p>
      )}
      {stale && <ProgressUnknown stale />}
      {pipe && Array.isArray(pipe.results) && pipe.results.length > 0 && (
        <ul className="flex flex-wrap gap-1.5 pl-6 text-xs">
          {pipe.results.map((r, i) => (
            <li key={`${r.step}-${i}`}
              className={`rounded px-1.5 py-px ${r.status === 'done' ? 'bg-emerald-500/15 text-emerald-300'
                : r.status === 'error' ? 'bg-rose-500/15 text-rose-300'
                : 'bg-black/20 text-content-subtle'}`}
              title={r.reason || r.detail || ''}>
              {r.status === 'done' ? '✅' : r.status === 'error' ? '⚠️' : '⏭️'} {STEP_SHORT[r.step] || r.step}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

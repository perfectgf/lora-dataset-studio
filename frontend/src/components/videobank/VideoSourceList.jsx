import {
  formatDuration, formatFileSize, sourceEncoding, sourceGeometry, sourceState,
} from './videoBankStatus'
import { firstShotBounds } from './videoClipEdit'
import { canRecut } from './videoShotCuts'

// The per-file actions share one look: they are peers, and a difference in
// weight between them would read as a difference in consequence.
const ACTION = 'rounded border border-border bg-surface-raised px-1.5 py-0.5 '
  + 'text-[0.625rem] font-semibold text-content-muted hover:bg-surface'

const TONE = {
  ok: 'bg-emerald-500/15 text-emerald-200',
  info: 'bg-sky-500/15 text-sky-200',
  idle: 'bg-surface-raised text-content-subtle',
  error: 'bg-rose-500/15 text-rose-200',
}

/** 🎬 The per-FILE view: what each rush is, and how far it got.
 *
 * Cards rather than a table, and that is a 400 px decision: a table with five
 * columns of which one is an unbreakable path either scrolls the whole page
 * sideways or truncates the only column that identifies the row.
 *
 * Clicking a file filters the shot grid to it — the question this list actually
 * gets asked is "what came out of THAT file", and answering it by scrolling a
 * grid of three hundred shots is not answering it.
 *
 * ✂ CUT A SHOT BY HAND lives here and not only in the lightbox, and that is the
 * whole point of it: every other retouch gesture needs an open shot, and a file
 * that detection missed entirely — or a bank on an install with no detector, which
 * the app explicitly says can still "scan, cut, watch and triage" — has none. It
 * makes the first shot, which the lightbox then trims and splits.
 *
 * ▣ SINGLE SHOT AND ↻ RE-CUT ARE PER-FILE ON PURPOSE. A folder of rushes is
 * mixed inside itself: the untouched phone clip and the tightly edited scene sit
 * next to each other, and there is no bank-wide number that is right for both.
 * Both buttons act on the ONE file whose name is on the card, which is also what
 * lets ↻ replace hand-made cuts — it is the way back from ▣, and it says so
 * before it does it.
 */
export default function VideoSourceList({
  sources, activeSourceId, onFilter, onCut, onSingleShot, onRecut,
}) {
  if (!sources?.length) {
    return (
      <p className="text-sm text-content-muted">
        No video file found in this folder yet — press ↻ Rescan folder after adding some.
      </p>
    )
  }
  return (
    <ul className="grid gap-2 grid-cols-1 sm:grid-cols-2">
      {sources.map((s) => {
        const state = sourceState(s)
        const active = activeSourceId === s.id
        return (
          <li key={s.id}
            className={`flex min-w-0 flex-col gap-1 rounded-lg border p-2.5 ${
              active ? 'border-primary/60 bg-primary/10' : 'border-border bg-surface'}`}>
            <div className="flex min-w-0 items-center gap-2">
              {/* min-w-0 + truncate: the relpath is the one string here that has
                  no break opportunity, and it is what stretched the grid past
                  the viewport on a phone in the image lane. */}
              <button type="button" onClick={() => onFilter(active ? null : s.id)}
                aria-pressed={active}
                title={active ? 'Show every shot again' : `Show only the shots from ${s.relpath}`}
                className="min-w-0 flex-1 truncate text-left font-mono text-xs text-content hover:underline">
                {s.relpath}
              </button>
              <span className={`shrink-0 rounded px-1.5 py-0.5 text-[0.625rem] font-semibold ${TONE[state.tone]}`}
                title={state.title}>
                {state.label}
              </span>
            </div>
            <p className="text-[0.6875rem] text-content-subtle">
              {formatDuration(s.duration_s)} · {formatFileSize(s.file_size)}
              {sourceGeometry(s) ? ` · ${sourceGeometry(s)}` : ''}
            </p>
            {/* The squeeze, on its own line: the line above already carries
                three facts and a fourth wraps at 400 px. Absent entirely when
                the container said nothing — a blank beats a fabricated 0. */}
            {sourceEncoding(s) && (
              <p className="text-[0.6875rem] text-content-subtle"
                title="How hard this file was compressed. Bits per pixel per frame is the comparable one: under ~0.05 is visibly damaged, over ~0.15 is comfortable. Shown only — the 🩻 Defects pass measures the damage this predicts.">
                {sourceEncoding(s)}
              </p>
            )}
            <div className="flex flex-wrap gap-1">
              {onCut && firstShotBounds(s) && (
                <button type="button" onClick={() => onCut(s, firstShotBounds(s))}
                  title="Add a 5 s shot at the start of this file, then trim or split it in the player"
                  className={ACTION}>
                  ✂ Cut a shot by hand
                </button>
              )}
              {onSingleShot && s.probe_state === 'ok'
                && s.detect_state !== 'single' && (
                <button type="button" onClick={() => onSingleShot(s)}
                  title="This file is one continuous take: replace its shots with a single full-length one"
                  className={ACTION}>
                  ▣ Single shot
                </button>
              )}
              {onRecut && canRecut(s) && (
                <button type="button" onClick={() => onRecut(s)}
                  title={s.detect_state === 'single'
                    ? 'Find the shots in this file again — undoes “single shot” for it'
                    : 'Cut this file again at its own threshold, from what the detector already measured'}
                  className={ACTION}>
                  ↻ Re-detect this file
                </button>
              )}
            </div>
          </li>
        )
      })}
    </ul>
  )
}

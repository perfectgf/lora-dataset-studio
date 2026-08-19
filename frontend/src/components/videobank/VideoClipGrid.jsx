import { useEffect, useRef } from 'react'
import { videoClipThumbUrl } from './videoBankApi'
import { clipLabel } from './videoClipFragment'
import { FLAG_LABELS } from './videoMetricsFilter'
import { cameraBadge } from './videoCameraMotion'
import { transitionChip } from './videoShotCuts'

/** 🎬 The shot gallery — JPEG thumbnails, and NOT ONE <video>.
 *
 * This is the load-bearing constraint of the whole lane, so it is stated where
 * someone would be tempted to break it: the obvious "nicer" version of this grid
 * mounts a muted <video preload="none"> per tile and plays it on hover.
 *
 * It cannot work here. There is no clip FILE to hover — a bank stores bounds and
 * encodes nothing until promotion, so each tile would have to seek into a
 * multi-gigabyte rush. And Chrome caps WebMediaPlayers at about sixty across the
 * whole browser: past that, new <video> elements never load and never error. A
 * bank holds hundreds of shots, so the failure would appear on the second screen
 * of scroll, on someone else's machine, with no message.
 *
 * `loading="lazy"` on every tile: an off-screen thumbnail costs nothing.
 *
 * ⌨ In burst mode one tile carries the CURSOR. It is marked three ways on
 * purpose — a ring, a ▸ badge and `aria-current` — because a ring alone is a
 * colour, and the whole run depends on knowing which shot the next keystroke
 * will hit. The tile is also scrolled back into view whenever the cursor moves,
 * since auto-advance can walk it off the bottom of the page in three keys.
 */
export default function VideoClipGrid({
  bankId, clips, selected, onToggle, onOpen, emptyMessage, matchLines, cursorId,
}) {
  const cursorRef = useRef(null)
  // `block: 'nearest'` and nothing more: it must bring the tile back when it has
  // left the viewport and stay perfectly still when it has not — a centred
  // scroll on every keystroke makes the whole grid twitch under the hand.
  useEffect(() => {
    cursorRef.current?.scrollIntoView({ block: 'nearest' })
  }, [cursorId])

  if (!clips.length) {
    return (
      <p className="rounded-xl border border-dashed border-border bg-app/30 px-4 py-8 text-center text-sm text-content-muted">
        {emptyMessage}
      </p>
    )
  }
  const chosen = new Set(selected)
  return (
    /* grid-cols-2 at the narrow end: at 400 px a single column makes each tile
       taller than the viewport, and three makes the timecode unreadable. */
    <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
      {clips.map((clip) => {
        const isChosen = chosen.has(clip.id)
        // Only dissolves get one. Hard cuts are the vast majority, and a chip on
        // every tile is a chip nobody reads — the same rule the flags follow.
        const fade = transitionChip(clip)
        const isCursor = cursorId != null && clip.id === cursorId
        return (
          <li key={clip.id}
            ref={isCursor ? cursorRef : null}
            aria-current={isCursor ? 'true' : undefined}
            className={`relative flex min-w-0 flex-col overflow-hidden rounded-lg border bg-surface transition-colors ${
              isCursor
                ? 'border-amber-400 ring-2 ring-amber-400'
                : (isChosen ? 'border-primary ring-1 ring-inset ring-primary/60' : 'border-border')}`}>
            <button type="button" onClick={(e) => onOpen(clip, e)}
              aria-label={`Play the shot at ${clipLabel(clip.start_s, clip.end_s)} of ${clip.relpath}`}
              className="relative block aspect-video w-full bg-surface-raised">
              {clip.thumb_state === 'ok' ? (
                <img src={videoClipThumbUrl(bankId, clip.id)} alt="" loading="lazy"
                  onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
                  className="h-full w-full object-cover" />
              ) : (
                /* A 404 on the thumb route is an ordinary state (the pass has
                   not run), not an error — so it draws a placeholder rather
                   than filling the console. */
                <span className="grid h-full w-full place-items-center text-2xl text-content-subtle"
                  aria-hidden>🎞</span>
              )}
              <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-black/70 px-1 font-mono text-[0.625rem] text-white">
                {clip.duration_s?.toFixed(1)}s
              </span>
              {clip.status !== 'pending' && (
                <span aria-hidden
                  className={`pointer-events-none absolute left-1 top-1 rounded px-1 text-[0.625rem] font-bold text-white ${
                    clip.status === 'keep' ? 'bg-emerald-600' : 'bg-rose-600'}`}>
                  {clip.status === 'keep' ? '✓' : '✕'}
                </span>
              )}
              {clip.promoted_dataset_id && (
                <span aria-hidden title="Already promoted into a dataset"
                  className="pointer-events-none absolute right-1 top-1 rounded bg-indigo-600 px-1 text-[0.625rem] font-bold text-white">
                  ▶
                </span>
              )}
              {/* Quality flags, derived at read time from the raw scores and the
                  cuts in force. Amber, not red: a flag is a reason to LOOK, and
                  the verdict stays the user's. */}
              {(clip.flags || []).length > 0 && (
                <span
                  title={clip.flags.map((f) => FLAG_LABELS[f] || f).join(' · ')}
                  className="pointer-events-none absolute bottom-1 left-1 rounded bg-amber-500/90 px-1 text-[0.625rem] font-bold text-black">
                  ⚑ {clip.flags.length > 1 ? clip.flags.length : (FLAG_LABELS[clip.flags[0]] || clip.flags[0])}
                </span>
              )}
              {/* 🎥 What the camera did. SLATE, not amber, and that is the whole
                  point of it being a separate badge: amber in this grid means
                  "a cut flagged this", and a pan is not a fault. It sits bottom-
                  RIGHT so it never collides with the flag badge on the left,
                  and it shows the first label with a count when there are more,
                  the same shape the flag badge uses for the same reason. */}
              {(clip.camera || []).length > 0 && (
                <span
                  title={cameraBadge(clip)}
                  className="pointer-events-none absolute bottom-1 right-1 rounded bg-slate-800/85 px-1 text-[0.625rem] font-semibold text-slate-100">
                  🎥 {clip.camera.length > 1 ? clip.camera.length : cameraBadge(clip)}
                </span>
              )}
              {/* The kind of boundary this shot sits between, read from the
                  detector's second head. Amber like every other advisory mark,
                  and top-centre so it never lands on the flags or the duration.
                  It means "the first or last frames here are a cross-fade of
                  the neighbouring shot" — worth knowing before training on it. */}
              {fade && (
                <span title={fade.title}
                  className="pointer-events-none absolute left-1/2 top-1 -translate-x-1/2 rounded bg-amber-500/90 px-1 text-[0.625rem] font-bold text-black">
                  {fade.label}
                </span>
              )}
            </button>
            <div className="flex min-w-0 items-center gap-1.5 px-1.5 py-1">
              {/* The cursor marker lives in the FOOTER, not over the thumbnail:
                  every corner up there is already taken (status, promoted,
                  flags, camera, dissolve, duration) and a badge that lands on
                  another badge is worse than no badge. It is here because the
                  ring is a colour, and the whole run depends on knowing which
                  shot the next keystroke will hit. */}
              {isCursor && (
                <span title="The next keystroke decides this shot"
                  className="shrink-0 rounded bg-amber-400 px-1 text-[0.625rem] font-bold leading-tight text-black">
                  ▸ next
                </span>
              )}
              <input type="checkbox" checked={isChosen}
                onChange={(e) => onToggle(clip.id, e)}
                aria-label={`Select the shot at ${clipLabel(clip.start_s, clip.end_s)}`}
                className="shrink-0 accent-indigo-500" />
              <span className="min-w-0 truncate font-mono text-[0.625rem] text-content-subtle"
                title={`${clip.relpath} — ${clipLabel(clip.start_s, clip.end_s)}`}>
                {clipLabel(clip.start_s, clip.end_s)}
              </span>
            </div>
            {/* Which SECOND of the shot matched a search, when there was one.
                A shot is a span and the phrase usually describes a moment of it;
                showing the tile without the moment sells the whole span as the
                answer. Absent outside a search — this is not a permanent
                property of a shot. */}
            {matchLines?.[clip.id] && (
              <p className="truncate px-1.5 pb-1 text-[0.625rem] text-indigo-300"
                title={matchLines[clip.id]}>
                🔎 {matchLines[clip.id]}
              </p>
            )}
          </li>
        )
      })}
    </ul>
  )
}

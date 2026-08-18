import { videoClipThumbUrl } from './videoBankApi'
import { clipLabel } from './videoClipFragment'
import { FLAG_LABELS } from './videoMetricsFilter'
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
 */
export default function VideoClipGrid({
  bankId, clips, selected, onToggle, onOpen, emptyMessage, matchLines,
}) {
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
        return (
          <li key={clip.id}
            className={`relative flex min-w-0 flex-col overflow-hidden rounded-lg border bg-surface transition-colors ${
              isChosen ? 'border-primary ring-1 ring-inset ring-primary/60' : 'border-border'}`}>
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

import { clipLabel } from './videoClipFragment'
import { clipDurationS, datasetClipPoster, hasCaption, isStillFile } from './videoDatasetClips'

/** 🎞 The clips of a built dataset — and, exactly like the bank's gallery, NOT
 * ONE <video>.
 *
 * The constraint is the same and it is worth restating where someone would break
 * it, because the reasoning that produced it on the bank does NOT transfer
 * unchanged: there, the honest excuse was that no clip file exists before
 * promotion. Here the files DO exist, each a few megabytes, and mounting one per
 * tile looks perfectly reasonable. It is not. Chrome caps WebMediaPlayers at
 * about sixty across the whole browser; past that, new <video> elements never
 * load and never error. A promoted set of 128 clips would fail on the second
 * screen of scroll, on someone else's machine, with nothing in the console.
 *
 * So a tile is a still image, and the poster is resolved in this order
 * (datasetClipPoster — the one rule the Video Test Studio's start-frame
 * picker follows too, so both pages agree on every clip):
 *
 *   · a STILLS set serves images through the media route itself — that is the
 *     real frame, at no extra cost;
 *   · a clip cut from a bank borrows the bank's JPEG thumbnail through its
 *     provenance columns. Free, already generated, and it shows the shot;
 *   · anything else draws a placeholder. A missing thumbnail is an ordinary
 *     state (the bank was deleted, the thumbs pass never ran), not an error, and
 *     it must not fill the console with 404s.
 */
export default function VideoDatasetGrid({
  datasetId, clips, selected, onToggle, onOpen, emptyMessage,
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
    /* grid-cols-2 at the narrow end for the same measured reason as the bank's
       grid: at 400 px a single column makes a tile taller than the viewport, and
       three columns make the file name unreadable. */
    <ul className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
      {clips.map((clip) => {
        const isChosen = chosen.has(clip.id)
        const still = isStillFile(clip.filename)
        const seconds = clipDurationS(clip)
        const poster = datasetClipPoster(datasetId, clip)
        return (
          <li key={clip.id}
            className={`relative flex min-w-0 flex-col overflow-hidden rounded-lg border bg-surface transition-colors ${
              isChosen ? 'border-primary ring-1 ring-inset ring-primary/60' : 'border-border'}`}>
            <button type="button" onClick={(e) => onOpen(clip, e)}
              aria-label={`${still ? 'View' : 'Play'} ${clip.filename}${
                seconds ? ` — ${clipLabel(clip.start_s, clip.end_s)}` : ''}`}
              className="relative block aspect-video w-full bg-surface-raised">
              {poster ? (
                <img src={poster} alt="" loading="lazy"
                  onError={(e) => { e.currentTarget.style.visibility = 'hidden' }}
                  className="h-full w-full object-cover" />
              ) : (
                <span className="grid h-full w-full place-items-center text-2xl text-content-subtle"
                  aria-hidden>🎞</span>
              )}
              {seconds != null && (
                <span className="pointer-events-none absolute bottom-1 right-1 rounded bg-black/70 px-1 font-mono text-[0.625rem] text-white">
                  {seconds.toFixed(1)}s
                </span>
              )}
              {/* The badge marks what is MISSING, not what is present: a set is
                  worked through by hunting the silent clips, and a ✓ on every
                  captioned tile would drown the three that still need one. */}
              {!hasCaption(clip) && (
                <span className="pointer-events-none absolute left-1 top-1 rounded bg-amber-600/90 px-1 text-[0.625rem] font-bold text-white"
                  title="No caption — this clip trains on the trigger word alone">
                  no caption
                </span>
              )}
            </button>
            <div className="flex min-w-0 items-center gap-1.5 px-1.5 py-1">
              <input type="checkbox" checked={isChosen}
                onChange={(e) => onToggle(clip, e.nativeEvent)}
                aria-label={`Select ${clip.filename}`}
                className="h-4 w-4 shrink-0 accent-indigo-500" />
              <span className="min-w-0 truncate font-mono text-[0.625rem] text-content-subtle"
                title={clip.src_relpath
                  ? `${clip.filename} — cut from ${clip.src_relpath}`
                  : clip.filename}>
                {clip.filename}
              </span>
            </div>
          </li>
        )
      })}
    </ul>
  )
}

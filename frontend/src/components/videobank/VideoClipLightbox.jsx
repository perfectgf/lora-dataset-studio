import { useEffect, useRef, useState } from 'react'
import {
  clipFragmentSrc, clipLabel, shouldRemountPlayer, playerBudgetWarning,
  MAX_MOUNTED_PLAYERS,
} from './videoClipFragment'
import { playFromSecond, captionStateNote } from './videoClipSearch'
import { videoClipCaptionUrl } from './videoBankApi'
import { patchJson } from '../../api/fetchClient'
import { videoSourceMediaUrl } from './videoBankApi'
import VideoClipTrimTools from './VideoClipTrimTools'

/** 🎬 Watching ONE shot — the only <video> element this lane ever mounts.
 *
 * The grid holds JPEGs and nothing else (see videoClipFragment.js for the two
 * reasons: no clip file exists before promotion, and Chrome silently stops
 * loading new players past roughly forty). This component is where that budget
 * is spent, once.
 *
 * The element is KEYED on the clip so React recreates it when you move to the
 * next shot: assigning a new `#t=` to a live <video> is ignored by several
 * browsers once the resource is loaded, and the viewer then watches the previous
 * shot's range while the caption names a different one. `shouldRemountPlayer`
 * owns that decision, and the key is derived from it rather than from the clip
 * id, so "same source, same start" never restarts playback from the head.
 *
 * That rule pays for itself once the ✂ retouch tools land here: saving a new
 * START swaps the clip row in place, the descriptor changes, and the player is
 * recreated on the new `#t=` — so you immediately watch the range you just cut
 * instead of the one you cut it from. Saving only the END deliberately does not
 * remount: the range you are watching still opens at the same instant, and
 * restarting playback there would be a punishment for trimming a tail.
 */
export default function VideoClipLightbox({
  bankId, clip, source, onClose, onPrev, onNext, onKeep, onReject, onRetouched,
  hasPrev, hasNext, playFrom,
}) {
  const [failed, setFailed] = useState(false)
  // The playhead, in SOURCE seconds — the element's timeline IS the rush's,
  // because the resource it loads is the whole file and the media fragment only
  // moves the initial seek and the stop point. That single fact is what lets the
  // retouch tools use `currentTime` as a bound with no conversion, and it is
  // asserted rather than assumed in videoClipEdit.playheadToSourceTime.
  const [playheadS, setPlayheadS] = useState(null)
  // 🗣 The caption, held locally while it is being typed. A generated caption is
  // a DRAFT: saving marks it 'edited' server-side so a bulk re-run leaves it
  // alone, which is what makes correcting one worth the effort.
  const [caption, setCaption] = useState('')
  const [savingCaption, setSavingCaption] = useState(false)
  // What the mounted element was built for, plus the key that forces React to
  // recreate it. Adjusted DURING render on purpose (React's documented
  // derive-during-render pattern) — an effect would run after paint and leave
  // one frame showing the previous shot's range under the new caption.
  //
  // Idempotent, which is what makes it safe: shouldRemountPlayer compares the
  // FIELDS, so a second render with an equal descriptor answers false and the
  // key stops moving. StrictMode's double render therefore bumps it once.
  const player = useRef({ desc: null, key: 0 })
  // `playFrom` is in the descriptor because it changes the media fragment, and a
  // fragment reassigned to a LIVE element is ignored by several browsers — the
  // viewer would silently watch the shot from its start while the caption said
  // it opened on the matched second.
  const openAt = playFromSecond(clip, playFrom)
  const next = clip ? { sourceId: clip.source_id, start: openAt } : null
  if (shouldRemountPlayer(player.current.desc, next)) {
    player.current = { desc: next, key: player.current.key + 1 }
  }
  const playerKey = player.current.key

  useEffect(() => { setCaption(clip?.caption || '') }, [clip?.id, clip?.caption])
  useEffect(() => { setFailed(false); setPlayheadS(clip ? openAt : null) },
    [playerKey])                                        // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.()
      else if (e.key === 'ArrowLeft') onPrev?.()
      else if (e.key === 'ArrowRight') onNext?.()
      else if (e.key === 'k' || e.key === 'K') onKeep?.()
      else if (e.key === 'r' || e.key === 'R') onReject?.()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onPrev, onNext, onKeep, onReject])

  if (!clip) return null
  // Opens ON the matched second when a search sent us here, on the shot's own
  // start otherwise. The END is always the shot's — a search result is a place
  // to look, never a re-cut of the shot.
  const src = clipFragmentSrc(videoSourceMediaUrl(bankId, clip.source_id),
    openAt, clip.end_s)
  // Defensive, and cheap: a malformed fragment does not throw — the browser
  // ignores it and plays the WHOLE file, which on a two-hour rush is the worst
  // available outcome. clipFragmentSrc answers null rather than hand that over.
  const budget = playerBudgetWarning(MAX_MOUNTED_PLAYERS)

  return (
    <div role="dialog" aria-modal="true" aria-label={`Shot ${clipLabel(clip.start_s, clip.end_s)}`}
      className="fixed inset-0 z-50 flex flex-col bg-black/90 p-2 sm:p-4">
      <div className="mx-auto flex w-full max-w-4xl min-w-0 flex-1 flex-col gap-2 overflow-y-auto">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <p className="min-w-0 truncate font-mono text-xs text-white/80" title={clip.relpath}>
            {clip.relpath}
          </p>
          <button type="button" onClick={onClose} aria-label="Close the player"
            className="ml-auto rounded-md border border-white/20 px-2 py-1 text-sm text-white hover:bg-white/10">
            ✕
          </button>
        </div>

        <div className="relative flex min-h-0 flex-1 items-center justify-center rounded-lg bg-black">
          {src && !failed ? (
            <video
              key={playerKey}
              src={src}
              controls
              autoPlay
              // No `loop`: the media fragment's end bound stops playback where
              // the shot ends, and a loop would replay the range forever behind
              // whatever you clicked next.
              preload="metadata"
              onError={() => setFailed(true)}
              // `timeupdate` fires about four times a second — enough to keep
              // "✂ Split here" honest, cheap enough not to be worth throttling.
              // `seeked` is what makes a paused scrub usable: dragging to a frame
              // and splitting there is the whole gesture, and it emits no
              // timeupdate while paused.
              onTimeUpdate={(e) => setPlayheadS(e.currentTarget.currentTime)}
              onSeeked={(e) => setPlayheadS(e.currentTarget.currentTime)}
              className="max-h-[60vh] w-full object-contain"
            >
              <track kind="captions" />
            </video>
          ) : (
            <p className="p-6 text-center text-sm text-white/70">
              {src
                ? /* An honest failure: the file is served fine, this BROWSER cannot
                     decode that codec (ProRes in a .mov, HEVC in a .mkv…). The
                     bank does not care — detection reads the file server-side and
                     promotion re-encodes it. Only this preview is unavailable. */
                  'Your browser can’t play this file’s format. The shot is still fine — '
                  + 'detection read it, its thumbnail is real, and promotion re-encodes it.'
                : 'This shot has no playable range — its start and end are the same.'}
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="font-mono text-white/90">{clipLabel(clip.start_s, clip.end_s)}</span>
          {clip.status !== 'pending' && (
            <span className={clip.status === 'keep' ? 'text-emerald-300' : 'text-rose-300'}>
              {clip.status === 'keep' ? '✓ Kept' : '✕ Rejected'}
            </span>
          )}
          <div className="ml-auto flex flex-wrap items-center gap-1.5">
            <button type="button" onClick={onPrev} disabled={!hasPrev}
              className="rounded-md border border-white/20 px-2 py-1 text-white hover:bg-white/10 disabled:opacity-30">
              ← Prev
            </button>
            <button type="button" onClick={onNext} disabled={!hasNext}
              className="rounded-md border border-white/20 px-2 py-1 text-white hover:bg-white/10 disabled:opacity-30">
              Next →
            </button>
            <button type="button" onClick={onKeep}
              className="rounded-md bg-emerald-600 px-3 py-1 font-semibold text-white hover:bg-emerald-500">
              ✓ Keep
            </button>
            <button type="button" onClick={onReject}
              className="rounded-md bg-rose-600 px-3 py-1 font-semibold text-white hover:bg-rose-500">
              ✕ Reject
            </button>
          </div>
        </div>
        {/* 🗣 The caption. Above the retouch tools because it is read far more
            often than a bound is moved: it is what this clip will TRAIN on and
            what a word search matches. */}
        <div className="rounded-lg border border-white/15 bg-black/40 p-2">
          <label htmlFor="video-clip-caption"
            className="block text-xs font-semibold text-white/80">
            Caption <span className="font-normal text-white/50">— the training prompt</span>
          </label>
          <textarea id="video-clip-caption" rows={2} value={caption}
            onChange={(e) => setCaption(e.target.value)}
            placeholder="What happens in this shot"
            className="mt-1 w-full resize-y rounded-md border border-white/20 bg-black/50 px-2 py-1 text-sm text-white placeholder:text-white/30" />
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <button type="button" disabled={savingCaption || caption === (clip.caption || '')}
              onClick={async () => {
                setSavingCaption(true)
                try {
                  const d = await patchJson(videoClipCaptionUrl(bankId, clip.id),
                    { caption })
                  onRetouched?.(d, 'caption')
                } catch {
                  /* the field keeps what was typed; nothing is lost */
                } finally {
                  setSavingCaption(false)
                }
              }}
              className="rounded-md bg-gradient-primary px-2.5 py-1 text-xs font-semibold text-gray-950 disabled:opacity-40">
              {savingCaption ? 'Saving…' : 'Save caption'}
            </button>
            <span className="text-[0.6875rem] text-white/50">
              {captionStateNote({ ...clip, caption })}
            </span>
          </div>
        </div>

        {/* The retouch tools. Folded by default: triage is a K/R/→ rhythm over
            hundreds of shots, and the mis-cut ones are the minority. */}
        <VideoClipTrimTools bankId={bankId} clip={clip} source={source}
          playheadS={playheadS} onChanged={onRetouched} />

        <p className="text-center text-[0.6875rem] text-white/50">
          ← → to move · K to keep · R to reject · Esc to close
          {/* Never rendered on a correct build; it is a tripwire for a future
              grid that reintroduces inline players. */}
          {budget ? ` · ${budget}` : ''}
        </p>
      </div>
    </div>
  )
}

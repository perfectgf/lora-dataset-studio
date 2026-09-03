import { useEffect, useRef, useState } from 'react'
import { videoDatasetClipComparisonUrl, videoDatasetClipMediaUrl } from './videoBankApi'
import SideBySideVideo from './SideBySideVideo'
import { neuralRenderTags } from './neuralRenderParams'
import { clipDurationS, isStillFile, lightboxKeyAction } from './videoDatasetClips'
import { clipLabel } from './videoClipFragment'

/** ▶ Watching ONE clip of a dataset — the only <video> this page ever mounts.
 *
 * The grid holds still images and nothing else (VideoDatasetGrid says why in
 * full: Chrome stops loading players past roughly sixty, silently). This is
 * where that budget is spent, once.
 *
 * KEYED ON THE CLIP, deliberately. Assigning a new `src` to a live <video> is
 * honoured by browsers, but the element keeps the previous resource's buffered
 * state and its playback position, so stepping to the next clip could show the
 * end of the previous one for a beat under the new file name. A fresh element
 * per clip costs nothing here — there is exactly one.
 *
 * THE CAPTION BOX IS THE POINT of opening a clip at all. It writes the .txt
 * sidecar next to the .mp4 through the server, and a sidecar that could not be
 * written is said OUT LOUD rather than swallowed: the trainer reads the file,
 * so a row saved without its file trains the previous text while this box shows
 * the new one.
 */
export default function VideoDatasetLightbox({
  datasetId, clip, caption, onCaptionChange, onSave, onClose, onPrev, onNext,
  onRemove, hasPrev, hasNext, saving,
  // ⇔ The kept original of a neural-rendered clip, or null: with it the
  // lightbox offers the side-by-side comparison, without it the button does
  // not exist — there is nothing to compare a clip that plays its original to.
  compareSrc = null,
  // ✨ The dials that made this clip's render, or null (the workspace reads
  // them from the dataset's neural-render state).
  nrParams = null,
}) {
  const [failed, setFailed] = useState(false)
  // ⇔ Whether the side-by-side comparison is open. Reset with the clip: the
  // next clip may play no render at all.
  const [comparing, setComparing] = useState(false)
  const typing = useRef(false)
  useEffect(() => { setFailed(false); setComparing(false) }, [clip?.id])

  // ⌨ Esc closes, ← → step. Guarded on `typing`, and that guard is not a nicety:
  // without it every arrow key pressed while writing a caption would jump to
  // another clip and abandon the text mid-word.
  useEffect(() => {
    // The DECISION lives in lightboxKeyAction, as a tested value; this only
    // dispatches it. Escape saves before it closes (a focused element removed
    // from the DOM never fires blur, and blur owns the save — so "Escape while
    // typing" used to drop the caption in silence), and the arrows are the
    // caret's while the user types. A source regex used to guard this and a
    // one-line early return walked straight through it with the suite green.
    const onKey = (e) => {
      const action = lightboxKeyAction(e.key, { typing: typing.current, hasPrev, hasNext })
      if (action === 'save-close') { onSave(); onClose() }
      else if (action === 'prev') onPrev()
      else if (action === 'next') onNext()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, onSave, onPrev, onNext, hasPrev, hasNext])

  if (!clip) return null
  const still = isStillFile(clip.filename)
  const src = videoDatasetClipMediaUrl(datasetId, clip.id)
  const seconds = clipDurationS(clip)

  return (
    <div role="dialog" aria-modal="true" aria-label={`Clip ${clip.filename}`}
      data-probe-chrome="lightbox" data-probe-layer
      className="fixed inset-0 z-50 flex flex-col bg-black/90 p-2 sm:p-4">
      <div className="mx-auto flex w-full max-w-4xl min-w-0 flex-1 flex-col gap-2 overflow-y-auto">
        {/* Sticky for the reason the bank's lightbox learned on a phone: this
            column scrolls, and the ✕ is the only way out where Esc does not
            exist. */}
        <div className="sticky top-0 z-10 flex min-w-0 flex-wrap items-center gap-2 bg-black/85">
          <p className="min-w-0 truncate font-mono text-xs text-white/80" title={clip.filename}>
            {clip.filename}
          </p>
          <button type="button" onClick={onClose} aria-label="Close the player"
            className="ml-auto min-h-10 rounded-md border border-white/20 px-3 py-1 text-sm text-white hover:bg-white/10 lg:min-h-0">
            ✕
          </button>
        </div>

        <div className="relative flex min-h-0 flex-1 items-center justify-center rounded-lg bg-black">
          {failed ? (
            <p className="p-6 text-center text-sm text-white/70">
              {/* Honest about WHOSE problem it is. The file is served fine; this
                  browser cannot decode what the encode produced. That is worth
                  knowing before a training run, so it does not read as "the
                  clip is broken". */}
              Your browser can’t play this file. The clip itself is what the trainer
              will read — check the encode target if this keeps happening.
            </p>
          ) : still ? (
            /* A stills set holds IMAGES. Wrapping one in a <video> renders a dead
               player — found on a phone the day stills shipped. */
            <img key={clip.id} src={src} alt={clip.caption || clip.filename}
              onError={() => setFailed(true)}
              className="max-h-[60vh] w-full object-contain" />
          ) : comparing ? (
            /* The single player yields to the pair while the comparison is
               open: two players of the same clip fighting for sound is not a
               comparison. Closing brings this one back, at the clip's start. */
            <p className="p-6 text-center text-xs text-white/60">Comparing with the original…</p>
          ) : (
            <video key={clip.id} src={src} controls autoPlay preload="metadata"
              onError={() => setFailed(true)}
              className="max-h-[60vh] w-full object-contain">
              <track kind="captions" />
            </video>
          )}
        </div>

        {nrParams && (
          <p className="text-[0.6875rem] text-white/70">
            ✨ Neural render: {neuralRenderTags(nrParams).join(' · ')}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {/* Provenance, on the screen where it is asked for: "which rush is
              this, and where in it?" is the question a badly cut clip raises,
              and the answer is what you take back to the bank to re-cut. */}
          <span className="min-w-0 truncate font-mono text-[0.6875rem] text-white/70"
            title={clip.src_relpath || ''}>
            {clip.src_relpath || 'no source recorded'}
            {seconds != null ? ` · ${clipLabel(clip.start_s, clip.end_s)}` : ''}
          </span>
          <div className="ml-auto flex flex-wrap items-center gap-1.5">
            <button type="button" onClick={onPrev} disabled={!hasPrev}
              className="min-h-10 rounded-md border border-white/20 px-2 py-1 text-white hover:bg-white/10 disabled:opacity-30 lg:min-h-0">
              ← Prev
            </button>
            <button type="button" onClick={onNext} disabled={!hasNext}
              className="min-h-10 rounded-md border border-white/20 px-2 py-1 text-white hover:bg-white/10 disabled:opacity-30 lg:min-h-0">
              Next →
            </button>
            {compareSrc && !still && (
              <button type="button" onClick={() => setComparing(true)}
                title="Play the kept original next to this neural render, in step"
                className="min-h-10 rounded-md border border-white/20 px-2 py-1 text-white hover:bg-white/10 lg:min-h-0">
                ⇔ Compare with original
              </button>
            )}
            <button type="button" onClick={() => onRemove(clip)}
              title="Remove this clip from the dataset — the bank keeps the shot"
              className="min-h-10 rounded-md border border-white/20 px-2 py-1 text-white hover:border-rose-400/60 hover:text-rose-300 lg:min-h-0">
              🗑 Remove
            </button>
          </div>
        </div>

        <label className="flex flex-col gap-1">
          <span className="text-[0.6875rem] text-white/70">
            Caption — written to <code className="font-mono">{clip.filename.replace(/\.[^.]+$/, '.txt')}</code> next to the clip
          </span>
          <textarea rows={3} value={caption}
            onChange={(e) => onCaptionChange(e.target.value)}
            onFocus={() => { typing.current = true }}
            onBlur={() => { typing.current = false; onSave() }}
            placeholder="Describe what happens in the clip — camera, subject, motion."
            className="w-full rounded border border-white/20 bg-black/60 px-2 py-1 text-sm text-white" />
          <span className="text-[0.625rem] text-white/50">
            {saving ? 'Saving…' : 'Saved when you click away. Esc closes; ← → step through the set.'}
          </span>
        </label>
      </div>
      {comparing && compareSrc && (
        <SideBySideVideo originalSrc={compareSrc} renderSrc={src} title={clip.filename}
          exportHref={videoDatasetClipComparisonUrl(datasetId, clip.id)}
          onClose={() => setComparing(false)} />
      )}
    </div>
  )
}

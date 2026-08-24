/** ✂ Retouching one shot's bounds — pure helpers (no JSX, so `node --test` runs them).
 *
 * The detector is good and it is not right. It misses a boundary on a slow
 * dissolve, and it hands back shots whose last second is a frozen frame. Until
 * these helpers existed the only gesture available on either was ✕ Reject, which
 * throws away eight good seconds to be rid of one bad one.
 *
 * THE PLAYHEAD IS IN SOURCE TIME, AND THAT IS THE ONE FACT EVERYTHING HERE RESTS
 * ON. The lightbox mounts ONE <video> pointed at the SOURCE file with a media
 * fragment (`…/source/12/media#t=41.25,50`). The resource is the whole rush, so
 * the element's timeline is the rush's timeline: `duration` is the file's, and
 * `currentTime` reads 43.9 while playing the shot that starts at 41.25. The
 * fragment moves the initial seek and the stop point, not the origin. So no
 * conversion is needed — but "no conversion needed" is exactly the kind of claim
 * that rots the day someone points the player at a per-clip file, and a split at
 * the wrong instant is silent: you get two shots, both plausible, neither where
 * you asked. Hence `playheadToSourceTime`, which asserts the invariant instead of
 * assuming it, and answers null rather than a guess when the reading cannot be in
 * this shot.
 *
 * WHY NO DRAGGABLE TIMELINE. Nudge buttons and "set to playhead" cover the two
 * things that actually happen during triage — shave a frozen tail, move a start
 * onto the first clean frame — with a gesture that works at 400 px and on a
 * trackpad. A scrubbable filmstrip is a large piece of UI whose extra precision
 * lands below one frame.
 */

/** The shortest span the server will accept (video_bank_service.MIN_CLIP_S).
 * Duplicated rather than fetched so a button can be disabled BEFORE a round trip;
 * the server stays the authority and refuses with its own sentence. */
export const MIN_CLIP_S = 0.5

/** The rate to assume when a source has not been probed, or reports nonsense.
 * 30 is a floor for "one frame" in a nudge — being one frame conservative on a
 * 60 fps file costs 16 ms, guessing 120 on a 24 fps one moves five frames. */
const FALLBACK_FPS = 30

/** One frame, in seconds, at the SOURCE's own rate — never the target's.
 *
 * The distinction is the same one that makes a 16 fps target accelerate motion if
 * you read the wrong column: bounds are timestamps in the source, so a "one frame"
 * nudge has to mean one frame OF THE SOURCE. */
export function frameStep(fpsNative) {
  const fps = Number(fpsNative)
  if (!Number.isFinite(fps) || fps <= 0 || fps > 1000) return 1 / FALLBACK_FPS
  return 1 / fps
}

/** The player's `currentTime` as a source timestamp, or null when it cannot be
 * one for this shot.
 *
 * Returns null — never a clamped value — when the reading falls outside the shot.
 * Clamping would hand `splitAt` the boundary itself, which is the one place a
 * split makes an empty clip; and it would hide the only symptom of the invariant
 * above having broken. A null disables the button, which is a question the user
 * can answer by moving the playhead. */
export function playheadToSourceTime(currentTime, clip) {
  const t = Number(currentTime)
  if (!clip || !Number.isFinite(t)) return null
  const start = Number(clip.start_s)
  const end = Number(clip.end_s)
  if (!Number.isFinite(start) || !Number.isFinite(end)) return null
  // A hair of tolerance: a browser that has just seeked to the fragment start
  // routinely reports 41.249999.
  if (t < start - 1e-3 || t > end + 1e-3) return null
  return round3(Math.min(Math.max(t, start), end))
}

/** Can this playhead position split this shot? `{ at }` or `{ why }`.
 *
 * Both halves have to be a shot in their own right, which is what makes a split
 * flush against a bound and a split 0.2 s in the same refusal — and it is the same
 * rule the server applies, so the button and the 400 cannot disagree. */
export function splitAvailability(clip, currentTime) {
  const at = playheadToSourceTime(currentTime, clip)
  if (at == null) {
    return { why: 'Move the playhead inside this shot to split it.' }
  }
  if (at - Number(clip.start_s) < MIN_CLIP_S
      || Number(clip.end_s) - at < MIN_CLIP_S) {
    return { why: `Both halves must last at least ${MIN_CLIP_S}s — move the `
      + 'playhead further from the ends.' }
  }
  return { at }
}

/** A draft of the bounds, moved. `{ start_s, end_s }` or null when the move is
 * not legal — a nudge that would invert the shot, push it before the file starts
 * or past its end is simply not applied, so a held-down button stops at the wall
 * instead of walking through it.
 *
 * `edge` is 'start' or 'end'; `deltaS` may be negative. `durationS` is the
 * SOURCE's, and may be null on a bank whose probe has not run — in which case the
 * upper wall is not enforced here and the server refuses instead. */
export function nudgedBounds(draft, edge, deltaS, durationS) {
  if (!draft) return null
  const start = Number(draft.start_s)
  const end = Number(draft.end_s)
  const delta = Number(deltaS)
  if (!Number.isFinite(start) || !Number.isFinite(end) || !Number.isFinite(delta)) {
    return null
  }
  const next = edge === 'start'
    ? { start_s: round3(start + delta), end_s: round3(end) }
    : { start_s: round3(start), end_s: round3(end + delta) }
  return isLegalSpan(next, durationS) ? next : null
}

/** Move one bound onto the playhead. Same walls as a nudge, same null. */
export function boundsAtPlayhead(draft, edge, currentTime, durationS) {
  const t = Number(currentTime)
  if (!draft || !Number.isFinite(t)) return null
  const next = edge === 'start'
    ? { start_s: round3(t), end_s: round3(Number(draft.end_s)) }
    : { start_s: round3(Number(draft.start_s)), end_s: round3(t) }
  return isLegalSpan(next, durationS) ? next : null
}

/** The rule, in one place: inside the file, and long enough to be trained on. */
export function isLegalSpan({ start_s: start, end_s: end }, durationS) {
  const a = Number(start)
  const b = Number(end)
  if (!Number.isFinite(a) || !Number.isFinite(b)) return false
  if (a < 0) return false
  if (b - a < MIN_CLIP_S - 1e-9) return false
  const duration = Number(durationS)
  if (Number.isFinite(duration) && duration > 0 && b > duration + 1e-6) return false
  return true
}

/** Has the draft actually moved? Sub-millisecond differences are float noise from
 * a `currentTime` reading, not an edit — saving on them would invalidate a
 * thumbnail and a full metrics pass for nothing. */
export function boundsChanged(clip, draft) {
  if (!clip || !draft) return false
  return Math.abs(Number(draft.start_s) - Number(clip.start_s)) > 1e-3
    || Math.abs(Number(draft.end_s) - Number(clip.end_s)) > 1e-3
}

/** "1.50s → 6.25s (4.75s)" — what is about to be saved, in the same units as the
 * nudge buttons, because "0:01 – 0:06" hides the very precision being edited. */
export function draftSummary(draft) {
  if (!draft) return ''
  const start = Number(draft.start_s)
  const end = Number(draft.end_s)
  return `${start.toFixed(2)}s → ${end.toFixed(2)}s (${(end - start).toFixed(2)}s)`
}

/** Default length of a hand-made shot, in seconds. Long enough to be a shot, short
 * enough that trimming it down is the usual next gesture rather than the reverse. */
const NEW_SHOT_S = 5

/** Bounds for a shot the detector missed entirely, starting at the playhead.
 * `{ start_s, end_s }`, or null when there is not room before the end of the file.
 *
 * Deliberately NOT built on `playheadToSourceTime`: that one refuses a reading
 * outside the open shot, which is exactly where a missed cut is. The media
 * fragment only sets the initial seek and the stop point — the element's timeline
 * is still the whole rush, so the user can scrub anywhere in the file and mark a
 * boundary the detector never drew. */
export function newShotBounds(currentTime, durationS, lengthS = NEW_SHOT_S) {
  const t = Number(currentTime)
  if (!Number.isFinite(t) || t < 0) return null
  const duration = Number(durationS)
  const hasDuration = Number.isFinite(duration) && duration > 0
  const end = hasDuration ? Math.min(t + Number(lengthS), duration) : t + Number(lengthS)
  const next = { start_s: round3(t), end_s: round3(end) }
  return isLegalSpan(next, hasDuration ? duration : null) ? next : null
}

/** The FIRST shot of a file, so hand-cutting is reachable on a bank that has no
 * shots at all. `{ start_s, end_s }`, or null when the file is too short to hold
 * one (or has not been probed and reports no duration to trust).
 *
 * The gap this closes is not hypothetical: every other retouch gesture lives
 * inside the lightbox, and the lightbox needs a shot to open. On an install
 * without the detector — which the capability strip explicitly says still lets
 * you "scan, cut, watch and triage" — there was no first shot, so the whole
 * feature was unreachable exactly where it was the only option.
 *
 * It starts at 0 rather than in the middle: a first cut is a starting point to
 * trim and split, and the beginning of the file is the one place the user can
 * predict without watching anything. */
export function firstShotBounds(source, lengthS = NEW_SHOT_S) {
  const duration = Number(source?.duration_s)
  if (!Number.isFinite(duration) || duration <= 0) return null
  const next = { start_s: 0, end_s: round3(Math.min(lengthS, duration)) }
  return isLegalSpan(next, duration) ? next : null
}

/** THE DISCOVERY THIS TOOL EXISTS TO SURFACE, in one line, where the gesture is.
 *
 * ai-toolkit conditions an image-to-video sample on the clip's FIRST frame
 * (`wan22_14b_i2v_model.py`, which slices frame 0 as the conditioning latent).
 * So on an i2v target, moving a shot's START is not trimming — it is choosing the
 * image the model is taught to animate FROM. Nothing in the app said so, and no
 * user would guess it from a control called "trim", which is how a dataset ends up
 * conditioned on three hundred dissolve frames.
 *
 * A constant rather than prose in the JSX so the wording is testable and lives
 * next to the rule it describes. */
export const I2V_FIRST_FRAME_HINT =
  'For image-to-video targets, the first frame is the conditioning image — '
  + 'moving the start picks which frame the model learns to animate from.'

/** What to tell the user right after a retouch.
 *
 * The thumbnail and the measurements of the old span are dropped server-side (a
 * thumbnail of a frame the shot no longer contains is not stale, it is wrong), so
 * the tile goes blank. Saying so turns a disappearing image into an expected
 * consequence — and the workspace's next-step line will offer the thumbnails pass
 * on its own, because `counts.thumbs` just fell. */
export function retouchToast(kind) {
  const thumbs = ' Run Make thumbnails again when you are done cutting.'
  if (kind === 'create') {
    // Nothing was dropped — this shot never had a thumbnail to lose.
    return `New shot added — it has no thumbnail yet.${thumbs}`
  }
  const dropped = ' Its thumbnail and quality scores were dropped, because they '
    + 'described the old bounds.'
  if (kind === 'split') return `Shot split in two.${dropped}${thumbs}`
  return `Bounds saved.${dropped}${thumbs}`
}

function round3(value) {
  return Math.round(Number(value) * 1000) / 1000
}

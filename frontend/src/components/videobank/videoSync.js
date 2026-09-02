/** ⇔ Keeping two <video> elements in step — the pure half.
 *
 * The side-by-side player shows the original and its neural render. Two
 * players that drift by a few frames make the comparison read as "the render
 * moves differently", so one is the LEADER (the user's controls) and the other
 * FOLLOWS: play, pause, seek and rate are mirrored, and the follower is nudged
 * back whenever it drifts past a tolerance.
 *
 * This module decides WHAT to do from two snapshots; the component applies it.
 * Pure so `node --test` can pin the rules — what a DOM test cannot do here.
 */

/** Seconds of drift tolerated before the follower is re-seeked. Two frames
 *  at 24 fps: below that a seek costs more (a decode stall) than it fixes. */
export const DRIFT_TOLERANCE_S = 0.08

/** Actions to bring `follower` in step with `leader`. Both are plain snapshots
 *  ({ currentTime, paused, playbackRate }). Order matters: rate before seek,
 *  seek before play, so a re-started follower starts at the right instant. */
export function syncActions(leader, follower, { tolerance = DRIFT_TOLERANCE_S } = {}) {
  const actions = []
  if (!leader || !follower) return actions
  if (Number.isFinite(leader.playbackRate) && leader.playbackRate !== follower.playbackRate) {
    actions.push({ type: 'rate', value: leader.playbackRate })
  }
  const drift = Math.abs((leader.currentTime || 0) - (follower.currentTime || 0))
  if (drift > tolerance) actions.push({ type: 'seek', value: leader.currentTime || 0 })
  if (leader.paused && !follower.paused) actions.push({ type: 'pause' })
  if (!leader.paused && follower.paused) actions.push({ type: 'play' })
  return actions
}

/** The label pair for the two sides, swapped when the user asked to. */
export function sidesFor(swapped) {
  const original = { key: 'original', label: 'Original' }
  const render = { key: 'render', label: 'Neural render (DLSS 5)' }
  return swapped ? [render, original] : [original, render]
}

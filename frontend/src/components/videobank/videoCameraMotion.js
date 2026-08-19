/** 🎥 The camera-motion facet: what each label means, and filtering the grid by it.
 *
 * A SEPARATE FACET FROM THE FLAG CHIPS, and the separation is the design rather
 * than tidiness. The amber ⚑ chips answer "what is wrong with this shot"; these
 * answer "what does this shot DO". Merging them would put "pans right" in a list
 * of defects, and — worse — would hide the case this pass exists to serve: the
 * person training a handheld look is FILTERING FOR the wobble, not against it.
 *
 * The vocabulary is Hunyuan's, not ours, for eight of the eleven labels — it is
 * the vocabulary of the model this app trains, so a filter here and a caption
 * later speak the same words. The three that are ours are marked as ours in
 * `CAMERA_OURS` and the UI says so, because a label the trainer has never heard
 * of, presented as if it were part of its vocabulary, is how a caption ends up
 * carrying a word no model knows.
 *
 * PINNED AGAINST THE BACKEND. `video-camera-contract.test.mjs` reads
 * backend/app/services/video_camera_motion.py and asserts this key list matches
 * CAMERA_LABELS exactly, in order — the labels arrive on the clip row already
 * derived, so a name that exists on one side only is a filter that silently
 * matches nothing.
 */

/** Label → what it says in the UI. ORDER IS THE BACKEND'S CANONICAL ORDER, and
 * it is what the chips and the card badge are rendered in: "pan right · zoom in
 * · handheld" reads as a sentence, alphabetical order does not. */
export const CAMERA_LABELS = {
  pan_left: 'Pan left',
  pan_right: 'Pan right',
  pan_up: 'Pan up',
  pan_down: 'Pan down',
  zoom_in: 'Zoom in',
  zoom_out: 'Zoom out',
  rolling: 'Rolling',
  static_shot: 'Static shot',
  handheld_shot: 'Handheld',
  slideshow: 'Slideshow',
  subject_motion: 'Subject moves',
}

/** The three labels that are this app's own rather than Hunyuan's fourteen. The
 * UI marks them, so nobody carries one into a caption expecting a trainer to
 * recognise it. */
export const CAMERA_OURS = ['rolling', 'slideshow', 'subject_motion']

/** The sentence under each chip. Every one of the three carries a limit rather
 * than a definition, because the limits are what a user filtering on these needs
 * and what they cannot see from the label. */
export const CAMERA_HINTS = {
  pan_left: 'The frame moves left across the scene.',
  pan_right: 'The frame moves right across the scene.',
  pan_up: 'The frame moves up across the scene.',
  pan_down: 'The frame moves down across the scene.',
  zoom_in: 'The framing tightens.',
  zoom_out: 'The framing widens.',
  rolling: 'The horizon turns — the camera rotates about its own axis. Not one '
    + 'of the trainer\'s own words, but it is the one movement a language model '
    + 'reading the footage reliably gets wrong, so it is measured here.',
  static_shot: 'Nothing moved enough to name — a tripod, a clamp, or a very '
    + 'steady pair of hands.',
  handheld_shot: 'The movement has a high-frequency part nobody is steering. '
    + 'Set from strong tremor, so gentle or stabilised handheld can read as '
    + 'static instead — the raw number is on the clip either way.',
  slideshow: 'The whole frame moved as one rigid picture, which is what a '
    + 'photograph panned across does. A real pan over a FLAT scene — a wall, a '
    + 'horizon, a distant skyline — has no depth either and can land here too.',
  subject_motion: 'Something in the shot moved more than the camera did, so the '
    + 'direction could not be read at all. No pan, zoom or roll is reported for '
    + 'these — a confident wrong answer would be worse than none.',
}

/** Pans and tilts are the same measurement here, and the UI says so once. A
 * camera that PIVOTS and one that SLIDES put the same movement on the sensor;
 * telling them apart needs depth. So the tilt half of the trainer's vocabulary
 * is never emitted, and this is the sentence that explains the absence rather
 * than leaving a user hunting for a "tilt up" chip that will never appear. */
export const CAMERA_FACET_NOTE =
  'Pan covers both pivoting and sliding — nothing in a flat picture separates '
  + 'them. Orbits are not detected at all.'

/** [{name, label, count, ours}] for the labels present, in canonical order.
 *
 * Present-only, like the flag chips: a facet listing all eleven with nine zeroes
 * is a wall the eye has to filter before the data does. Counted over the clips
 * LOADED, which the caller says out loud when the grid is paged. */
export function cameraChips(clips) {
  const counts = {}
  for (const clip of clips || []) {
    for (const name of clip.camera || []) {
      counts[name] = (counts[name] || 0) + 1
    }
  }
  return Object.keys(CAMERA_LABELS)
    .filter((name) => counts[name])
    .map((name) => ({
      name,
      label: CAMERA_LABELS[name],
      count: counts[name],
      ours: CAMERA_OURS.includes(name),
    }))
}

/** The clips carrying one camera label, or every clip when none is selected.
 *
 * A shot carries SEVERAL labels (a handheld pan that also zooms is all three),
 * so this is a membership test and not an equality one — filtering on 'zoom_in'
 * must keep the handheld pan that also zooms, or the two filters would silently
 * exclude each other. */
export function filterByCamera(clips, name) {
  if (!name) return clips
  return (clips || []).filter((clip) => (clip.camera || []).includes(name))
}

/** The camera clause for a card badge, e.g. 'Pan right · Zoom in'. '' when the
 * shot has no reading — never a placeholder, because an empty badge and a badge
 * reading "unknown" send a user to two different places. */
export function cameraBadge(clip) {
  const names = (clip && clip.camera) || []
  return names.map((name) => CAMERA_LABELS[name] || name).join(' · ')
}

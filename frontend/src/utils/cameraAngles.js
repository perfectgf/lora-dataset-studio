/* 📷 Camera angles — the vocabulary, the selection model and the wording.
 *
 * WHY THIS IS NOT "ANOTHER SHOT". The dataset's shot catalog can already ask
 * for a profile or a three-quarter view, and an edit model answers those by
 * turning the PERSON — the room behind stays exactly where it was. Measured on
 * this app's own Klein lane (2026-08-25, one reference, seed held constant,
 * every phrasing tried): the backdrop never moved. This lane moves the CAMERA,
 * so the backdrop reprojects with it. Same sentence, different picture, which
 * is why they are two verbs rather than one with a checkbox. The long version
 * lives in backend/app/services/camera_angles.py.
 *
 * Pure module, no JSX — `node --test` cannot parse JSX, and this is the part
 * with the cases worth pinning. The picker draws itself from these tables.
 *
 * ⚠️ Every `id` below is written into user databases (a produced picture stores
 * the pose it was asked for) and into localStorage. They mirror
 * backend/app/services/camera_angles.py character for character, and
 * cameraCatalogContract.test.js reads BOTH files so the two cannot drift.
 * Renaming one without an alias path strands every row already there.
 */

/** The LoRA's trigger. Present in every prompt it was trained on; without it
 *  the adapter is loaded and inert — and an inert adapter does not fail, it
 *  quietly turns the subject instead of the camera, which looks like a
 *  success. */
export const TRIGGER = '<sks>'

/** Where the camera stands. `degrees` is what the dial draws; 0° is the
 *  reference photo's own viewpoint, not a compass bearing. */
export const AZIMUTHS = [
  { id: 'front', degrees: 0, token: 'front view', label: 'Front' },
  { id: 'front_right', degrees: 45, token: 'front-right quarter view', label: 'Front-right' },
  { id: 'right', degrees: 90, token: 'right side view', label: 'Right side' },
  { id: 'back_right', degrees: 135, token: 'back-right quarter view', label: 'Back-right' },
  { id: 'back', degrees: 180, token: 'back view', label: 'Back' },
  { id: 'back_left', degrees: 225, token: 'back-left quarter view', label: 'Back-left' },
  { id: 'left', degrees: 270, token: 'left side view', label: 'Left side' },
  { id: 'front_left', degrees: 315, token: 'front-left quarter view', label: 'Front-left' },
]

/** How high the camera is. Ordered low → high, the way the ladder is drawn. */
export const ELEVATIONS = [
  { id: 'low', degrees: -30, token: 'low-angle shot', label: 'Low', hint: 'from below' },
  { id: 'eye', degrees: 0, token: 'eye-level shot', label: 'Eye level', hint: 'level with the subject' },
  { id: 'elevated', degrees: 30, token: 'elevated shot', label: 'Elevated', hint: 'slightly above' },
  { id: 'high', degrees: 60, token: 'high-angle shot', label: 'High', hint: 'looking down' },
]

/** How far the camera is. ⚠️ The loose axis — see DISTANCE_CAVEAT. */
export const DISTANCES = [
  { id: 'close', factor: 0.6, token: 'close-up', label: 'Close-up' },
  { id: 'medium', factor: 1.0, token: 'medium shot', label: 'Medium' },
  { id: 'wide', factor: 1.8, token: 'wide shot', label: 'Wide' },
]

/** Said on the screen, not discovered on a finished dataset. Measured before
 *  shipping: several poses asked at `medium` came back tighter than the
 *  reference. It is a hint the model mostly honours, not a focal length. */
export const DISTANCE_CAVEAT = 'Framing is approximate — the model treats distance as a hint.'

/** What the picture is assumed to already be. The dial lights this ring as
 *  "you are here"; picking it is allowed (it is the control that proves the
 *  lane works) and simply says what it is. */
export const REFERENCE_POSE = 'front/eye/medium'

/** Every pose that exists: 8 azimuths × 4 heights × 3 distances. */
export const POSE_COUNT = 96

/** The only ceiling is the vocabulary. There WAS an arbitrary 12 here, and it
 *  was wrong in the most ordinary case: eight sides at two distances is 16, so
 *  the cap refused a request nobody would call excessive. A limit that blocks
 *  the normal case to prevent a rare one is a bug with a justification.
 *
 *  What the button owed was to say what it costs, not to decide it — hence
 *  LONG_RUN_SECONDS below, and the view count that moves while you choose. */
export const MAX_VIEWS = POSE_COUNT

/** Past this, the cost line stops being a note and starts being a warning
 *  (~5 minutes). Nothing is blocked at any length. */
export const LONG_RUN_SECONDS = 300

/** Measured on a 4090: 12–16 s once the model is resident, ~54 s for the first
 *  view of a session (a 20 GB model loads from disk). Used to say what a
 *  selection will cost BEFORE it is spent, never to promise a deadline. */
export const SECONDS_PER_VIEW = 13
const FIRST_VIEW_LOAD_SECONDS = 54

export const CAMERA_ANGLE = 'camera_angle'

/** True when this row IS a camera view produced by the lane. */
export const isCameraView = (img) => img?.derivation_kind === CAMERA_ANGLE

export const poseId = (azimuth, elevation, distance) => `${azimuth}/${elevation}/${distance}`

const byId = (list) => Object.fromEntries(list.map((e) => [e.id, e]))
const AZ = byId(AZIMUTHS)
const EL = byId(ELEVATIONS)
const DI = byId(DISTANCES)

/** `'right/low/medium'` → `{azimuth, elevation, distance}`, or null.
 *  Null rather than a throw for anything malformed: this parses values that
 *  arrive from the server and from old rows, and a tile with an unreadable
 *  pose should render without a label, not crash the grid. */
export function parsePose(value) {
  if (typeof value !== 'string') return null
  const parts = value.split('/')
  // Exactly three, like the Python side: destructuring alone would accept
  // 'right/low/wide/extra' and quietly label a row from a string the server
  // would have refused.
  if (parts.length !== 3) return null
  const [a, e, d] = parts
  if (!AZ[a] || !EL[e] || !DI[d]) return null
  return { azimuth: a, elevation: e, distance: d }
}

/** The label a tile shows under a camera view. Null when unreadable. */
export function poseLabel(value) {
  const p = parsePose(value)
  if (!p) return null
  return `${AZ[p.azimuth].label} · ${EL[p.elevation].label} · ${DI[p.distance].label}`
}

/** The prompt one pose sends, in the LoRA's published grammar. Shown in the
 *  picker so what leaves the app is never a mystery. */
export function posePrompt(azimuth, elevation, distance) {
  if (!AZ[azimuth] || !EL[elevation] || !DI[distance]) return null
  return `${TRIGGER} ${AZ[azimuth].token} ${EL[elevation].token} ${DI[distance].token}`
}

/**
 * The selection model: the user picks AXES, and the views are their product.
 *
 * Ninety-six checkboxes would be a worse screen and a worse mental model —
 * someone covering a subject wants "all eight sides at eye level", which is one
 * gesture on an axis and eight on a grid. The product is also what makes the
 * cost legible: the count moves as an axis is toggled, before anything is spent.
 *
 * Order is deliberate: elevation outer, azimuth inner. A run therefore walks
 * all the way around the subject at one height before changing height, so an
 * interrupted run leaves a COMPLETE ring rather than a scattered handful — and
 * a complete ring is the thing that is actually useful as training data.
 */
export function posesFor({ azimuths = [], elevations = [], distances = [] }) {
  const out = []
  for (const d of DISTANCES) {
    if (!distances.includes(d.id)) continue
    for (const e of ELEVATIONS) {
      if (!elevations.includes(e.id)) continue
      for (const a of AZIMUTHS) {
        if (!azimuths.includes(a.id)) continue
        out.push(poseId(a.id, e.id, d.id))
      }
    }
  }
  return out
}

/**
 * Why this selection cannot be sent, or null when it can.
 *
 * Worded as the backend refuses (camera_angles.NO_VIEWS_PICKED /
 * TOO_MANY_VIEWS), so the screen explains itself BEFORE the click instead of
 * surfacing a 400 after it. A REASON rather than a boolean is what lets the
 * button be shown disabled with the reason attached.
 */
export function selectionRefusal(selection) {
  const poses = posesFor(selection)
  if (!poses.length) return 'pick at least one camera position'
  // Nothing else is refused. Length is a COST, not an error, and the footer
  // states it before the click.
  return null
}

/** Seconds a selection will take, model load included when it is not resident. */
export function runSeconds(count, { modelResident = false } = {}) {
  if (!count) return 0
  return count * SECONDS_PER_VIEW + (modelResident ? 0 : FIRST_VIEW_LOAD_SECONDS)
}

/** True when the run is long enough that the cost line should read as a
 *  warning rather than a note. Never blocks anything. */
export function isLongRun(count, opts) {
  return runSeconds(count, opts) >= LONG_RUN_SECONDS
}

/** Why 📷 cannot be offered for this picture, or null when it can. Mirrors the
 *  server guards (camera_angles.ALREADY_DERIVED / SOURCE_NOT_DONE). */
export function cameraRefusal(img) {
  if (!img || !Number.isInteger(Number(img.id))) {
    return 'This picture has no library entry to re-shoot.'
  }
  if (isCameraView(img)) {
    // A view of a view re-invents what the first pass already invented, and
    // sells the result as a photograph of the original scene.
    return 'A camera view cannot itself be re-shot from another angle.'
  }
  // ✨ An improve result IS allowed, deliberately. It is the same scene from the
  // same viewpoint, only cleaner — the best source this lane can be handed, not
  // a compounded guess. This used to refuse every derived row, and one look at a
  // real library settled it: the newest six tiles were all improve results, so
  // the verb was greyed out on exactly the pictures people keep.
  if (img.status && img.status !== 'done') return 'This image is still rendering.'
  return null
}

/** Why 📷 cannot be offered for THIS dataset image, or null when it can.
 *
 *  The dataset's own statuses, not the gallery's: a row here is keep / pending
 *  / reject / failed, and what qualifies it as a source is having its FILE —
 *  a pending row still rendering has none, a failed one never got one. An
 *  import qualifies (a real photo is the best source there is) and so does an
 *  ✨ improve result; only a camera view is refused, for the compounding
 *  reason the backend states. */
export function datasetCameraRefusal(img) {
  if (!img || !Number.isInteger(Number(img.id))) {
    return 'This picture has no dataset entry to re-shoot.'
  }
  if (isCameraView(img)) {
    return 'A camera view cannot itself be re-shot from another angle.'
  }
  if (!img.filename) return 'This image has no file yet.'
  return null
}

/** The dataset toast: results land as pending candidates HERE, in the
 *  keep/reject cycle — naming that is what stops "queued" reading as a dead
 *  click on a grid where nothing moves for a minute. */
export function datasetCameraLaunchMessage(queued) {
  return `${queued} camera view${queued > 1 ? 's' : ''} queued — they arrive in this `
    + 'dataset as pending candidates, with the angle already in the caption'
}

/** Roughly how long a selection will take, in words. Never a countdown: the
 *  queue is shared and a promise this cannot keep is worse than a range. */
export function costSentence(count, opts) {
  if (!count) return ''
  // Same arithmetic as isLongRun, on purpose: a warning that fires at a
  // different number from the one on screen is worse than no warning.
  const seconds = runSeconds(count, opts)
  const mins = Math.round(seconds / 60)
  const time = seconds < 90 ? 'about a minute' : `about ${mins} minutes`
  return `${count} view${count > 1 ? 's' : ''}, ${time}`
}

/** The toast after a run is queued. Names WHERE the pictures will appear —
 *  nothing moves on screen when the jobs start, and a bare "started" reads as
 *  a dead click. */
export function cameraLaunchMessage(queued) {
  return `${queued} camera view${queued > 1 ? 's' : ''} queued — they arrive here as they render`
}

/** The one-line explanation the picker opens with. It has one job: stop
 *  someone expecting a crop or a rotation of the picture they are looking at. */
export const CAMERA_INTRO =
  'Re-photograph this scene from another camera position. The subject stays put; '
  + 'the background moves with the camera, so what was behind the subject comes into view.'

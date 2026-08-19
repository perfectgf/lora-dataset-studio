/** 🎬 The video extra, reported as THREE pieces — and never as one verdict.
 *
 * This is the whole reason this lane exists as a separate one. Today a `.mp4`
 * dropped into an image bank is skipped in silence: no row, no warning, nothing.
 * Replacing that silence with a single "video is unavailable" banner would be
 * the same defect wearing a hat — it is how a user reinstalls the wrong thing.
 *
 * Decoding, shot detection and encoding fail INDEPENDENTLY and are fixed
 * differently: one is a pip package, one is a pip package that drags torch, one
 * is a binary on PATH. So the UI says which piece is missing, what it costs, and
 * — the part that matters most — what still works without it. With no encoder
 * you can scan a folder, detect every shot, watch them and triage the whole
 * bank; the only thing you cannot do is promote.
 *
 * PURE: no JSX, no fetch. `node --test` imports it directly.
 */

/** The three pieces, in the order they are needed. `fix` is what the user does,
 * not what failed — "install av" is a diagnosis, "Setup › install the video
 * extra" is an address. */
export const VIDEO_PIECES = [
  {
    key: 'decode',
    label: 'Reading video files',
    blurb: 'Opening your files to read their length, size and frame rate, and to grab thumbnails.',
    fix: 'Install the video extra from Setup.',
    // The /api/capabilities key the Setup card for this piece turns green —
    // the machine half of the `fix` sentence (see mlInstallCards.test.js).
    setupCap: 'video_decode',
  },
  {
    key: 'detect',
    label: 'Finding shots',
    blurb: 'Cutting each file at its shot boundaries, so you triage shots instead of whole rushes.',
    fix: 'Install the shot-detection extra from Setup (it pulls in torch).',
    setupCap: 'video_detect',
  },
  {
    key: 'encode',
    label: 'Cutting clips',
    // "Only promotion needs this" was true for four waves and stopped being
    // true when 🩻 Defects landed: that pass hands whole files to the same
    // binary. A blurb that still said "only promotion" would send someone
    // without ffmpeg looking for a different reason their sweep button is grey.
    blurb: 'Re-encoding the shots you kept into a training set, and sweeping '
      + 'your files for compression damage. Promotion and 🩻 Defects need this.',
    fix: 'Install ffmpeg, or put it on your PATH.',
  },
]

/** Which pieces a pass genuinely needs. Read off the service, not guessed:
 *   probe   → av (services/video_probe)
 *   detect  → transnetv2 (services/shot_detect)
 *   thumbs  → av + PIL. NOT ffmpeg — thumbnails are decoded in-process, which is
 *             why a missing encoder still leaves you a browsable grid.
 *   embed   → av + PIL, plus the ✨ Score interpreter (checked server-side).
 *   promote → ffmpeg, the only pass that writes media.
 * `pipeline` chains probe → detect → thumbs, so it needs both of theirs: with no
 * detector it would probe, find no shots and make no thumbnails, and report
 * success. */
export const PASS_REQUIREMENTS = {
  probe: ['decode'],
  detect: ['detect'],
  thumbs: ['decode'],
  measure: ['decode'],
  // Embedding decodes frames like every other reading pass. It ALSO needs an
  // interpreter that can run CLIP, which is not one of the three video pieces —
  // that one is checked server-side and refused with its own sentence, because
  // it is the ✨ Score environment and belongs to a different install step.
  embed: ['decode'],
  // Same shape as embed: frames are decoded here, the model runs in the ✨ Score
  // interpreter, and THAT requirement is checked server-side with its own sentence.
  caption: ['decode'],
  // ✂ Near-duplicates need NOTHING from the video extra, and that is the point
  // of building them on the vectors 🔎 Find scenes already cached: the pass
  // re-reads an .npz and does dot products. Requiring `decode` here would grey
  // the button out on a machine that can perfectly well answer the question.
  dedup: [],
  // 🔖 Watermarks decode ONE frame per shot, like every other reading pass. The
  // detector's own environment and weights are a separate install step, checked
  // server-side with its own sentence — the same split as embed and caption.
  watermark: ['decode'],
  // 🔳 Safe zone decodes three frames per shot, so `decode` and nothing else.
  // Its OCR engine is deliberately NOT listed here, and it is the only pass in
  // this table whose second dependency is left out on purpose: without it the
  // pass still measures letterbox and pillarbox bands on every shot and says so.
  // Requiring it would grey out a button that works — the exact mistake this
  // table exists to avoid, in the opposite direction.
  safezone: ['decode'],
  // 🩻 Defects is the ONLY reading pass that needs `encode` — and it needs
  // nothing else, which is the mirror image of every row above. The three
  // measurements are ffmpeg filters running inside ffmpeg's own decode loop, so
  // the file is never opened here: an install with `av` and no ffmpeg cannot
  // run it, and an install with ffmpeg and no `av` can. Listing `decode` too
  // would grey out a button that works, and listing `encode` on the others
  // would grey out ones that never touch the binary.
  defects: ['encode'],
  // 🤖 AI check decodes sixteen frames per shot with PyAV, so `decode` and
  // nothing else. The encoder runs in the ✨ Score interpreter — the same split
  // as embed, caption and watermark, checked server-side with its own sentence,
  // because that environment is a different install step and not one of the
  // three video pieces.
  aicheck: ['decode'],
  // 🎥 Camera tracks frames it decodes itself, in the app's own interpreter, so
  // `decode` and nothing else. Note that the decode probe now imports cv2 and
  // numpy alongside av for exactly this pass — the capability is one row on the
  // Setup screen and one install button, so it stays one requirement here.
  camera: ['decode'],
  pipeline: ['decode', 'detect'],
  promote: ['encode'],
}

const isMissing = (capability, key) => !(capability || {})[key]

/** The pieces that are NOT available, as full descriptors. Empty when ready. */
export function missingVideoPieces(capability) {
  return VIDEO_PIECES.filter((p) => isMissing(capability, p.key))
}

/** Null when the pass can run; otherwise the FIRST piece it is missing plus a
 * sentence that names the piece rather than the lane.
 *
 * Returning the piece (not a boolean) is what lets the button's tooltip say
 * "Cutting clips is unavailable — install ffmpeg" instead of "unavailable". */
export function passBlockedBy(capability, pass) {
  const required = PASS_REQUIREMENTS[pass] || []
  for (const key of required) {
    if (isMissing(capability, key)) {
      const piece = VIDEO_PIECES.find((p) => p.key === key)
      return { ...piece, why: `${piece.label} is unavailable. ${piece.fix}` }
    }
  }
  return null
}

/** What the workspace's capability strip says.
 *
 * `null` when everything is present — a green "all good" banner on a working
 * install is noise that trains people to skip the strip on the day it matters.
 *
 * `stillWorks` is the field this whole module is for: it is the difference
 * between "video is unavailable" and "you can do everything except promote".
 */
export function videoCapabilityNotice(capability) {
  const missing = missingVideoPieces(capability)
  if (missing.length === 0) return null
  const names = missing.map((p) => p.label)
  const working = []
  if (!isMissing(capability, 'decode')) working.push('scan your files')
  if (!isMissing(capability, 'detect')) working.push('find the shots')
  // Playback needs NOTHING installed here: the browser decodes the source it
  // streams from the app, so a bank with no `av` at all is still watchable.
  // Triage needs nothing either — a decision is a row in a database.
  working.push('watch any shot', 'triage')
  return {
    pieces: missing,
    // Names the pieces; never "video is unavailable".
    headline: missing.length === 3
      ? 'None of the video pieces are installed yet.'
      : `${joinEnglish(names)} ${missing.length === 1 ? 'is' : 'are'} missing.`,
    // Never empty, by construction: watching and triaging need no local piece,
    // and saying so is what keeps a half-installed bank usable rather than
    // feeling broken.
    stillWorks: `You can still ${joinEnglish(working)}.`,
    // The server's own sentence, kept verbatim: it names the exact package.
    detail: (capability || {}).detail || null,
  }
}

/** "a, b and c" — Oxford-free, because these are UI fragments, not prose. */
export function joinEnglish(items) {
  const list = items.filter(Boolean)
  if (list.length === 0) return ''
  if (list.length === 1) return list[0]
  return `${list.slice(0, -1).join(', ')} and ${list[list.length - 1]}`
}

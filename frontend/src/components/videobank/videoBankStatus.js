/** 🎬 Reading a video bank at a glance — counters, the live pass, and what to
 * click next.
 *
 * A video bank has a strict order of operations that an image bank does not:
 * detection needs the probe's frame rate, thumbnails need the detected shots,
 * and promotion needs kept clips. Running them out of order is not an error —
 * each pass simply finds nothing to do and reports success, which reads exactly
 * like "this app does not work with my files". So the workspace names the ONE
 * next step instead of laying out four equal buttons and wishing you luck.
 *
 * PURE: no JSX, no fetch.
 */
import { etaPhrase } from '../bank/passEta.js'

const n = (v) => Number(v) || 0

/** The counter line under the bank's name. Zeroes are dropped rather than shown
 * as "0 rejected": on a fresh bank that column is noise, and on a triaged one
 * its absence is information. */
export function countsSummary(counts) {
  const c = counts || {}
  const parts = [`${n(c.sources)} file${n(c.sources) === 1 ? '' : 's'}`]
  if (n(c.clips)) parts.push(`${n(c.clips)} shots`)
  if (n(c.keep)) parts.push(`${n(c.keep)} kept`)
  if (n(c.reject)) parts.push(`${n(c.reject)} rejected`)
  if (n(c.promoted)) parts.push(`${n(c.promoted)} promoted`)
  return parts.join(' · ')
}

/** Problems worth a line of their own, because they are silent otherwise: files
 * the decoder could not open, and files whose detection failed. Both leave a
 * bank looking thinner than the folder it points at, with no explanation. */
export function countsProblems(counts) {
  const c = counts || {}
  const out = []
  if (n(c.unreadable)) {
    out.push(`${n(c.unreadable)} file${n(c.unreadable) === 1 ? '' : 's'} could not be read`)
  }
  if (n(c.detect_errors)) {
    out.push(`${n(c.detect_errors)} file${n(c.detect_errors) === 1 ? '' : 's'} failed shot detection`)
  }
  return out
}

/** Human names for the passes — used by the buttons, the progress line and the
 * 409 "busy" refusal, so those three cannot describe the same pass differently. */
export const PASS_LABELS = {
  probe: 'Scan files',
  detect: 'Find shots',
  thumbs: 'Make thumbnails',
  measure: 'Measure quality',
  embed: 'Find scenes',
  caption: 'Describe shots',
  dedup: '✂ Duplicates',
  watermark: '🔖 Watermarks',
  safezone: '🔳 Safe zone',
  defects: '🩻 Defects',
  aicheck: '🤖 AI check',
  pipeline: 'Run everything',
  promote: 'Build the dataset',
}

/** Present participle, for "⏳ Finding shots — 3/12". */
export const PASS_RUNNING_LABELS = {
  probe: 'Scanning files',
  detect: 'Finding shots',
  thumbs: 'Making thumbnails',
  measure: 'Measuring clips',
  embed: 'Looking at shots',
  caption: 'Describing shots',
  dedup: 'Comparing shots',
  watermark: 'Looking for watermarks',
  safezone: 'Measuring the safe zone',
  // "Sweeping files", not "shots": this pass advances one FILE at a time while
  // its progress bar counts clips, and a participle that promised shots would
  // make a bar standing still for a whole rush look stuck.
  defects: 'Sweeping files',
  // "Checking how shots move", not "Detecting AI": the participle a user reads
  // for twenty minutes must describe the measurement, not promise the verdict.
  aicheck: 'Checking how shots move',
  pipeline: 'Running everything',
  promote: 'Building the dataset',
}

export function passLabel(kind) {
  return PASS_LABELS[kind] || kind
}

/** The live pass, as one readable line — or null when nothing is running.
 *
 * A finished job is deliberately NOT running: the server keeps the snapshot for
 * a while after `finished` so the last result can be read, and treating that as
 * activity leaves a spinner up forever. */
/** Which `counts` key already holds the work a given pass does. A pass only ever
 * iterates what is LEFT, so this is what turns its own slice back into the whole
 * picture. */
const _PASS_COUNTS = {
  detect: (c) => n(c?.detected) + n(c?.detect_errors),
  thumbs: (c) => n(c?.thumbs),
  embed: (c) => n(c?.embedded),
  caption: (c) => n(c?.captioned),
  probe: (c) => n(c?.probed) + n(c?.unreadable),
}

/** Overall progress of a running pass: `{done, total, alreadyDone, resumed}`, or
 * null when nothing is running.
 *
 * WHY THIS EXISTS. A resumed pass filters out everything already done BEFORE it
 * fixes its total, so it honestly reports "3 of 117" while 132 of 246 sources are
 * cut. Accurate, and it reads as a restart from zero — which makes people afraid
 * to ever stop a one-hour pass, and that fear costs more than the display bug.
 *
 * `alreadyDone` is derived by subtraction rather than snapshotted: the counts are
 * live and already include what this job has done so far, so the difference is
 * exactly what preceded it. That also keeps failed files inside the total — they
 * are not retried by a plain resume, and leaving them out would shrink the total
 * a little more on every restart.
 */
export function passProgress(activity, counts) {
  if (!activity || activity.finished) return null
  // No counts, no overall view: fall back to the job's own numbers rather than
  // computing an "already done" of zero, which would report a pass as being at
  // its very start no matter how far along it is.
  if (!counts) return null
  const fromCounts = _PASS_COUNTS[activity.kind]
  const total = n(activity.total)
  if (!fromCounts || !total) return null
  const done = fromCounts(counts)
  const alreadyDone = Math.max(0, done - n(activity.done))
  return { done, total: total + alreadyDone, alreadyDone, resumed: alreadyDone > 0 }
}

/** "129 sources are already cut and stay cut — stopping is safe." Null when there
 * is nothing yet to lose.
 *
 * Worth saying out loud rather than leaving to trust: the guarantee is real (a
 * source is marked done only in the same transaction that writes its shots, so an
 * interrupted one is simply picked up again), but a guarantee nobody can see
 * protects nobody. */
export function resumeSafetyNote(activity, counts) {
  const p = passProgress(activity, counts)
  if (!p || !p.resumed) return null
  return `${p.alreadyDone} already done and kept — stopping is safe, this pass resumes where it left off.`
}

export function activityLine(activity, counts) {
  if (!activity || activity.finished) return null
  const label = PASS_RUNNING_LABELS[activity.kind] || activity.kind || 'Working'
  const overall = passProgress(activity, counts)
  const done = overall ? overall.done : n(activity.done)
  const total = overall ? overall.total : n(activity.total)
  const progress = total ? ` — ${done}/${total}` : ''
  // Free ride: the video lane keys into the SAME `bank_jobs` registry as the
  // image passes (see `video_bank_service.job_key`), so its snapshot already
  // carries the measured remaining time. The figure is about the work THIS job
  // has left, which is the right one even on a resume — `overall` above only
  // widens the counter to include what a previous run finished.
  const eta = etaPhrase(activity)
  const detail = activity.detail ? ` (${activity.detail})` : ''
  return `${label}${progress}${eta ? ` · ${eta}` : ''}${detail}`
}

/** 0–100, or null when the job does not know its total (a pass that is still
 * counting). Null must render as an indeterminate bar, never as 0 % — a bar
 * pinned at zero for two minutes reads as a hang. */
export function activityPercent(activity, counts) {
  if (!activity || activity.finished || !n(activity.total)) return null
  const overall = passProgress(activity, counts)
  const done = overall ? overall.done : n(activity.done)
  const total = overall ? overall.total : n(activity.total)
  return Math.min(100, Math.round((done / total) * 100))
}

/** True while a pass owns the bank. Every pass button reads this, so the UI
 * refuses the click the server would answer 409 to. */
export function isBusy(activity) {
  return !!activity && !activity.finished
}

/** The result of the pass that just ended, or null. Shown once, then dropped —
 * "done — 340 shots" is worth a toast and worth nothing on the tenth poll. */
export function finishedOutcome(activity) {
  if (!activity || !activity.finished) return null
  if (activity.cancelled) return { tone: 'info', text: `${passLabel(activity.kind)} — stopped.` }
  if (activity.error) return { tone: 'error', text: activity.error }
  return { tone: 'success', text: activity.detail || `${passLabel(activity.kind)} — done.` }
}

/** Should the UI announce this job snapshot, given what it last announced?
 *
 * Answers `{announce, marker}`; the caller stores `marker` and passes it back.
 *
 * The poll returns a NEW object every two seconds and the server keeps a
 * finished job's snapshot around for a while, so "announce when finished" alone
 * repeats the toast on a timer. Keying on the job's contents fixes that — and
 * introduces a second bug on its own: running the SAME pass twice with the same
 * result produces the same key, so the second completion is swallowed, silently.
 *
 * Hence the null marker while a job is RUNNING: every genuine second run passes
 * through that state, which re-arms the announcement without needing a job id
 * the server does not give us.
 */
export function announcement(previousMarker, activity) {
  if (!activity) return { announce: false, marker: previousMarker }
  if (!activity.finished) return { announce: false, marker: null }
  const marker = [activity.kind, activity.done, activity.error || '',
    activity.cancelled ? 'x' : ''].join(':')
  if (marker === previousMarker) return { announce: false, marker }
  return { announce: true, marker, outcome: finishedOutcome(activity) }
}

/** The ONE thing to do next, given where the bank is.
 *
 * Order matters and follows the data dependency, not the button row: a bank with
 * files but no probe cannot detect; a bank with shots but no thumbnails shows an
 * empty-looking grid; a bank fully triaged is waiting to be promoted.
 *
 * `blocked` (from videoCapability.passBlockedBy) turns the suggestion into an
 * explanation of what to install instead of a button that 503s. */
export function nextStep(counts, capability, blockedBy) {
  const c = counts || {}
  if (!n(c.sources)) {
    return { pass: null, text: 'This bank is empty — press ↻ Rescan folder if you have added files.' }
  }
  const step = (pass, text) => {
    const blocked = blockedBy ? blockedBy(capability, pass) : null
    return blocked ? { pass, text, blocked } : { pass, text }
  }
  if (n(c.probed) < n(c.sources)) {
    return step('pipeline', 'Start with ▶ Run everything — it scans your files, finds the shots and makes the thumbnails in one go.')
  }
  if (!n(c.clips)) {
    return step('detect', 'Your files are scanned. Find the shots next.')
  }
  if (n(c.thumbs) < n(c.clips)) {
    return step('thumbs', 'The shots are cut. Make the thumbnails so you can see what you are triaging.')
  }
  if (n(c.pending)) {
    return { pass: null, text: `${n(c.pending)} shot${n(c.pending) === 1 ? '' : 's'} still to triage — keep the ones worth training on.` }
  }
  if (n(c.keep)) {
    return step('promote', 'Everything is triaged. Build the training set from what you kept.')
  }
  return { pass: null, text: 'Nothing is kept yet — keep a few shots before building a dataset.' }
}

// --- one source file, formatted ------------------------------------------------

/** "1:23:45", "4:07" — h:mm:ss past the hour, because rushes routinely are. */
export function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(Number(seconds))) return '—'
  const total = Math.max(0, Math.round(Number(seconds)))
  const s = String(total % 60).padStart(2, '0')
  const m = Math.floor(total / 60) % 60
  const h = Math.floor(total / 3600)
  return h > 0 ? `${h}:${String(m).padStart(2, '0')}:${s}` : `${m}:${s}`
}

/** Binary units, one decimal past a kilobyte. These files are measured in GB and
 * the difference between 1.2 and 12 GB is the difference between a coffee and an
 * afternoon. */
export function formatFileSize(bytes) {
  const b = Number(bytes)
  if (!Number.isFinite(b) || b <= 0) return '—'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = b
  let i = 0
  while (value >= 1024 && i < units.length - 1) { value /= 1024; i += 1 }
  return `${i === 0 ? value : value.toFixed(1)} ${units[i]}`
}

/** "1920×1080 · 29.97 fps · h264", with the unknown parts simply absent rather
 * than rendered as "null fps". */
export function sourceGeometry(source) {
  const s = source || {}
  const parts = []
  if (s.width && s.height) parts.push(`${s.width}×${s.height}`)
  if (s.fps_native) parts.push(`${Math.round(Number(s.fps_native) * 100) / 100} fps`)
  if (s.codec) parts.push(s.codec)
  return parts.join(' · ')
}

/** "High · 5.2 Mb/s · 0.215 bpp" — how hard this file was squeezed, or ''.
 *
 * ITS OWN LINE on the card, not appended to `sourceGeometry`. That line already
 * carries three facts, and a fourth, fifth and sixth would wrap into three lines
 * at 400 px wide — the width every card in this app has to survive.
 *
 * BITS PER PIXEL is the number worth reading and the reason the other two are
 * shown at all: a bitrate means nothing without a resolution to divide it by
 * (5 Mb/s is generous at 480p and starving at 4K), while bits per pixel per
 * frame is comparable across a mixed bank. Roughly, under 0.05 is visibly
 * damaged and over 0.15 is comfortable for ordinary 8-bit H.264.
 *
 * INFORMATIVE, never a cut, and that is a deliberate refusal rather than an
 * omission: bits per pixel PREDICTS compression damage, and the 🩻 Defects pass
 * MEASURES it. Offering a threshold on the prediction while the measurement sits
 * one panel away would be the app guessing in front of its own answer.
 *
 * Silent when the container did not say — MKV and WebM routinely carry no
 * per-stream bitrate, so a dash there is a property of the file rather than a
 * pass nobody ran, and inventing a number from the file size would fold the
 * audio track into a video statistic.
 */
export function sourceEncoding(source) {
  const s = source || {}
  const parts = []
  if (s.profile) parts.push(String(s.profile))
  const rate = Number(s.bit_rate)
  if (Number.isFinite(rate) && rate > 0) {
    parts.push(rate >= 1e6 ? `${(rate / 1e6).toFixed(1)} Mb/s`
      : `${Math.round(rate / 1e3)} kb/s`)
  }
  const bpp = Number(s.bits_per_pixel)
  if (Number.isFinite(bpp) && bpp > 0) parts.push(`${bpp.toFixed(3)} bpp`)
  return parts.join(' · ')
}

/** The state chip on a source row.
 *
 * "Not scanned yet" and "could not be read" are DIFFERENT and used to look the
 * same (both showed nothing): the first is waiting for a pass, the second means
 * this file will never contribute a shot and the folder is quietly short. */
export function sourceState(source) {
  const s = source || {}
  if (s.probe_state === 'unreadable') {
    return { tone: 'error', label: 'Unreadable', title: 'The decoder could not open this file — it will produce no shots.' }
  }
  if (!s.probe_state) {
    return { tone: 'idle', label: 'Not scanned', title: 'Run the scan pass to read this file’s length, size and frame rate.' }
  }
  if (s.detect_state === 'error') {
    return { tone: 'error', label: 'Detection failed', title: 'This file was read, but shot detection failed on it.' }
  }
  if (s.detect_state === 'single') {
    // Its own badge and not "1 shot": the number is the same, the meaning is
    // not. This file was DECLARED a single take, which is why the bank-wide
    // re-cut and the detection pass both walk past it — and someone wondering
    // why it never changes has to be able to see the reason on the card.
    return { tone: 'info', label: 'Single shot',
             title: 'You marked this file as one continuous take. Bulk passes leave it alone; ↻ Re-detect this file is the way back.' }
  }
  if (s.detect_state === 'ok') {
    const clips = n(s.clips)
    return { tone: 'ok', label: `${clips} shot${clips === 1 ? '' : 's'}`, title: 'Shots detected.' }
  }
  return { tone: 'info', label: 'Scanned', title: 'Read, but no shot detection has run on it yet.' }
}

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  countsSummary, countsProblems, passLabel, activityLine, activityPercent,
  isBusy, finishedOutcome, announcement, nextStep, formatDuration,
  formatFileSize, sourceGeometry, sourceEncoding, sourceState, passProgress,
  resumeSafetyNote,
} from './videoBankStatus.js'
import { passBlockedBy } from './videoCapability.js'

const COUNTS = {
  sources: 12, probed: 12, unreadable: 0, detected: 12, detect_errors: 0,
  clips: 340, pending: 0, keep: 128, reject: 212, promoted: 0, thumbs: 340,
}
const READY = { ok: true, detail: '', decode: true, detect: true, encode: true }

test('the summary drops the zero columns', () => {
  assert.equal(countsSummary(COUNTS), '12 files · 340 shots · 128 kept · 212 rejected')
  assert.equal(countsSummary({ sources: 1 }), '1 file')
  assert.equal(countsSummary(null), '0 files')
})

test('unreadable and undetectable files get their own line, never silence', () => {
  // Both leave the bank thinner than the folder it points at. Folding them into
  // the summary is how "where did my other 40 files go" happens.
  assert.deepEqual(countsProblems({ ...COUNTS, unreadable: 3, detect_errors: 1 }),
    ['3 files could not be read', '1 file failed shot detection'])
  assert.deepEqual(countsProblems(COUNTS), [])
})

// ---- the live pass -----------------------------------------------------------

test('a running pass reads as a sentence with its progress', () => {
  assert.equal(
    activityLine({ kind: 'detect', done: 3, total: 12, finished: false, detail: null }),
    'Finding shots — 3/12')
  assert.equal(
    activityLine({ kind: 'thumbs', done: 0, total: 0, finished: false, detail: 'making thumbnails' }),
    'Making thumbnails (making thumbnails)')
})

test('a FINISHED job is not activity', () => {
  // The server keeps the snapshot around after `finished` so the result can be
  // read. Treating that as running leaves the spinner up forever.
  assert.equal(activityLine({ kind: 'detect', finished: true, done: 12, total: 12 }), null)
  assert.equal(isBusy({ kind: 'detect', finished: true }), false)
  assert.equal(isBusy({ kind: 'detect', finished: false }), true)
  assert.equal(isBusy(null), false)
})

test('a pass with no known total is indeterminate, NOT zero percent', () => {
  // A bar pinned at 0 % for two minutes reads as a hang.
  assert.equal(activityPercent({ kind: 'probe', done: 0, total: 0, finished: false }), null)
  assert.equal(activityPercent({ kind: 'probe', done: 3, total: 12, finished: false }), 25)
  assert.equal(activityPercent({ kind: 'probe', done: 12, total: 12, finished: true }), null)
})

test('the outcome of a finished pass distinguishes stopped from failed', () => {
  assert.equal(finishedOutcome({ finished: true, cancelled: true, kind: 'detect' }).tone, 'info')
  assert.equal(finishedOutcome({ finished: true, error: 'boom', kind: 'detect' }).tone, 'error')
  assert.equal(finishedOutcome({ finished: true, kind: 'detect', detail: 'done — 340 shots' }).text,
    'done — 340 shots')
  assert.equal(finishedOutcome({ finished: false }), null)
})

test('a finished pass is announced exactly once, not on every poll', () => {
  // The poll returns a new object every 2 s and the server keeps a finished
  // job's snapshot for a while — "announce when finished" alone toasts on a timer.
  const done = { kind: 'detect', done: 12, total: 12, finished: true, detail: 'done — 340 shots' }
  const first = announcement(null, done)
  assert.equal(first.announce, true)
  assert.equal(first.outcome.text, 'done — 340 shots')
  assert.equal(announcement(first.marker, { ...done }).announce, false)
  assert.equal(announcement(first.marker, { ...done }).announce, false)
})

test('running the SAME pass again is announced again', () => {
  // The bug the content key introduces on its own: a second identical run
  // produces the same key and is swallowed. A running snapshot re-arms it.
  const done = { kind: 'thumbs', done: 340, total: 340, finished: true, detail: 'done' }
  const after = announcement(null, done)
  const running = announcement(after.marker, { kind: 'thumbs', done: 3, total: 340, finished: false })
  assert.equal(running.announce, false)
  assert.equal(running.marker, null, 'a running job must clear the marker')
  assert.equal(announcement(running.marker, { ...done }).announce, true)
})

test('no job at all changes nothing', () => {
  assert.deepEqual(announcement('m', null), { announce: false, marker: 'm' })
})

test('a failed and a cancelled pass are announced with their own tone', () => {
  assert.equal(announcement(null, { kind: 'detect', finished: true, error: 'boom' }).outcome.tone, 'error')
  assert.equal(announcement(null, { kind: 'detect', finished: true, cancelled: true }).outcome.tone, 'info')
})

test('pass names are shared, so the button and the 409 agree', () => {
  assert.equal(passLabel('pipeline'), 'Run everything')
  assert.equal(passLabel('promote'), 'Build the dataset')
  assert.equal(passLabel('something-new'), 'something-new')
})

// ---- what to do next ---------------------------------------------------------

test('the next step follows the DATA dependency, not the button row', () => {
  // Each pass reports success on an empty input, so a wrong order reads as
  // "this app does not work with my files" rather than as a mistake.
  assert.match(nextStep({ sources: 0 }, READY, passBlockedBy).text, /empty/)
  assert.equal(nextStep({ sources: 12, probed: 0 }, READY, passBlockedBy).pass, 'pipeline')
  assert.equal(nextStep({ sources: 12, probed: 12, clips: 0 }, READY, passBlockedBy).pass, 'detect')
  assert.equal(
    nextStep({ sources: 12, probed: 12, clips: 340, thumbs: 0 }, READY, passBlockedBy).pass,
    'thumbs')
  assert.match(
    nextStep({ sources: 12, probed: 12, clips: 340, thumbs: 340, pending: 40 }, READY, passBlockedBy).text,
    /40 shots still to triage/)
  assert.equal(nextStep(COUNTS, READY, passBlockedBy).pass, 'promote')
})

test('a triaged bank with nothing kept is told so, not sent to promote', () => {
  const step = nextStep({ ...COUNTS, keep: 0, reject: 340 }, READY, passBlockedBy)
  assert.equal(step.pass, null)
  assert.match(step.text, /Nothing is kept yet/)
})

test('the suggested step carries WHAT IS MISSING when it cannot run', () => {
  // Suggesting "Build the dataset" on an install with no ffmpeg would be a
  // button that 503s. The suggestion stays, with the reason attached.
  const noEncoder = { ok: false, decode: true, detect: true, encode: false, detail: 'missing: ffmpeg' }
  const step = nextStep(COUNTS, noEncoder, passBlockedBy)
  assert.equal(step.pass, 'promote')
  assert.equal(step.blocked.key, 'encode')
  assert.match(step.blocked.why, /ffmpeg/)
})

// ---- formatting --------------------------------------------------------------

test('durations pass the hour', () => {
  assert.equal(formatDuration(47), '0:47')
  assert.equal(formatDuration(247), '4:07')
  assert.equal(formatDuration(5025), '1:23:45')
  assert.equal(formatDuration(null), '—')
  assert.equal(formatDuration('nope'), '—')
})

test('file sizes read in the unit these files actually use', () => {
  assert.equal(formatFileSize(512), '512 B')
  assert.equal(formatFileSize(1536), '1.5 KB')
  assert.equal(formatFileSize(6 * 1024 ** 3), '6.0 GB')
  assert.equal(formatFileSize(0), '—')
  assert.equal(formatFileSize(null), '—')
})

test('geometry omits what is unknown instead of printing null', () => {
  assert.equal(sourceGeometry({ width: 1920, height: 1080, fps_native: 29.97, codec: 'h264' }),
    '1920×1080 · 29.97 fps · h264')
  assert.equal(sourceGeometry({ codec: 'h264' }), 'h264')
  assert.equal(sourceGeometry({}), '')
  assert.equal(sourceGeometry(null), '')
})

test('"not scanned yet" and "could not be read" are DIFFERENT states', () => {
  // They used to look identical (both showed nothing). One is waiting for a
  // pass; the other means the folder is permanently short by that file.
  assert.equal(sourceState({ probe_state: null }).label, 'Not scanned')
  assert.equal(sourceState({ probe_state: 'unreadable' }).label, 'Unreadable')
  assert.equal(sourceState({ probe_state: 'unreadable' }).tone, 'error')
})

test('a source reports its shot count once detection has run', () => {
  assert.equal(sourceState({ probe_state: 'ok', detect_state: 'ok', clips: 41 }).label, '41 shots')
  assert.equal(sourceState({ probe_state: 'ok', detect_state: 'ok', clips: 1 }).label, '1 shot')
  assert.equal(sourceState({ probe_state: 'ok', detect_state: 'error' }).label, 'Detection failed')
  assert.equal(sourceState({ probe_state: 'ok', detect_state: null }).label, 'Scanned')
})

// --- resuming a stopped pass --------------------------------------------------
// A pass only ever iterates what is LEFT to do, so a resumed job legitimately
// reports "3 of 117" while 132 of 246 sources are actually cut. That is accurate
// and it reads as a restart from zero — which is the one thing that would make
// someone afraid to ever stop a one-hour pass.

test('a resumed pass reports overall progress, not just its own slice', () => {
  const p = passProgress({ kind: 'detect', done: 3, total: 117, finished: false },
                         { detected: 132, detect_errors: 0 })
  assert.equal(p.done, 132)
  assert.equal(p.total, 246)
  assert.equal(p.alreadyDone, 129)
})

test('a pass that starts from nothing shows no resume note', () => {
  const p = passProgress({ kind: 'detect', done: 10, total: 246, finished: false },
                         { detected: 10, detect_errors: 0 })
  assert.equal(p.alreadyDone, 0)
  assert.equal(p.resumed, false)
})

test('files that failed detection still count as already visited', () => {
  // They are not retried by a plain resume, so leaving them out would make the
  // total shrink every time the pass is restarted.
  const p = passProgress({ kind: 'detect', done: 1, total: 10, finished: false },
                         { detected: 40, detect_errors: 5 })
  assert.equal(p.alreadyDone, 44)
})

test('the thumbnail and probe passes resume the same way', () => {
  assert.equal(passProgress({ kind: 'thumbs', done: 2, total: 20, finished: false },
                            { thumbs: 302 }).total, 320)
  assert.equal(passProgress({ kind: 'probe', done: 5, total: 50, finished: false },
                            { probed: 105 }).total, 150)
})

test('a finished or absent pass has no progress to report', () => {
  assert.equal(passProgress({ finished: true, done: 5, total: 5 }, COUNTS), null)
  assert.equal(passProgress(null, COUNTS), null)
})

test('the activity line shows the overall figures when a pass was resumed', () => {
  const line = activityLine({ kind: 'detect', done: 3, total: 117, finished: false },
                            { detected: 132, detect_errors: 0 })
  assert.match(line, /132\/246/)
  assert.doesNotMatch(line, /3\/117/)
})

test('the activity line is unchanged when nothing was done before', () => {
  const line = activityLine({ kind: 'detect', done: 10, total: 246, finished: false },
                            { detected: 10, detect_errors: 0 })
  assert.match(line, /10\/246/)
})

test('the percentage follows the overall progress too', () => {
  // Otherwise a resumed pass shows a bar near zero while most of the work is done.
  const pct = activityPercent({ kind: 'detect', done: 3, total: 117, finished: false },
                              { detected: 132, detect_errors: 0 })
  assert.equal(pct, 54)
})

test('stopping is described as safe, and says exactly what is kept', () => {
  const note = resumeSafetyNote({ kind: 'detect', done: 3, total: 117, finished: false },
                                { detected: 132, detect_errors: 0 })
  assert.match(note, /129/)
  assert.match(note, /kept|keeps|safe/i)
})

test('no safety note when there is nothing yet to lose', () => {
  assert.equal(resumeSafetyNote({ kind: 'detect', done: 1, total: 246, finished: false },
                                { detected: 1 }), null)
})

test('the squeeze line reads profile, bitrate and bits per pixel', () => {
  assert.equal(
    sourceEncoding({ profile: 'High', bit_rate: 5200000, bits_per_pixel: 0.1004 }),
    'High · 5.2 Mb/s · 0.100 bpp')
  // Below a megabit, kb/s — "0.4 Mb/s" reads as a rounding artefact.
  assert.equal(sourceEncoding({ bit_rate: 420000 }), '420 kb/s')
})

test('a container that carries no bitrate shows nothing rather than a zero', () => {
  // MKV and WebM routinely carry no per-stream bitrate — measured, on files
  // holding the same footage as an .mp4 that reports one fine. That absence is a
  // property of the container, and "0.000 bpp" would read as a broken file.
  assert.equal(sourceEncoding({ profile: 'Profile 0', bit_rate: null,
    bits_per_pixel: null }), 'Profile 0')
  assert.equal(sourceEncoding({ bit_rate: 0, bits_per_pixel: 0 }), '')
  assert.equal(sourceEncoding({}), '')
  assert.equal(sourceEncoding(null), '')
})

test('the squeeze is its own line and never lengthens the geometry one', () => {
  // Both lines have to survive a 400 px card. Six facts on one line wrap into
  // three, which is what put this on a line of its own.
  const source = { width: 1920, height: 1080, fps_native: 25, codec: 'h264',
    profile: 'High', bit_rate: 5200000, bits_per_pixel: 0.1 }
  assert.ok(!sourceGeometry(source).includes('bpp'))
  assert.ok(!sourceGeometry(source).includes('Mb/s'))
  assert.ok(!sourceEncoding(source).includes('1920'))
})

test('the defect sweep is a named pass everywhere a pass is named', () => {
  // A pass missing from these tables shows the user a raw identifier in the
  // button, in the progress line AND in the 409 "busy" refusal.
  assert.equal(passLabel('defects'), '🩻 Defects')
  assert.match(activityLine({ kind: 'defects', done: 3, total: 40 }, COUNTS),
    /Sweeping files/)
})

test('the sweep is the one reading pass that needs the encoder', () => {
  // Every other reading pass decodes with PyAV; this one hands whole files to
  // the ffmpeg binary. An install with `av` and no ffmpeg must be refused for
  // the binary it is actually missing, not for the extra it already has.
  const noFfmpeg = { ok: false, decode: true, detect: true, encode: false }
  assert.equal(passBlockedBy(noFfmpeg, 'measure'), null)
  assert.equal(passBlockedBy(noFfmpeg, 'safezone'), null)
  const blocked = passBlockedBy(noFfmpeg, 'defects')
  assert.ok(blocked, 'the sweep must be refused without ffmpeg')
  assert.match(blocked.why, /ffmpeg/i)

  const noAv = { ok: false, decode: false, detect: true, encode: true }
  assert.equal(passBlockedBy(noAv, 'defects'), null,
    'the sweep never opens the file itself, so `av` is not its dependency')
})

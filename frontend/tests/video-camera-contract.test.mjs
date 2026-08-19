/** 🎥 The camera facet, pinned against the Python that produces it.
 *
 * The labels arrive on the clip row ALREADY DERIVED — the backend reads the raw
 * rates and emits names. So a name that exists on one side only is not a type
 * error anywhere: it is a filter chip that silently matches nothing, or a label
 * that arrives and renders as a raw identifier. Neither shows up in any other
 * test, which is why this one reads the source of both sides.
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

import {
  CAMERA_LABELS, CAMERA_OURS, CAMERA_HINTS, CAMERA_FACET_NOTE,
  cameraChips, filterByCamera, cameraBadge,
} from '../src/components/videobank/videoCameraMotion.js'
import { FLAG_LABELS, thresholdFields } from '../src/components/videobank/videoMetricsFilter.js'
import { PASS_LABELS, PASS_RUNNING_LABELS, passLabel } from '../src/components/videobank/videoBankStatus.js'
import { passBlockedBy } from '../src/components/videobank/videoCapability.js'

const here = dirname(fileURLToPath(import.meta.url))
const python = readFileSync(
  join(here, '..', '..', 'backend', 'app', 'services', 'video_camera_motion.py'),
  'utf8')

/** A tuple of quoted strings out of the Python source, in order. */
function pyTuple(name) {
  // `\r?\n`: git hands this file back in CRLF on a Windows checkout, and the
  // author's worktree had it in LF — so an `\n`-only tail passed where it was
  // written and failed in every fresh clone.
  const match = python.match(new RegExp(`^${name} = \\(([\\s\\S]*?)\\)\\r?\\n`, 'm'))
  assert.ok(match, `${name} not found in video_camera_motion.py`)
  return [...match[1].matchAll(/'([a-z_]+)'/g)].map((m) => m[1])
}

test('the facet offers every label the backend emits, in the same order', () => {
  // ORDER included on purpose: it is what the chips and the card badge render
  // in, and the backend calls it canonical because "pan right · zoom in ·
  // handheld" reads as a sentence while alphabetical order does not.
  assert.deepEqual(Object.keys(CAMERA_LABELS), pyTuple('CAMERA_LABELS'))
})

test('the labels marked as ours are exactly the ones not borrowed', () => {
  const all = pyTuple('CAMERA_LABELS')
  const borrowed = new Set(pyTuple('HUNYUAN_LABELS'))
  assert.deepEqual(CAMERA_OURS, all.filter((n) => !borrowed.has(n)))
})

test('every label has a hint, and every hint belongs to a label', () => {
  assert.deepEqual(Object.keys(CAMERA_HINTS).sort(),
    Object.keys(CAMERA_LABELS).sort())
})

test('the three labels that are ours each state a limit rather than a definition', () => {
  // A user filtering on these needs the limit, and it is the one thing the label
  // itself cannot carry. Checked as "long enough to be a sentence about the
  // edge case" rather than by matching words, which would pin prose.
  for (const name of CAMERA_OURS) {
    assert.ok(CAMERA_HINTS[name].length > 90,
      `${name} needs a hint that says where it goes wrong`)
  }
})

test('the missing half of the vocabulary is explained somewhere the user reads', () => {
  // People WILL look for "tilt up"; the note above the chips is what stops that
  // becoming "the detection is broken".
  assert.match(CAMERA_FACET_NOTE, /pan covers both/i)
  assert.match(CAMERA_FACET_NOTE, /orbit/i)
})

test('no camera label is also a quality flag', () => {
  // They are two rows with two meanings — a description and an accusation. One
  // name in both lists would make a shot appear in both, wearing amber.
  for (const name of Object.keys(CAMERA_LABELS)) {
    assert.ok(!(name in FLAG_LABELS), `${name} is in both the camera facet and the flags`)
  }
})

test('the pass is registered in both voices and is not greyed out where it works', () => {
  assert.equal(PASS_LABELS.camera, '🎥 Camera')
  assert.ok(PASS_RUNNING_LABELS.camera)
  assert.equal(passLabel('camera'), '🎥 Camera')
  assert.equal(passBlockedBy({ decode: true, detect: true, encode: true }, 'camera'), null)
  // It decodes, so it is blocked without the decode extra and by nothing else —
  // an install with no ffmpeg can still read camera motion.
  assert.ok(passBlockedBy({ decode: false, detect: true, encode: true }, 'camera'))
  assert.equal(passBlockedBy({ decode: true, detect: true, encode: false }, 'camera'), null)
})

test('the one cut this pass feeds names its flag and its direction', () => {
  const row = thresholdFields().find((f) => f.key === 'camera_shake_max')
  assert.ok(row, 'camera_shake_max has no row in the thresholds panel')
  assert.equal(row.flag, 'shaky')
  assert.equal(row.direction, 'above')
  assert.ok(row.flag in FLAG_LABELS)
})

test('the cut hint carries the scale and says it is not the handheld label', () => {
  // Both are corrections to a wrong assumption a user would otherwise make and
  // never find out about: that there is no comparable scale, and that moving
  // this threshold moves the labels.
  const row = thresholdFields().find((f) => f.key === 'camera_shake_max')
  assert.match(row.hint, /0\.10/)
  assert.match(row.hint, /1\.16/)
  assert.match(row.hint, /not the same threshold/i)
  assert.match(row.hint, /no default/i)
})

test('a chip is offered for every label present and none that is absent', () => {
  const clips = [
    { camera: ['pan_right', 'handheld_shot'] },
    { camera: ['pan_right'] },
    { camera: [] },
    {},
  ]
  assert.deepEqual(cameraChips(clips),
    [{ name: 'pan_right', label: 'Pan right', count: 2, ours: false },
     { name: 'handheld_shot', label: 'Handheld', count: 1, ours: false }])
})

test('the chips come out in canonical order, not by count', () => {
  const clips = [{ camera: ['handheld_shot'] }, { camera: ['handheld_shot'] },
                 { camera: ['pan_left'] }]
  assert.deepEqual(cameraChips(clips).map((c) => c.name),
    ['pan_left', 'handheld_shot'])
})

test('our own labels are flagged as ours on the chip', () => {
  assert.equal(cameraChips([{ camera: ['slideshow'] }])[0].ours, true)
  assert.equal(cameraChips([{ camera: ['pan_left'] }])[0].ours, false)
})

test('filtering keeps a shot that carries the label among others', () => {
  // A membership test and not an equality one: a handheld pan that also zooms
  // must survive a filter on any one of its three labels, or two filters would
  // silently exclude each other.
  const clips = [
    { id: 1, camera: ['pan_right', 'zoom_in', 'handheld_shot'] },
    { id: 2, camera: ['static_shot'] },
  ]
  assert.deepEqual(filterByCamera(clips, 'zoom_in').map((c) => c.id), [1])
  assert.deepEqual(filterByCamera(clips, 'handheld_shot').map((c) => c.id), [1])
  assert.deepEqual(filterByCamera(clips, 'static_shot').map((c) => c.id), [2])
})

test('no filter shows everything, including shots with no reading', () => {
  const clips = [{ id: 1, camera: ['pan_left'] }, { id: 2 }]
  assert.equal(filterByCamera(clips, null).length, 2)
})

test('a shot with no reading gets an empty badge and not a placeholder', () => {
  // An empty badge and a badge reading "unknown" send a user to two different
  // places, and only one of them is right.
  assert.equal(cameraBadge({}), '')
  assert.equal(cameraBadge({ camera: [] }), '')
  assert.equal(cameraBadge(null), '')
  assert.equal(cameraBadge({ camera: ['pan_right', 'zoom_in'] }), 'Pan right · Zoom in')
})

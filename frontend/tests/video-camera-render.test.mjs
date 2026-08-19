/**
 * 🎥 The camera facet, RENDERED — what a reader actually sees on a thumbnail.
 *
 * The contract test next door pins the vocabulary and the helpers; this one
 * executes the grid, because two of the things that matter here are properties
 * of the MARKUP and of nothing else:
 *
 *  · THE CAMERA BADGE AND THE FLAG BADGE MUST NOT BE THE SAME COLOUR, and they
 *    must not sit in the same corner. Amber in this grid means "a cut flagged
 *    this"; a pan is not a fault, and a camera label wearing amber would read as
 *    an accusation on every shot in the bank. A helper test cannot see a class
 *    name, and a screenshot cannot be asserted.
 *
 *  · A SHOT WITH NO READING MUST RENDER NOTHING AT ALL, not an empty badge and
 *    not a placeholder. Every bank is in that state until the pass has run once,
 *    so it is the state most users see first.
 *
 * ⚠️ This file MOUNTS components, so it needs `frontend/node_modules`. A git
 * worktree of this repo does not have one — there it fails with `Cannot find
 * package 'react'`, which is a missing dependency and not a regression.
 */
import test from 'node:test'
import assert from 'node:assert/strict'

import { render } from './support/mountJsx.mjs'

const { default: VideoClipGrid } =
  await import('../src/components/videobank/VideoClipGrid.jsx')

const base = {
  source_id: 7, relpath: 'day1/a.mp4', start_s: 0, end_s: 5, duration_s: 5,
  thumb_state: 'ok', status: 'pending', promoted_dataset_id: null,
}
const grid = (clips) => render(VideoClipGrid, {
  bankId: 4, clips, selected: [], onToggle: () => {}, onOpen: () => {},
  emptyMessage: 'nothing',
})

test('a single camera label is spelled out on the thumbnail', () => {
  const html = grid([{ ...base, id: 1, camera: ['pan_right'] }])
  assert.match(html, /Pan right/)
})

test('several labels collapse to a count, with all of them in the tooltip', () => {
  // The same shape the flag badge uses, for the same reason: three words do not
  // fit on a thumbnail, and the tooltip is where the detail belongs.
  const html = grid([{ ...base, id: 1, camera: ['pan_right', 'zoom_in', 'handheld_shot'] }])
  assert.match(html, /title="Pan right · Zoom in · Handheld"/)
  assert.doesNotMatch(html, />Pan right</)
})

test('a shot with no reading renders no camera badge at all', () => {
  for (const clip of [{ ...base, id: 1 }, { ...base, id: 2, camera: [] }]) {
    const html = grid([clip])
    assert.doesNotMatch(html, /🎥/, 'an unread shot must show nothing, not an empty badge')
  }
})

test('the camera badge is not amber and does not sit where the flags do', () => {
  // Both halves matter. Amber is this grid's word for "flagged", and the flag
  // badge is bottom-LEFT — a camera label in the same corner would overlap it
  // on every shot that has both.
  const html = grid([{ ...base, id: 1, camera: ['pan_right'], flags: ['shaky'] }])
  const badge = html.match(/<span[^>]*title="Pan right"[^>]*>/)
  assert.ok(badge, 'the camera badge was not rendered')
  assert.doesNotMatch(badge[0], /amber/, 'the camera badge must not wear the flag colour')
  assert.match(badge[0], /bottom-1 right-1/, 'it must not sit on the flag badge')
})

test('a shot carrying both a flag and a camera label shows both', () => {
  const html = grid([{ ...base, id: 1, camera: ['handheld_shot'], flags: ['shaky'] }])
  assert.match(html, /Camera shake/)     // the ⚑ flag, amber, bottom-left
  assert.match(html, /Handheld/)         // the 🎥 label, slate, bottom-right
})

test('the grid still holds no <video> once camera labels are on it', () => {
  // The lane's load-bearing constraint, re-checked in this state because the
  // badge is new markup inside the same tile.
  const html = grid([{ ...base, id: 1, camera: ['pan_right'] }])
  assert.equal((html.match(/<video[\s>]/g) || []).length, 0)
})

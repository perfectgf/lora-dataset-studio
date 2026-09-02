import test from 'node:test'
import assert from 'node:assert/strict'
import { appendImages, datasetClips, galleryPage } from './videoPickerFeeds.js'

/* The payload below is the shape the running server answers, keys copied from
   a live GET /api/video-dataset/<id>: `clips` is the COUNT and `items` is the
   list. Reading the plausible name instead of the measured one is what took
   the page down, so the count stays in this fixture — a test written against
   a tidied-up payload would pass on the broken code. */
const DATASET = {
  id: 1, name: 'a training set', clips: 3, fps: 24, frames: 56,
  items: [
    { id: 11, filename: 'a.mp4', caption: 'one', start_s: 0, end_s: 2 },
    { id: 12, filename: 'b.mp4', caption: 'two', start_s: 2, end_s: 4 },
    { id: 13, filename: 'c.mp4', caption: 'three', start_s: 4, end_s: 6 },
  ],
}

test('the clip list is items, and the count is never mistaken for it', () => {
  const clips = datasetClips(DATASET)
  assert.equal(clips.length, 3)
  assert.deepEqual(clips.map((c) => c.filename), ['a.mp4', 'b.mp4', 'c.mp4'])
  // The bug, pinned: `clips` is a number, and a renderer that got it crashed.
  assert.equal(typeof DATASET.clips, 'number')
  assert.ok(Array.isArray(datasetClips(DATASET)))
})

test('anything that is not a list of clips is an empty list, not a crash', () => {
  for (const payload of [null, undefined, {}, { items: 7 }, { items: null }, { clips: 21 }]) {
    const out = datasetClips(payload)
    assert.ok(Array.isArray(out), `not an array for ${JSON.stringify(payload)}`)
    assert.equal(out.length, 0)
  }
})

test('a gallery page says where the next one starts', () => {
  const page = galleryPage({ images: [{ id: 9 }, { id: 7 }], has_more: true, next_before_id: 7 })
  assert.deepEqual(page.images.map((i) => i.id), [9, 7])
  assert.equal(page.before, 7)
  assert.equal(page.more, true)
  // Cursor missing: the oldest id on the page is the same answer.
  assert.equal(galleryPage({ images: [{ id: 9 }, { id: 4 }], has_more: true }).before, 4)
  // Last page, and an install that never generated anything.
  assert.equal(galleryPage({ images: [{ id: 3 }], has_more: false }).more, false)
  const empty = galleryPage({ images: [], has_more: true })
  assert.equal(empty.more, false, 'an empty page ends the walk instead of looping')
  assert.equal(galleryPage(null).images.length, 0)
})

test('paging appends without ever showing a picture twice', () => {
  const first = [{ id: 9 }, { id: 8 }, { id: 7 }]
  // A picture generated while the picker is open shifts the window, so id 7
  // comes back in the next page — React keys must stay unique.
  const merged = appendImages(first, [{ id: 7 }, { id: 6 }])
  assert.deepEqual(merged.map((i) => i.id), [9, 8, 7, 6])
  assert.deepEqual(appendImages(first, []).map((i) => i.id), [9, 8, 7])
  assert.deepEqual(appendImages(null, [{ id: 1 }]).map((i) => i.id), [1])
  assert.deepEqual(appendImages(first, null).map((i) => i.id), [9, 8, 7])
})

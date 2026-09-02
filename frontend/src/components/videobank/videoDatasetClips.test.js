import test from 'node:test'
import assert from 'node:assert/strict'

import {
  isStillFile, datasetClipPoster, clipDurationS, hasCaption, filterClipsByCaption, clipFilterCount,
  searchClips, sortClips, visibleClips, clipCounts, captionCoverageNote,
  removeClipsConfirmation, removeClipsReport, lightboxTargets, lightboxKeyAction, purgeDraft,
  CLIP_FILTERS, CLIP_SORTS,
  normalizeClipFilter, normalizeClipSort,
} from './videoDatasetClips.js'

const clip = (over = {}) => ({
  id: 1, filename: 'clip_0001.mp4', caption: null,
  src_relpath: 'rushes/a.mp4', start_s: 0, end_s: 3, ...over,
})

// ---- stills are not videos, and the difference is a dead player ---------------

test('a stills set is recognised by its extension, in either case', () => {
  assert.equal(isStillFile('clip_0001.png'), true)
  assert.equal(isStillFile('CLIP_0002.JPEG'), true)
  assert.equal(isStillFile('clip_0003.webp'), true)
  assert.equal(isStillFile('clip_0004.mp4'), false)
  // The footgun: a name that merely CONTAINS an image extension is a video.
  assert.equal(isStillFile('a.png.mp4'), false)
  assert.equal(isStillFile(null), false)
})

test('a still has no duration - it has no bounds to have one from', () => {
  assert.equal(clipDurationS(clip({ start_s: 2, end_s: 5 })), 3)
  assert.equal(clipDurationS(clip({ start_s: 2, end_s: 2 })), null)
  assert.equal(clipDurationS(clip({ start_s: null, end_s: null })), null)
})

// ---- the caption filter, which is what the grid is worked through ------------

test('whitespace is not a caption', () => {
  assert.equal(hasCaption(clip({ caption: '   ' })), false)
  assert.equal(hasCaption(clip({ caption: null })), false)
  assert.equal(hasCaption(clip({ caption: 'a woman walks' })), true)
})

test('the three filters partition the set - captioned + uncaptioned = all', () => {
  const clips = [clip({ id: 1, caption: 'x' }), clip({ id: 2 }), clip({ id: 3, caption: ' ' })]
  assert.equal(clipFilterCount(clips, 'all'), 3)
  assert.equal(clipFilterCount(clips, 'captioned'), 1)
  assert.equal(clipFilterCount(clips, 'uncaptioned'), 2)
  assert.equal(clipFilterCount(clips, 'captioned') + clipFilterCount(clips, 'uncaptioned'),
    clipFilterCount(clips, 'all'))
  // Every id the UI can offer must be one this function really answers.
  for (const f of CLIP_FILTERS) assert.equal(normalizeClipFilter(f.id), f.id)
  assert.equal(normalizeClipFilter('nonsense'), 'all')
  assert.deepEqual(filterClipsByCaption(clips, 'nonsense'), clips)
})

// ---- search: two rules, and the exclusion is the one that gets dropped -------

test('terms are ANDed across the file name, the caption and the source path', () => {
  const clips = [
    clip({ id: 1, filename: 'clip_0001.mp4', caption: 'a woman on a beach', src_relpath: 'trip/sea.mp4' }),
    clip({ id: 2, filename: 'clip_0002.mp4', caption: 'a man in a car', src_relpath: 'city/road.mp4' }),
  ]
  assert.deepEqual(searchClips(clips, 'woman').map((c) => c.id), [1])
  assert.deepEqual(searchClips(clips, 'sea').map((c) => c.id), [1])
  assert.deepEqual(searchClips(clips, 'clip_0002').map((c) => c.id), [2])
  // AND, not OR: a clip must satisfy every term.
  assert.deepEqual(searchClips(clips, 'woman car').map((c) => c.id), [])
  assert.deepEqual(searchClips(clips, 'a woman').map((c) => c.id), [1])
})

test('a -term excludes, and a bare dash is not a term', () => {
  const clips = [clip({ id: 1, caption: 'a woman on a beach' }), clip({ id: 2, caption: 'a man in a car' })]
  assert.deepEqual(searchClips(clips, '-woman').map((c) => c.id), [2])
  assert.deepEqual(searchClips(clips, 'a -woman').map((c) => c.id), [2])
  // A lone dash would otherwise exclude the empty string, i.e. everything.
  assert.deepEqual(searchClips(clips, '-').map((c) => c.id), [1, 2])
  assert.deepEqual(searchClips(clips, '').map((c) => c.id), [1, 2])
})

// ---- sort: the default is the order the trainer reads the folder in ----------

test('every offered sort really orders, and the default is the file order', () => {
  const clips = [
    clip({ id: 3, filename: 'clip_0003.mp4', start_s: 0, end_s: 1, caption: 'c' }),
    clip({ id: 1, filename: 'clip_0001.mp4', start_s: 0, end_s: 5 }),
    clip({ id: 2, filename: 'clip_0002.mp4', start_s: 0, end_s: 3, caption: 'b' }),
  ]
  // Exercised, not asserted by count: each id in CLIP_SORTS is really run.
  const seen = {}
  for (const s of CLIP_SORTS) {
    assert.equal(normalizeClipSort(s.id), s.id)
    seen[s.id] = sortClips(clips, s.id).map((c) => c.id)
  }
  assert.deepEqual(seen.filename, [1, 2, 3])
  assert.deepEqual(seen.longest, [1, 2, 3])
  assert.deepEqual(seen.shortest, [3, 2, 1])
  assert.deepEqual(seen['caption-first'], [1, 2, 3])
  assert.equal(Object.keys(seen).length, CLIP_SORTS.length)
  assert.deepEqual(sortClips(clips, 'nonsense').map((c) => c.id), [1, 2, 3])
})

test('sorting does not mutate the list it was given', () => {
  const clips = [clip({ id: 2, filename: 'b.mp4' }), clip({ id: 1, filename: 'a.mp4' })]
  sortClips(clips, 'filename')
  assert.deepEqual(clips.map((c) => c.id), [2, 1])
})

test('the three stages compose in the order the screen applies them', () => {
  const clips = [
    clip({ id: 1, filename: 'clip_0001.mp4', caption: 'a woman walks', start_s: 0, end_s: 2 }),
    clip({ id: 2, filename: 'clip_0002.mp4', caption: null, start_s: 0, end_s: 9 }),
    clip({ id: 3, filename: 'clip_0003.mp4', caption: 'a woman runs', start_s: 0, end_s: 5 }),
  ]
  assert.deepEqual(
    visibleClips(clips, { query: 'woman', filter: 'captioned', sort: 'longest' })
      .map((c) => c.id),
    [3, 1])
})

// ---- the counts under the grid ----------------------------------------------

test('the counts add up and the seconds are the summed bounds', () => {
  const clips = [
    clip({ id: 1, caption: 'x', start_s: 0, end_s: 2.5 }),
    clip({ id: 2, start_s: 0, end_s: 3 }),
    clip({ id: 3, filename: 'clip_0003.png', start_s: 0, end_s: 0 }),
  ]
  const c = clipCounts(clips)
  assert.deepEqual(c, { total: 3, captioned: 1, uncaptioned: 2, stills: 1, seconds: 5.5 })
  assert.deepEqual(clipCounts(null), { total: 0, captioned: 0, uncaptioned: 0, stills: 0, seconds: 0 })
})

// ---- the coverage line REPORTS; it must not invent a failure -----------------

test('an uncaptioned clip trains on the trigger, and the line says which one', () => {
  const note = captionCoverageNote({ total: 4, captioned: 1, uncaptioned: 3 }, 'sks_woman')
  assert.match(note, /1 of 4/)
  assert.match(note, /trigger word alone/)
  assert.match(note, /sks_woman/)
  // No trigger is a DIFFERENT outcome (an empty .txt) and must not wear the
  // same sentence.
  const none = captionCoverageNote({ total: 4, captioned: 1, uncaptioned: 3 }, '')
  assert.match(none, /EMPTY \.txt/)
  assert.doesNotMatch(none, /trigger word alone/)
})

test('a fully captioned set says so instead of counting a zero', () => {
  assert.equal(captionCoverageNote({ total: 3, captioned: 3, uncaptioned: 0 }, 't'),
    'All 3 clips carry a caption.')
  assert.equal(captionCoverageNote({ total: 0 }, 't'), 'No clip in this dataset yet.')
})

// ---- the destructive confirm names what survives -----------------------------

test('the removal confirm names the count and promises the bank is untouched', () => {
  assert.equal(removeClipsConfirmation([]), null)
  const one = removeClipsConfirmation(['clip_0001.mp4'], { mode: 'app_trash' })
  assert.match(one, /clip_0001\.mp4/)
  const many = removeClipsConfirmation(['a.mp4', 'b.mp4', 'c.mp4'], { mode: 'app_trash' })
  assert.match(many, /Remove 3 clips/)
  for (const text of [one, many]) {
    assert.match(text, /\.txt/)
    assert.match(text, /bank .* keeps every shot/)
    // The destination comes from the app-wide wording, named from the mode the
    // server reports — never a sentence typed here (that sentence was false).
    assert.match(text, /the app's Trash \(Settings ▸ Storage\)/)
    assert.match(text, /recoverable until you empty it/)
  }
})

test('the destination is the MODE, not a belief — and an unknown mode promises nothing', () => {
  // The first version hardcoded "the app's Trash — recoverable from Settings" in
  // three places while a default install sends files elsewhere. The mode decides.
  assert.match(removeClipsConfirmation(['a.mp4'], { mode: 'trash' }), /your system Recycle Bin/)
  const gone = removeClipsConfirmation(['a.mp4'], { mode: 'delete' })
  assert.match(gone, /deleted for good/)
  assert.doesNotMatch(gone, /recoverable/)
  // No mode at all (an old server, a missing field) must not invent recovery.
  assert.doesNotMatch(removeClipsConfirmation(['a.mp4']), /recoverable/)
})

test('a STILLS set is not promised a bank it does not have', () => {
  // Its rows are written straight from an image dataset, with no source_bank_id
  // and no source_clip_id — "you can promote them again" would be a promise
  // about something that does not exist.
  const stills = removeClipsConfirmation(['clip_0001.png'], { fromBank: false, mode: 'app_trash' })
  assert.doesNotMatch(stills, /keeps every shot/,
    'a stills set has no bank to keep anything')
  assert.doesNotMatch(stills, /promote them again without/)
  assert.match(stills, /no bank/)
  assert.match(stills, /image dataset/)
  assert.match(stills, /the app's Trash/)
})

// ---- the player: two lists, and the reason they are two ----------------------

test('the open clip survives leaving the filter — the player does not close on success', () => {
  // The page's headline workflow: filter "No caption", open a clip, write one.
  // The instant it is written the clip leaves the filter. Resolving the open
  // clip against the filtered list closed the player at exactly that moment —
  // measured in a real browser, and the whole reason these are two lists.
  const items = [clip({ id: 1, caption: 'now captioned' }), clip({ id: 2 }), clip({ id: 3 })]
  const shown = filterClipsByCaption(items, 'uncaptioned')      // ids 2 and 3
  const t = lightboxTargets(items, shown, 1, 0)                 // it held slot 0
  assert.equal(t.clip?.id, 1, 'the player must keep showing the clip just captioned')
  assert.equal(t.index, -1)
  // The walk RESUMES from its old slot: nothing before it, and the clip that
  // took its place is next. Both arrows dead here used to strand the workflow.
  assert.equal(t.prevId, null)
  assert.equal(t.nextId, 2)
})

test('stepping walks the FILTERED list, and stops at both ends', () => {
  const items = [clip({ id: 1 }), clip({ id: 2 }), clip({ id: 3 }), clip({ id: 4 })]
  const shown = [items[0], items[2]]                    // a filter kept 1 and 3
  const first = lightboxTargets(items, shown, 1)
  assert.deepEqual([first.prevId, first.nextId], [null, 3], 'next skips the filtered-out clip')
  const last = lightboxTargets(items, shown, 3)
  assert.deepEqual([last.prevId, last.nextId], [1, null])
  // An id nobody holds is not a crash — and with no clip there is no player,
  // so what the arrows would do is moot; only the two facts that matter are pinned.
  const none = lightboxTargets(items, shown, 99)
  assert.deepEqual([none.clip, none.index], [null, -1])
  assert.deepEqual(lightboxTargets(null, null, 1).clip, null)
})

test('the removal report stops saying "removed" about a file still in the folder', () => {
  assert.equal(removeClipsReport({ removed: 3, files_kept: 0, delete_mode: 'app_trash' }),
    "3 clips removed — sent to the app's Trash (Settings ▸ Storage).")
  assert.equal(removeClipsReport({ removed: 1, files_kept: 0, delete_mode: 'trash' }),
    '1 clip removed — sent to your system Recycle Bin.')
  // A no-op is not a success: "0 clips removed" in green was a lie about nothing.
  assert.equal(removeClipsReport({ removed: 0, files_kept: 0 }), 'Nothing was removed — no clip matched.')
  assert.equal(removeClipsReport({}), 'Nothing was removed — no clip matched.')
  // The dangerous case: the row is gone from nothing, the file is still read by
  // the trainer. It must NOT wear the success wording.
  const partial = removeClipsReport({ removed: 2, files_kept: 1 })
  assert.match(partial, /2 clips removed/)
  assert.match(partial, /could not be moved/)
  assert.match(partial, /the trainer reads the folder/)
  const none = removeClipsReport({ removed: 0, files_kept: 2 })
  assert.match(none, /Nothing was removed/)
  assert.match(none, /2 files/)
  assert.deepEqual(typeof removeClipsReport({}), 'string')
})

// ---- the draft purge: the user's typing wins over the bookkeeping -------------

test('a draft is dropped only if it still holds the value that was posted', () => {
  // Typed on during the save: the draft is no longer what went to the server,
  // and dropping it would throw the new text away in silence.
  const typedOn = { 7: 'abc', 8: 'other' }
  assert.equal(purgeDraft(typedOn, 7, 'ab'), typedOn, 'the same object comes back, untouched')
  // Unchanged since the post: the purge is legitimate, and the neighbour stays.
  assert.deepEqual(purgeDraft({ 7: 'ab', 8: 'other' }, 7, 'ab'), { 8: 'other' })
  // No draft for that id at all: nothing to do, nothing to break.
  const none = { 8: 'other' }
  assert.equal(purgeDraft(none, 7, 'ab'), none)
  assert.deepEqual(purgeDraft(undefined, 7, 'ab'), {})
})

// ---- the keyboard grammar of the player, as values --------------------------

test('Escape saves-and-closes whether or not the caret is in the caption box', () => {
  // The case that lost captions: Escape WHILE typing. It is the only case where
  // there is a caption to lose, and it must not be the one that skips the save.
  assert.equal(lightboxKeyAction('Escape', { typing: true }), 'save-close')
  assert.equal(lightboxKeyAction('Escape', { typing: false }), 'save-close')
  assert.equal(lightboxKeyAction('Escape', {}), 'save-close')
})

test('the arrows belong to the caret while typing, and to the player otherwise', () => {
  const free = { typing: false, hasPrev: true, hasNext: true }
  assert.equal(lightboxKeyAction('ArrowLeft', free), 'prev')
  assert.equal(lightboxKeyAction('ArrowRight', free), 'next')
  const typing = { typing: true, hasPrev: true, hasNext: true }
  assert.equal(lightboxKeyAction('ArrowLeft', typing), null)
  assert.equal(lightboxKeyAction('ArrowRight', typing), null)
  // At either end the arrow does nothing rather than wrapping.
  assert.equal(lightboxKeyAction('ArrowLeft', { hasPrev: false, hasNext: true }), null)
  assert.equal(lightboxKeyAction('ArrowRight', { hasPrev: true, hasNext: false }), null)
  assert.equal(lightboxKeyAction('Enter', free), null)
})

// ---- the player resumes from the slot the clip held, once it leaves the filter

test('when the open clip leaves the filtered list, the walk resumes from its old slot', () => {
  // Filter "No caption", clip 2 (slot 1 of [1,2,3]) just got a caption: shown is
  // now [1,3]. Next must be the clip that took its slot (3), prev the one before
  // (1) — instead of both arrows dying, which stranded the page's own workflow.
  const items = [clip({ id: 1 }), clip({ id: 2, caption: 'done' }), clip({ id: 3 })]
  const shown = filterClipsByCaption(items, 'uncaptioned')      // [1, 3]
  const t = lightboxTargets(items, shown, 2, 1)
  assert.equal(t.clip?.id, 2)
  assert.equal(t.index, -1)
  assert.deepEqual([t.prevId, t.nextId], [1, 3])
  // It was first: nothing before it, the new first is next.
  assert.deepEqual((({ prevId, nextId }) => [prevId, nextId])(lightboxTargets(items, shown, 2, 0)), [null, 1])
  // It was last: the last remaining is prev, nothing after.
  assert.deepEqual((({ prevId, nextId }) => [prevId, nextId])(lightboxTargets(items, shown, 2, 5)), [3, null])
  // No memory of a slot at all (-1): behaves as "it was first".
  assert.deepEqual((({ prevId, nextId }) => [prevId, nextId])(lightboxTargets(items, shown, 2)), [null, 1])
})

test('a clip has one poster, the same on the training set page and in the studio picker', () => {
  // Keys as the live detail answers them: a stills set is served by the media
  // route; a clip cut from a bank borrows the bank's thumbnail; a clip
  // imported from disk has no picture on the server.
  const still = { id: 4, filename: 'portrait_004.webp', source_bank_id: null, source_clip_id: null }
  const cut = { id: 5, filename: 'walk_012.mp4', source_bank_id: 3, source_clip_id: 88 }
  const imported = { id: 6, filename: 'walk_013.mp4', source_bank_id: null, source_clip_id: null }
  assert.equal(datasetClipPoster(1, still), '/api/video-dataset/1/clip/4/media')
  assert.equal(datasetClipPoster(1, cut), '/api/video-bank/3/clip/88/thumb')
  assert.equal(datasetClipPoster(1, imported), null)
  // A still keeps its own frame even when it was cut from a bank: the media
  // route serves the real picture, the bank thumb is a JPEG of it.
  assert.equal(datasetClipPoster(1, { ...cut, filename: 'a.PNG' }), '/api/video-dataset/1/clip/5/media')
  // Half a provenance is no provenance, whichever half is missing — a URL
  // with `null` in it 404s loudly.
  assert.equal(datasetClipPoster(1, { ...cut, source_clip_id: null }), null)
  assert.equal(datasetClipPoster(1, { ...cut, source_bank_id: null }), null)
  assert.equal(datasetClipPoster(1, null), null)
  assert.equal(datasetClipPoster(null, still), null)
})

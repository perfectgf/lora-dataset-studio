/**
 * 🎬 The video lane, RENDERED — the constraints that only a real render can pin.
 *
 * Three of the four assertions here cannot be made any other way:
 *
 *  · THE GRID CONTAINS NO <video>. That is the load-bearing design decision of
 *    the whole lane, and the "nicer" version someone will eventually write — a
 *    muted <video preload="none"> per tile, played on hover — is broken in two
 *    ways that never show up in review. There is no clip FILE to hover (a bank
 *    stores bounds and encodes nothing until promotion), and Chrome caps
 *    WebMediaPlayers at roughly sixty across the browser: past that, new
 *    elements simply never load, with no error of any kind. A bank holds
 *    hundreds of shots, so the failure would land on someone else's second
 *    screen of scroll. Counting the elements in the rendered markup is the only
 *    check that survives a rewrite of the component.
 *
 *  · THE LIGHTBOX CONTAINS EXACTLY ONE, pointed at the SOURCE with a media
 *    fragment. A missing `#t=` does not throw and does not warn — the browser
 *    ignores it and plays the whole file, which on a two-hour rush is the worst
 *    outcome available.
 *
 *  · THE TARGET PICKER SHOWS `licence_note` AND `training_verified`. Both are
 *    what stop a wasted week, and both are the kind of label a tidy-up quietly
 *    removes. The strings are asserted against the real catalogue's own wording.
 *
 * ⚠️ This file MOUNTS components, so it needs `frontend/node_modules` (react +
 * esbuild). Git worktrees of this repo do not have one — there it fails with
 * `Cannot find package 'react'`, exactly like the canvas render tests, and that
 * is a missing dependency rather than a regression.
 */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

import { createElement, render, renderToStaticMarkup } from './support/mountJsx.mjs'

/* ⚠️ Dynamic, and it has to be: the hooks that teach Node to read .jsx are
   installed while mountJsx.mjs is EVALUATED, and a static import of a .jsx file
   would already have been loaded by then — the whole graph is linked before the
   first line of any module runs. */
const { default: VideoClipGrid } =
  await import('../src/components/videobank/VideoClipGrid.jsx')
const { default: VideoClipLightbox } =
  await import('../src/components/videobank/VideoClipLightbox.jsx')
const { default: VideoCapabilityStrip } =
  await import('../src/components/videobank/VideoCapabilityStrip.jsx')
const { default: VideoSourceList } =
  await import('../src/components/videobank/VideoSourceList.jsx')
const { default: VideoTargetPicker } =
  await import('../src/components/videobank/VideoTargetPicker.jsx')

const { ToastProvider } = await import('../src/components/common/Toast.jsx')

/* The lightbox now holds the ✂ retouch tools, which call `useToast` — a hook that
   THROWS outside its provider rather than degrading, on purpose. Rendering the
   host inside the provider is what the app does; stubbing the hook here would
   have made this file green while the real lightbox threw on open. */
const renderLightbox = (props) => renderToStaticMarkup(
  createElement(ToastProvider, null, createElement(VideoClipLightbox, props)))

const read = readSource
const countTag = (html, tag) => (html.match(new RegExp(`<${tag}[\\s>]`, 'g')) || []).length

const CLIPS = [
  { id: 1, source_id: 7, relpath: 'day1/a.mp4', start_s: 0, end_s: 8, duration_s: 8, thumb_state: 'ok', status: 'pending', promoted_dataset_id: null },
  { id: 2, source_id: 7, relpath: 'day1/a.mp4', start_s: 41.25, end_s: 46.5, duration_s: 5.25, thumb_state: 'ok', status: 'keep', promoted_dataset_id: null },
  // thumb_state null on purpose: the pass has not run, which is an ordinary
  // state and must render a placeholder rather than a broken image or a player.
  { id: 3, source_id: 9, relpath: 'day2/b.mkv', start_s: 3, end_s: 9, duration_s: 6, thumb_state: null, status: 'reject', promoted_dataset_id: null },
  { id: 4, source_id: 9, relpath: 'day2/b.mkv', start_s: 12, end_s: 17, duration_s: 5, thumb_state: 'ok', status: 'keep', promoted_dataset_id: 3 },
]

// ---- (1) the grid holds no player, in ANY state -----------------------------

test('the clip grid renders NOT ONE <video>', () => {
  const html = render(VideoClipGrid, {
    bankId: 4, clips: CLIPS, selected: [2], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'nothing',
  })
  assert.equal(countTag(html, 'video'), 0, 'the grid must contain zero <video> elements')
  assert.ok(!html.includes('<source'), 'no <source> either — that is a player by another name')
})

test('every thumbnail in the grid loads lazily', () => {
  const html = render(VideoClipGrid, {
    bankId: 4, clips: CLIPS, selected: [], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'nothing',
  })
  const imgs = html.match(/<img[^>]*>/g) || []
  assert.equal(imgs.length, 3, 'one <img> per clip WITH a thumbnail, and no more')
  for (const img of imgs) {
    assert.match(img, /loading="lazy"/, `thumbnail not lazy: ${img}`)
    assert.match(img, /\/api\/video-bank\/4\/clip\/\d+\/thumb/)
  }
})

test('a clip with no thumbnail yet draws a placeholder, not a broken image', () => {
  const html = render(VideoClipGrid, {
    bankId: 4, clips: [CLIPS[2]], selected: [], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'nothing',
  })
  assert.equal((html.match(/<img/g) || []).length, 0)
  assert.ok(html.includes('🎞'))
})

test('the grid source file declares no video element in any branch', () => {
  // The mount only covers the states this file passes it. A branch nobody
  // rendered is how `dropHint === "leaving"` shipped broken on the canvas — so
  // the source is checked too, comments stripped: the file's own docstring
  // quotes the `<video preload="none">` version precisely to rule it out, and
  // that sentence must stay allowed to exist.
  const src = read('src/components/videobank/VideoClipGrid.jsx')
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '')
  assert.ok(!/<video[\s>]/.test(src), 'VideoClipGrid.jsx must not render a <video> tag')
})

test('an empty grid renders its explanation, not an empty box', () => {
  const html = render(VideoClipGrid, {
    bankId: 4, clips: [], selected: [], onToggle: () => {}, onOpen: () => {},
    emptyMessage: 'No kept shot in this bank.',
  })
  assert.ok(html.includes('No kept shot in this bank.'))
})

// ---- (2) the lightbox: exactly one player, with the fragment ----------------

test('the lightbox mounts EXACTLY ONE <video>, pointed at the source + fragment', () => {
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], onClose: () => {}, onPrev: () => {}, onNext: () => {},
    onKeep: () => {}, onReject: () => {}, hasPrev: true, hasNext: true,
  })
  assert.equal(countTag(html, 'video'), 1)
  // The SOURCE, not a clip file — no clip file exists before promotion.
  assert.match(html, /src="\/api\/video-bank\/4\/source\/7\/media#t=41\.25,46\.5"/)
})

test('the fragment carries sub-second bounds rather than rounded seconds', () => {
  // Rounding a bound to the nearest second moves a quarter of a two-second clip.
  const html = renderLightbox({
    bankId: 1, clip: { ...CLIPS[1], start_s: 41.25, end_s: 43.5 },
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.match(html, /#t=41\.25,43\.5/)
})

test('a zero-length shot gets no player at all rather than a whole-file one', () => {
  // clipFragmentSrc answers null on an unplayable range; handing a malformed
  // fragment to a <video> makes it play the ENTIRE rush, silently.
  const html = renderLightbox({
    bankId: 1, clip: { ...CLIPS[0], start_s: 5, end_s: 5 },
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.equal(countTag(html, 'video'), 0)
  assert.match(html, /no playable range/)
})

test('the lightbox renders its position and keeps its triage keys visible', () => {
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], onClose: () => {}, hasPrev: false, hasNext: true,
  })
  assert.ok(html.includes('0:41 – 0:46 (5.3s)'), 'the shot’s position in its source')
  assert.match(html, /K to keep/)
  assert.match(html, /R to reject/)
})

// ---- (2b) the retouch tools live UNDER that one player ----------------------
//
// The tools are the reason the lightbox is now mounted inside a ToastProvider.
// Two things are worth a render rather than a source regex: the panel must not
// have added a second player (the ceiling is one, by construction), and the i2v
// line must actually reach the screen — it is the piece of knowledge the feature
// was built around, and it is exactly the kind of line a tidy-up deletes.

test('the retouch tools add NO second player', () => {
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], source: { id: 7, duration_s: 120, fps_native: 25 },
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.equal(countTag(html, 'video'), 1, 'the lane holds ONE player, tools included')
  assert.match(html, /Trim &amp; split this shot/)
})

test('the trim panel says the first frame is the i2v conditioning image', () => {
  // ai-toolkit conditions an image-to-video sample on the clip's FIRST frame, so
  // moving a start IS choosing the conditioning image. Nothing else in the app
  // says it and no user would guess it from a control called "trim".
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], source: { id: 7, duration_s: 120, fps_native: 25 },
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.match(html, /first frame is the conditioning image/)
})

test('the nudge buttons name one frame at the SOURCE rate, not the target rate', () => {
  // Reading the target's fps here is the same mistake that turns a 16 fps profile
  // into accelerated motion: bounds are timestamps in the source file.
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], source: { id: 7, duration_s: 120, fps_native: 25 },
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.match(html, /one frame = 0\.040s/)
})

test('a shot whose source was never probed still renders its tools', () => {
  // duration_s and fps_native are both null before the probe pass. The panel has
  // to degrade to a conservative frame step and let the server own the upper wall
  // — throwing here would make the lightbox unopenable on a fresh bank.
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], source: { id: 7, duration_s: null, fps_native: null },
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.match(html, /one frame = 0\.033s/)
})

test('with no playhead yet, splitting says what to do instead of failing quietly', () => {
  // The very first render happens before any timeupdate has fired.
  const html = renderLightbox({
    bankId: 4, clip: CLIPS[1], source: null,
    onClose: () => {}, hasPrev: false, hasNext: false,
  })
  assert.match(html, /Move the playhead inside this shot/)
})

test('a probed file offers a hand cut even when it has no shots at all', () => {
  // The reachability hole this closes: every other retouch gesture needs an OPEN
  // shot, and a file detection missed — or a bank on an install with no detector,
  // which the app says can still "scan, cut, watch and triage" — has none.
  const html = render(VideoSourceList, {
    sources: [{ id: 7, relpath: 'day1/a.mp4', duration_s: 120, fps_native: 25,
      file_size: 400, probe_state: 'ok', detect_state: null, clips: 0 }],
    activeSourceId: null, onFilter: () => {}, onCut: () => {},
  })
  assert.match(html, /Cut a shot by hand/)
})

test('an UNPROBED file offers no hand cut — its length is not known yet', () => {
  // A bound cannot be checked against a duration nobody has read, and offering the
  // button there would produce a shot that only the server could refuse.
  const html = render(VideoSourceList, {
    sources: [{ id: 8, relpath: 'day1/b.mkv', duration_s: null, fps_native: null,
      file_size: 400, probe_state: null, detect_state: null, clips: 0 }],
    activeSourceId: null, onFilter: () => {}, onCut: () => {},
  })
  assert.ok(!html.includes('Cut a shot by hand'))
})

// ---- (3) the capability strip names PIECES, never one verdict ---------------

test('a healthy install renders nothing', () => {
  const html = render(VideoCapabilityStrip, {
    capability: { ok: true, detail: 'video extra ready', decode: true, detect: true, encode: true },
  })
  assert.equal(html, '')
})

test('a missing encoder names the piece, the fix, and what still works', () => {
  const html = render(VideoCapabilityStrip, {
    capability: {
      ok: false, detail: 'missing: ffmpeg (clip encoding)',
      decode: true, detect: true, encode: false,
    },
  })
  assert.ok(html.includes('Cutting clips'), 'the missing PIECE must be named')
  assert.ok(html.includes('ffmpeg'))
  assert.ok(html.includes('You can still'), 'what survives must be said')
  assert.ok(html.includes('triage'))
  // The exact sentence this lane exists to stop saying.
  assert.ok(!/video is unavailable/i.test(html))
})

test('the server’s own detail line is rendered verbatim', () => {
  const html = render(VideoCapabilityStrip, {
    capability: { ok: false, detail: 'missing: av (video decoding)', decode: false, detect: true, encode: true },
  })
  assert.ok(html.includes('missing: av (video decoding)'))
})

// ---- (4) the target picker shows the two fields that save a week -----------

const TARGETS = [
  { key: 'wan22_14b', label: 'Wan 2.1 / 2.2 14B', fps: 16, frame_choices: [17, 81], frame_default: 81, size_multiple: 16, recommended_sizes: [[832, 480]], keep_audio: false, training_verified: true, licence_note: null },
  { key: 'wan22_ti2v5b', label: 'Wan 2.2 TI2V-5B', fps: 24, frame_choices: [121], frame_default: 121, size_multiple: 32, recommended_sizes: [], keep_audio: false, training_verified: false, licence_note: null },
  { key: 'minimax_h3', label: 'MiniMax H3', fps: 24, frame_choices: [124], frame_default: 124, size_multiple: 32, recommended_sizes: [], keep_audio: true, training_verified: false, licence_note: 'MiniMax H3 Community License grants NO rights in the EU, UK, South Korea or USA — and the restriction covers the outputs, not just the model. Check your territory first.' },
]

test('every target carries its verdict ON its own row, before you pick it', () => {
  const html = render(VideoTargetPicker, { targets: TARGETS, targetKey: 'wan22_14b', onPick: () => {} })
  assert.ok(html.includes('Trainable'))
  assert.ok(html.includes('Not trainable yet'))
  assert.ok(html.includes('Licence limits'))
})

test('selecting MiniMax H3 puts the territory restriction on screen', () => {
  const html = render(VideoTargetPicker, { targets: TARGETS, targetKey: 'minimax_h3', onPick: () => {} })
  for (const territory of ['EU', 'UK', 'South Korea', 'USA']) {
    assert.ok(html.includes(territory), `${territory} must be named where the choice is made`)
  }
  assert.ok(html.includes('outputs'), 'the restriction reaching the outputs is the part people miss')
  assert.match(html, /role="alert"/)
})

test('an unverified target says a trainer may not exist', () => {
  const html = render(VideoTargetPicker, { targets: TARGETS, targetKey: 'wan22_ti2v5b', onPick: () => {} })
  assert.ok(html.includes('No LoRA trainer is known to exist'))
})

test('the verified target draws no warning box', () => {
  const html = render(VideoTargetPicker, { targets: TARGETS, targetKey: 'wan22_14b', onPick: () => {} })
  assert.ok(!html.includes('No LoRA trainer is known to exist'))
  assert.ok(!html.includes('role="alert"'))
})

// ---- (5) the source list survives a half-probed bank ------------------------

test('the file list renders every probe state without throwing', () => {
  const html = render(VideoSourceList, {
    sources: [
      { id: 1, relpath: 'day1/a.mp4', file_size: 6 * 1024 ** 3, duration_s: 5025, fps_native: 29.97, width: 1920, height: 1080, codec: 'h264', probe_state: 'ok', detect_state: 'ok', clips: 41 },
      { id: 2, relpath: 'day2/b.mkv', file_size: null, duration_s: null, fps_native: null, width: null, height: null, codec: null, probe_state: null, detect_state: null, clips: 0 },
      { id: 3, relpath: 'day2/c.avi', file_size: 12, duration_s: null, fps_native: null, width: null, height: null, codec: null, probe_state: 'unreadable', detect_state: null, clips: 0 },
    ],
    activeSourceId: 1, onFilter: () => {},
  })
  assert.ok(html.includes('1:23:45') && html.includes('6.0 GB'))
  assert.ok(html.includes('41 shots') && html.includes('Not scanned') && html.includes('Unreadable'))
})

// ---- (6) responsive: the 400 px rules, asserted as classes ------------------

test('nothing in this lane can scroll the page sideways at 400 px', () => {
  // Two rules, both learned the hard way in the image lane:
  //  · a card grid must start at grid-cols-1/2 — the IMPLICIT auto column is
  //    sized on max-content, so an unbreakable source PATH inside a card
  //    stretches it past the viewport and `truncate` never fires;
  //  · anything holding a path needs min-w-0 + truncate, because a flex child's
  //    default min-width is auto, not 0.
  const grid = read('src/components/videobank/VideoClipGrid.jsx')
  assert.match(grid, /grid-cols-2/, 'the clip grid must declare its narrow column count')
  for (const rel of [
    'src/pages/VideoBankPage.jsx',
    'src/components/videobank/VideoSourceList.jsx',
    'src/components/videobank/VideoBankWorkspace.jsx',
  ]) {
    const src = read(rel)
    assert.match(src, /min-w-0/, `${rel}: a path container needs min-w-0`)
    assert.match(src, /truncate/, `${rel}: a path needs truncate`)
    assert.ok(!/grid gap-\d+ sm:grid-cols/.test(src),
      `${rel}: a grid must declare grid-cols-1 before its sm: breakpoint`)
  }
  const page = read('src/pages/VideoBankPage.jsx')
  assert.match(page, /grid-cols-1 sm:grid-cols-2/)
})

// ---- (7) the surfaces that announce this lane exist -------------------------

test('the video bank is announced and documented', async () => {
  const { WHATS_NEW } = await import('../src/whatsNew.js')
  // The entry may have moved to the archive since it shipped — search the union.
  const { WHATS_NEW_ARCHIVE } = await import('../src/whatsNewArchive.js')
  const entry = [...WHATS_NEW, ...WHATS_NEW_ARCHIVE].find((e) => e.id === '2026-08-04-video-bank')
  assert.ok(entry, 'the What’s-new entry is what a release note is generated from')
  // Archived entries drop their in-app target by doctrine (whatsNew.js,
  // rule "Keep the list tidy") — the deep link lived while the entry was live.
  assert.equal(entry.to, undefined)
  // Benefit-first, and it names the silence it removes.
  assert.match(entry.title, /rushes/)

  const { getHelpTopic } = await import('../src/help/helpRegistry.js')
  for (const id of ['page-video-bank', 'video-bank-passes', 'video-capability-pieces',
    'video-datasets', 'video-promote-target']) {
    assert.ok(getHelpTopic(id), `missing help topic ${id}`)
  }
  assert.equal(getHelpTopic('page-video-bank').app.route, '/video-bank')
})

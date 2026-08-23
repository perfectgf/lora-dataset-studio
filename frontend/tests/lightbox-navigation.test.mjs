/**
 * MOVING THROUGH A DATASET WITHOUT CLOSING THE IMAGE.
 *
 * The lightbox had no ⟨ / ⟩: reviewing image 41 of 340 meant closing it, finding
 * tile 42 on the wall, and opening that. This pins the three decisions that make
 * such a navigation safe rather than merely present:
 *
 *  1. it walks the list the grid SHOWS (filtered, sorted), so ⟩ never lands on a
 *     picture the current filters hide;
 *  2. per-image state — zoom, the comparison pane, "improving" — does NOT travel
 *     with the move. A comparison pane captioned "original" still showing the
 *     PREVIOUS image's parent is the failure this feature could ship;
 *  3. an end of the list is a disabled button that SAYS which end, in both the
 *     title and the aria-label — not a mute no-op, and not a silent wrap.
 *
 * The UI assertions are on the RENDERED markup, not on the .jsx source: whether
 * an arrow survives the props it is computed from is exactly what a regex over
 * the file cannot answer, and the props are handed down through two components.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import { readFileSync } from 'node:fs'

import { renderToStaticMarkup, createElement } from './support/mountJsx.mjs'

const {
  freshLightboxImageState, lightboxImageState, lightboxNeighbours, stampedPatch,
} = await import('../src/components/dataset/lightboxNavigation.js')
const { ownsTypedKeys, reviewKeyAction } = await import(
  '../src/components/shared/reviewShortcuts.js')
const { pageOfIndex, GRID_PAGE_SIZE } = await import(
  '../src/components/dataset/gridPaging.js')
const { default: DatasetLightbox } = await import(
  '../src/components/dataset/DatasetLightbox.jsx')
const { CapabilitiesProvider } = await import('../src/context/CapabilitiesContext.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { datasetBusyReason } = await import(
  '../src/components/dataset/datasetBusyReason.js')

const IMAGES = [
  { id: 11, filename: 'a.png', variation_label: 'portrait', source: 'import' },
  { id: 22, filename: 'b.png', variation_label: 'bust', source: 'import' },
  { id: 33, filename: 'c.png', variation_label: 'full body', source: 'import' },
]

// ── 1. Where am I, and what do the arrows do from here ──────────────────────

test('the middle of the list offers both directions and says the position', () => {
  const nav = lightboxNeighbours(IMAGES, 22)
  assert.equal(nav.available, true)
  assert.equal(nav.index, 1)
  assert.equal(nav.total, 3)
  assert.equal(nav.position, '2 / 3')
  assert.equal(nav.prev.id, 11)
  assert.equal(nav.next.id, 33)
  assert.equal(nav.prevReason, null)
  assert.equal(nav.nextReason, null)
})

test('the ends do not wrap — they name the end they are', () => {
  const first = lightboxNeighbours(IMAGES, 11)
  assert.equal(first.prev, null, 'the first image must not wrap round to the last')
  assert.equal(first.next.id, 22)
  assert.equal(first.prevReason, 'You are on the first of the 3 images shown here.')

  const last = lightboxNeighbours(IMAGES, 33)
  assert.equal(last.next, null, 'the last image must not wrap round to the first')
  assert.equal(last.prev.id, 22)
  assert.equal(last.nextReason, 'You are on the last of the 3 images shown here.')
})

test('a list of one says so instead of claiming a first and a last', () => {
  const nav = lightboxNeighbours([IMAGES[0]], 11)
  assert.equal(nav.position, '1 / 1')
  assert.equal(nav.prevReason, 'The current filters show only this image.')
  assert.equal(nav.nextReason, 'The current filters show only this image.')
})

test('an image that is not in the shown list gets no navigation at all', () => {
  // The rescue-review preview (a Curation pair), and an image a poll retired
  // under an open lightbox. Inventing a position for it would be a lie.
  for (const nav of [
    lightboxNeighbours(IMAGES, 999),
    lightboxNeighbours(null, 22),
    lightboxNeighbours([], 22),
    lightboxNeighbours(IMAGES, null),
  ]) {
    assert.equal(nav.available, false)
    assert.equal(nav.prev, null)
    assert.equal(nav.next, null)
    assert.equal(nav.position, '')
  }
})

// ── 2. The state that must NOT travel ───────────────────────────────────────

test('state computed for one image is never rendered for another', () => {
  // Everything a previous image could have been left in: zoomed to 100 %, its
  // comparison pane open, an improve pass running. `compareMode` covers BOTH
  // comparisons (against the original, against the reference photo) — the
  // reference one is the dangerous one to carry over, since it is offered on
  // every image and would silently follow you all the way down the grid.
  // `actionsOpen` joined them for a narrow-screen reason: the panel is a
  // full-screen drawer on a phone, and ⟩ pressed behind it would land you on an
  // image you cannot see, under a panel you never reopened.
  // `deciding` — a ✓ Keep / ✕ Reject in flight — is in the same slot for the
  // plainest reason of the lot: the verdict advances to the next picture as
  // soon as it lands, so a slow POST must not grey out the buttons of the image
  // you moved to.
  // `repairOpen` — the ✦ Repair dialog — is the newest member and the most
  // literal case for the rule: it is a full-screen overlay showing ONE image,
  // with a description typed about THAT image. Carried over, ⟩ would leave you
  // about to repaint a picture that is no longer the one you are looking at.
  const stale = {
    imageId: 11, full: true, compareMode: 'reference', improving: true, actionsOpen: true,
    repairOpen: true, deciding: true,
  }
  const live = lightboxImageState(stale, 22)
  assert.deepEqual(live, {
    imageId: 22, full: false, compareMode: 'none', improving: false, actionsOpen: false,
    repairOpen: false, deciding: false,
  })
  // The derived pane is the one that would be actively MISLEADING: captioned
  // "Original", showing the previous image's parent.
  assert.equal(
    lightboxImageState({ imageId: 11, full: false, compareMode: 'derived', improving: false }, 22)
      .compareMode,
    'none')
  // …while the image it WAS computed for keeps it — a reset that fires on every
  // render would make the zoom un-holdable.
  assert.equal(lightboxImageState(stale, 11), stale)
})

test('a write that lands after the move is ignored, not applied to the new image', () => {
  // The `finally` of an improve started on image 11: it stamps 11, and the
  // render for 22 discards it. No cancellation token, no ordering to get right.
  const late = { ...freshLightboxImageState(11), improving: false }
  assert.equal(lightboxImageState(late, 22).imageId, 22)
  assert.equal(lightboxImageState(late, 22).improving, false)
  const lateBusy = { ...freshLightboxImageState(11), improving: true }
  assert.equal(lightboxImageState(lateBusy, 22).improving, false,
    'an improve running on the PREVIOUS image must not lock the new one')
})

test('a missing or malformed stored state degrades to a fresh one', () => {
  for (const stored of [null, undefined, {}, { imageId: undefined }]) {
    assert.deepEqual(lightboxImageState(stored, 22), freshLightboxImageState(22))
  }
})

// ── 3. The arrow keys, and who owns them ────────────────────────────────────

test('a focused field keeps its own caret keys', () => {
  // The guard moved into components/shared/reviewShortcuts.js when the lightbox
  // started answering K/R/S as well: one grammar for both review surfaces. A
  // text field still owns ← →; a CHECKBOX deliberately does not, which is what a
  // private copy here got wrong in the Bank (the focus trap lands on the 🎲 box
  // and every shortcut went inert).
  for (const tag of ['INPUT', 'TEXTAREA', 'SELECT', 'input', 'textarea']) {
    assert.equal(ownsTypedKeys({ tagName: tag }), true, `${tag} owns ← →`)
  }
  assert.equal(ownsTypedKeys({ tagName: 'DIV', isContentEditable: true }), true)
  for (const tag of ['DIV', 'BUTTON', 'IMG', 'BODY']) {
    assert.equal(ownsTypedKeys({ tagName: tag }), false, `${tag} does not own ← →`)
  }
  assert.equal(ownsTypedKeys(null), false)
  assert.equal(reviewKeyAction({ key: 'ArrowRight', target: { tagName: 'INPUT' } }), null)
  assert.equal(reviewKeyAction({ key: 'ArrowRight', target: { tagName: 'DIV' } }), 'skip')
})

// ── 4. The page underneath follows ──────────────────────────────────────────

test('the grid page that holds an index', () => {
  assert.equal(GRID_PAGE_SIZE, 500)
  assert.equal(pageOfIndex(0), 0)
  assert.equal(pageOfIndex(499), 0)
  assert.equal(pageOfIndex(500), 1, 'crossing the boundary must turn the page')
  assert.equal(pageOfIndex(1234), 2)
  assert.equal(pageOfIndex(-1), 0)
  assert.equal(pageOfIndex(null), 0)
})

// ── 5. What is actually rendered ────────────────────────────────────────────

const lightbox = (props) => renderToStaticMarkup(
  createElement(ToastProvider, null,
    createElement(CapabilitiesProvider, null,
      createElement(DatasetLightbox, {
        img: IMAGES[1], datasetId: 3, onClose: () => {},
        images: IMAGES, onNavigate: () => {},
        ...props,
      }))))

/* The `<button …>` opening tag that CONTAINS `needle` — the arrows wrap their
   glyph in an aria-hidden <span>, so stopping at the nearest `<` would read the
   span's attributes instead of the button's. */
const buttonAround = (markup, needle) => {
  const at = markup.indexOf(needle)
  assert.ok(at >= 0, `not found in the markup: ${needle}`)
  const open = markup.lastIndexOf('<button', at)
  assert.ok(open >= 0, `no button encloses: ${needle}`)
  return markup.slice(open, markup.indexOf('>', open) + 1)
}
// The ATTRIBUTE, not the Tailwind class: the arrows carry `disabled:opacity-30`
// either way, so a bare /disabled/ would pass on a perfectly live button.
const DISABLED_ATTR = / disabled=""/

test('the lightbox renders both arrows and the position, in the middle', () => {
  const html = lightbox({})
  assert.match(html, />2 \/ 3</, 'the "where am I" counter must be on screen')
  const prev = buttonAround(html, 'Previous image (←)')
  const next = buttonAround(html, 'Next image (→)')
  assert.doesNotMatch(prev, DISABLED_ATTR)
  assert.doesNotMatch(next, DISABLED_ATTR)
  // The keyboard is invisible; the arrows are where it gets announced.
  assert.match(prev, /aria-label="Previous image \(←\)"/)
  assert.match(next, /aria-label="Next image \(→\)"/)
})

test('an end is a disabled button that says which end, in BOTH channels', () => {
  const html = lightbox({ img: IMAGES[0] })
  const prev = buttonAround(html, 'You are on the first of the 3 images shown here.')
  assert.match(prev, DISABLED_ATTR, 'the first image must not offer a ⟨ that does nothing')
  // `title` is what a mouse reads, `aria-label` what a screen reader announces —
  // and a title is unreachable on a touch screen. Asserting the sentence appears
  // SOMEWHERE would pass while one of the two silently reverted.
  assert.match(prev, /title="You are on the first of the 3 images shown here\."/)
  assert.match(prev, /aria-label="You are on the first of the 3 images shown here\."/)
  assert.doesNotMatch(buttonAround(html, 'Next image (→)'), DISABLED_ATTR)

  const lastHtml = lightbox({ img: IMAGES[2] })
  const next = buttonAround(lastHtml, 'You are on the last of the 3 images shown here.')
  assert.match(next, DISABLED_ATTR)
  assert.match(next, /aria-label="You are on the last of the 3 images shown here\."/)
})

test('no shown list, no arrows — nothing is invented for a rescue preview', () => {
  for (const props of [{ images: null, onNavigate: null }, { images: [], onNavigate: () => {} },
    { images: IMAGES, onNavigate: null }]) {
    const html = lightbox(props)
    assert.doesNotMatch(html, /Previous image/)
    assert.doesNotMatch(html, /Next image/)
    assert.doesNotMatch(html, />\d+ \/ \d+</)
  }
})

test('a running pass leaves navigation open — moving is a read', () => {
  // Consistent with the tiles behind the overlay, where inspecting and ticking
  // were deliberately taken back out of the `busy` lock. Nothing about ⟨ / ⟩
  // writes to the dataset.
  const reason = datasetBusyReason({ kind: 'generate', done: 12, total: 64, started_at: 0 })
  const html = lightbox({ busy: true, busyReason: reason, onCrop: () => {} })
  assert.doesNotMatch(buttonAround(html, 'Previous image (←)'), DISABLED_ATTR)
  assert.doesNotMatch(buttonAround(html, 'Next image (→)'), DISABLED_ATTR)
  // …and the writes in the same bar are still refused, so this is not a test
  // that would pass on a lightbox with no lock at all.
  assert.match(buttonAround(html, '✂ Crop'), DISABLED_ATTR)
})

test('the lightbox still renders in every view state it has, with arrows on', () => {
  // mountJsx only covers the states a test asks for; a branch nobody rendered is
  // how a ReferenceError has shipped here before.
  const compare = {
    available: true, beforeLabel: 'Original', afterLabel: 'Improved',
    parent: { id: 11, filename: 'a.png' },
  }
  for (const props of [{ compare }, { compare: { available: false, reason: 'no parent' } },
    { improvePending: true }, { improveReady: true }, { mirrorBusy: true },
    { onCrop: () => {}, onMirror: () => {}, onRotate: () => {}, onImprove: () => {} }]) {
    const html = lightbox(props)
    assert.match(html, /Next image \(→\)/)
  }
})

// ── 5b. Judging the picture you are looking at ──────────────────────────────
//
// The verdict used to live on the thumbnail BEHIND the overlay: you inspected an
// image full screen, then closed it to press ✓. The lightbox now carries the
// Bank's review bar — same three verdicts, same keys — wired to the dataset's
// own pending|keep|reject. These assertions are on the RENDERED markup for the
// same reason the arrows' are: whether a button survives the props it is
// computed from is what a regex over the source cannot answer.

test('the review bar appears only when a verdict can actually be written', () => {
  // No handler, no buttons — the rescue-review preview resolves its pair in
  // Curation, and three dead controls would be worse than none.
  const bare = lightbox({})
  for (const label of ['✓ Keep', '✕ Reject', '⏭ Skip']) {
    assert.ok(!bare.includes(label), `${label} must not appear without onStatus`)
  }
  const html = lightbox({ onStatus: () => {} })
  for (const label of ['✓ Keep', '✕ Reject', '⏭ Skip']) {
    assert.ok(html.includes(label), `${label} must be on screen`)
  }
  // The keys are printed on the caps AND spelled out under them.
  assert.match(html, /<kbd[^>]*>K<\/kbd>/)
  assert.match(html, /<kbd[^>]*>R<\/kbd>/)
  assert.match(html, /<kbd[^>]*>S<\/kbd>/)
  assert.ok(html.includes('K keep · R reject · S skip'))
  // …and in the tooltip AND the aria-label of each button, not in one channel:
  // a title is invisible to a screen reader and unreachable on a touch screen.
  const keep = buttonAround(html, '✓ Keep')
  assert.match(keep, /title="Keep this image and move on \(K\)/)
  assert.match(keep, /aria-label="Keep bust and move to the next image"/)
  assert.match(buttonAround(html, '✕ Reject'), /title="Reject this image and move on \(R\)/)
  assert.match(buttonAround(html, '⏭ Skip'), /title="Decide later \(S\)/)
})

test('the chip says which verdict the image already carries', () => {
  // Three buttons that also CHANGE the state cannot be its only reading: on the
  // last image of a list, where nothing moves, "did my K land?" would otherwise
  // have no answer at all.
  const kept = lightbox({ onStatus: () => {}, img: { ...IMAGES[1], status: 'keep' } })
  assert.ok(kept.includes('✓ kept'))
  const rejected = lightbox({ onStatus: () => {}, img: { ...IMAGES[1], status: 'reject' } })
  assert.ok(rejected.includes('✕ rejected'))
  const pending = lightbox({ onStatus: () => {}, img: { ...IMAGES[1], status: 'pending' } })
  assert.ok(pending.includes('· undecided'),
    'a never-judged image must say so rather than look kept')
})

test('⏭ Skip goes dead at the end of the list, and says why', () => {
  // Same rule as the ⟩ arrow: an end is a disabled control that NAMES the end,
  // never a mute no-op. The two verdicts stay live there — the picture simply
  // stays put once it is judged.
  const html = lightbox({ onStatus: () => {}, img: IMAGES[2] })
  const skip = buttonAround(html, '⏭ Skip')
  assert.match(skip, DISABLED_ATTR)
  assert.match(skip, /title="You are on the last of the 3 images shown here\."/)
  assert.match(skip, /aria-label="You are on the last of the 3 images shown here\."/)
  assert.doesNotMatch(buttonAround(html, '✓ Keep'), DISABLED_ATTR)
})

test('a running pass refuses the verdict in words, like the other writes', () => {
  // A status IS a write, unlike moving and zooming: it waits for the pass, and
  // says which one holds it rather than going quietly grey.
  const reason = datasetBusyReason({ kind: 'generate', done: 12, total: 64, started_at: 0 })
  const html = lightbox({ onStatus: () => {}, busy: true, busyReason: reason })
  const keep = buttonAround(html, '✓ Keep')
  assert.match(keep, DISABLED_ATTR)
  assert.match(keep, /title="[^"]*generat/i)
  // …while ⏭ Skip stays live: moving on is a read.
  assert.doesNotMatch(buttonAround(html, '⏭ Skip'), DISABLED_ATTR)
})

// ── 6. The wiring the harness cannot mount ──────────────────────────────────

test('the lightbox and the grid are handed the SAME list', () => {
  /* DatasetWorkspace is ~2 200 lines of context (routing, capabilities, a dozen
     hooks) and is not mountable here — so this one is a source assertion, said
     plainly. What it pins is the decision that matters: the arrows walk
     `gridImages`, the identifier the grid itself receives. A second filter
     computed for the lightbox would drift from the first one, silently. */
  const src = readFileSync(
    new URL('../src/components/dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8')
  assert.match(src, /<DatasetGrid images=\{gridImages\}/)
  assert.match(src, /images=\{viewImgLive\._rescueReviewPreview \? null : gridImages\}/)
  assert.match(src, /onNavigate=\{viewImgLive\._rescueReviewPreview \? null : setViewImg\}/)
  // The verdict taken in the lightbox is the SAME write the grid tile makes —
  // one status, not a second notion of "kept" for export and training to
  // disagree with.
  assert.match(src, /onStatus=\{viewImgLive\._rescueReviewPreview \? null : ds\.setStatus\}/)
  assert.match(src, /onStatus=\{ds\.setStatus\}|<DatasetGrid images=\{gridImages\} datasetId=\{d\.id\} onStatus=\{ds\.setStatus\}/)
  assert.match(src, /viewingImageId=\{viewImg\?\.id \?\? null\}/)
})

/* A LATE WRITER MUST NOT RESET THE IMAGE YOU MOVED TO.
 *
 * There is ONE state slot, stamped with the image it belongs to. Reading a
 * foreign stamp already falls back to a fresh state — but that is precisely
 * what makes a stale WRITE harmful: stamping the slot with the previous image
 * makes the read hand a FRESH state to the image on screen, silently dropping
 * its zoom and closing its comparison pane.
 *
 * The sequence below is the one that produced it: improve on A, ⟩ to B, zoom B,
 * then A's `finally` resolves. It is expressed against the pure helper because
 * the render harness is static markup and cannot drive an async resolution.
 * `stampedPatch(stored, patch, stampId, currentId)` — stampId is what the
 * callback closed over, currentId is what is on screen when it runs.
 */
test('an improve resolving on the previous image leaves the current one alone', () => {
  let stored = freshLightboxImageState(11)
  stored = stampedPatch(stored, { improving: true }, 11, 11)   // Improve clicked on A
  stored = stampedPatch(stored, { full: true }, 22, 22)        // ⟩ to B, then zoom B
  assert.equal(lightboxImageState(stored, 22).full, true)

  stored = stampedPatch(stored, { improving: false }, 11, 22)  // A's `finally` lands
  assert.equal(lightboxImageState(stored, 22).full, true,
    "B's zoom survived A's improve finishing")
  assert.equal(lightboxImageState(stored, 22).imageId, 22)
})

test('a patch from the image on screen still applies', () => {
  // The guard drops FOREIGN writers only -- it must not freeze the live one,
  // which would be a far worse cure than the disease.
  const stored = stampedPatch(freshLightboxImageState(22), { compareMode: 'reference' }, 22, 22)
  assert.equal(lightboxImageState(stored, 22).compareMode, 'reference')
  // Switching modes is the same single write, so the two comparisons can never
  // both be open — the state has one slot, not two flags.
  const switched = stampedPatch(stored, { compareMode: 'derived' }, 22, 22)
  assert.equal(lightboxImageState(switched, 22).compareMode, 'derived')
})

test('the lightbox routes its state writes through the stamp guard', () => {
  // The guard lives in the pure module; this pins that the component actually
  // uses it, and compares against the PRESENT (a ref) rather than the render it
  // closed over -- reintroducing a raw setStoredState would restore the bug
  // with every pure test above still green.
  const src = readFileSync(
    new URL('../src/components/dataset/DatasetLightbox.jsx', import.meta.url), 'utf8')
  assert.match(src, /stampedPatch\(prev, patch, imageId, currentIdRef\.current\)/)
  assert.doesNotMatch(src, /setStoredState\(\(prev\) => \(\s*\{ \.\.\.lightboxImageState/)
})

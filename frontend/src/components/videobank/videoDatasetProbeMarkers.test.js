/* 📐 The markers the responsive probe measures the video dataset workspace by.
 *
 * Same contract as datasetProbeMarkers.test.js and bankProbeMarkers.test.js:
 * `scripts/responsiveProbe.mjs` finds its surfaces by attribute, and these
 * assertions keep the attributes in place. Source text cannot say the layout is
 * good; it can say the probe is still pointed at the right elements — and a
 * probe pointed at nothing reports a clean page.
 *
 * First measured 2026-09-01, and the numbers are why this file exists: the page
 * landed at 359 px of fixed chrome on a 360×800 phone — 45% of the fold against
 * a 28% budget — and 184 px (47%) on a 844×390 phone held sideways. Four
 * changes brought it to 26% and 22%: the identity line hides below `sm`, the
 * sort shares the search row (`basis-0` — flex wraps on BASE sizes, so an input
 * whose base is `auto` pushes its neighbour onto a second row), the counts
 * moved out of the filter rail into the content line under it, and the
 * destinations rail no longer draws for a single destination. On a fold under
 * 500 px the whole clip toolbar folds behind a header button, because a header
 * plus a section rail already spends a quarter of 390 px.
 */
import assert from 'node:assert/strict'
import fs from 'node:fs'
import test from 'node:test'

const read = (rel) => fs.readFileSync(new URL(rel, import.meta.url), 'utf8')
const workspace = read('./VideoDatasetWorkspace.jsx')
const lightbox = read('./VideoDatasetLightbox.jsx')
const probe = read('../../../scripts/responsiveProbe.mjs')

test('every fixed surface of the workspace is marked for the responsive probe', () => {
  for (const surface of ['header', 'sections', 'destinations', 'grid-toolbar', 'filter-bar']) {
    assert.ok(workspace.includes(`data-probe-chrome="${surface}"`),
      `the ${surface} lost its data-probe-chrome marker — the probe stops measuring it`)
  }
  // The desktop rail is a side COLUMN: a panel (measured for fill), never
  // chrome — a column is not a share of the fold.
  assert.match(workspace, /data-probe-panel="sections-rail"/)
})

test('the lightbox is a layer, not chrome — it covers the page by design', () => {
  assert.match(lightbox, /data-probe-chrome="lightbox" data-probe-layer/)
})

test('the probe knows this page, and every state it opens has something to click', () => {
  assert.match(probe, /'#\/video-dataset':/,
    'responsiveProbe has no entry for this page — it would be measured at rest only')
  // Each state's selector must name something this workspace really renders.
  assert.match(probe, /nav\[aria-label="Video dataset sections"\]/)
  assert.match(workspace, /aria-label="Video dataset sections"/)
  assert.match(probe, /button:has-text\("Filter & sort"\)/)
  assert.match(workspace, /🔎 Filter & sort/)
})

test('the clip toolbar folds away on a fold under 500 px, and comes back', () => {
  // The three halves of one mechanism. Losing any one of them either leaves the
  // toolbar permanently hidden on a landscape phone, or puts it back on the
  // fold with no way to fold it again.
  assert.match(workspace, /id="vds-clips-tools"/)
  assert.match(workspace,
    /toolsOpen \? '' : '\[@media\(max-height:500px\)\]:hidden'/)
  assert.match(workspace, /aria-expanded=\{toolsOpen\} aria-controls="vds-clips-tools"/)
  assert.match(workspace,
    /\[@media\(max-height:500px\)\]:inline-flex/,
    'the button that unfolds it must exist ONLY where the toolbar is folded')
})

test('the search input keeps basis-0 — without it the toolbar doubles', () => {
  // Measured: 88 px instead of 40 at 360 px wide. flex wraps on base sizes, and
  // a search input's base size is its `auto` width (~170 px).
  assert.match(workspace, /type="search"[\s\S]{0,400}?grow basis-0/)
})

test('the per-clip Saving line RESERVES its space instead of inserting a row', () => {
  // blur is a discrete event: React flushes the state update before the browser
  // delivers the mouseup. A line appearing under the clip therefore pushed
  // everything below it down MID-CLICK — measured at +15 px between mousedown
  // and mouseup — and the click meant for a Caption tools button landed beside
  // it. The height has to exist whether or not anything is being saved.
  assert.match(workspace, /min-h-4 text-\[0\.625rem\] text-content-subtle/)
  assert.match(workspace, /\{savingId === clip\.id \? 'Saving…' : ''\}/)
  assert.ok(!/\{savingId === clip\.id && \(/.test(workspace),
    'a conditionally INSERTED status line shifts the page under the pointer')
})

test('the keydown handler DELEGATES to lightboxKeyAction — the decision is a tested value', () => {
  // The first version of this guard was a source regex on `onSave()` followed by
  // `onClose()` inside the Escape branch. One inserted line — an early return
  // while typing, i.e. the original bug — walked straight through it with the
  // suite green. So the decision now lives in a pure function with its own
  // tests (videoDatasetClips.test.js), and this only pins that the handler asks
  // it, with the typing flag, and does what it answers.
  assert.match(lightbox, /lightboxKeyAction\(e\.key, \{ typing: typing\.current, hasPrev, hasNext \}\)/,
    'the handler must ask lightboxKeyAction with the typing flag')
  assert.match(lightbox, /if \(action === 'save-close'\) \{ onSave\(\); onClose\(\) \}/,
    'save-close must save BEFORE it closes')
  assert.ok(!/if \(e\.key === 'Escape'\)/.test(lightbox),
    'no hand-written Escape branch may sit in front of the delegated decision')
  assert.match(lightbox, /\[onClose, onSave, onPrev, onNext, hasPrev, hasNext\]/,
    'onSave has to be in the effect deps, or Escape saves a stale draft')
})

test('the player is resolved on the FULL list and walks the FILTERED one, from its last slot', () => {
  // lightboxTargets is tested as a pure function; this pins the CALL SITE, the
  // only place a real regression can still happen. `(shown, shown, …)` — the
  // exact bug the rewire killed — left 4495 tests green until this line.
  assert.match(workspace, /const player = lightboxTargets\(items, shown, openId, lastIndex\.current\)/,
    'the clip comes from items, the stepping from shown, the fallback from the ref')
  assert.match(workspace, /if \(player\.index >= 0\) lastIndex\.current = player\.index/,
    'the slot has to be remembered while the clip is still in the filtered list')
})

test('every draft purge goes through purgeDraft with the value that was POSTED', () => {
  // purgeDraft is tested as a pure function; these pin the two CALL SITES, the
  // only places an unconditional purge can come back. A single save purges
  // against the caption it sent; the bulk pass folds purgeDraft over each
  // (id, after) it wrote. Typing during the round-trip must survive both.
  assert.match(workspace, /setDrafts\(\(m\) => purgeDraft\(m, clip\.id, caption\)\)/,
    'saveCaption must purge only if the draft still equals what was posted')
  assert.match(workspace, /written\.reduce\(\(acc, \{ id, after \}\) => purgeDraft\(acc, id, after\), m\)/,
    'applyCaptionOp must purge per clip against the value it posted')
  assert.ok(!/delete next\[clip\.id\]/.test(workspace),
    'no hand-written unconditional purge may remain')
})

test('the identity line and the destinations rail cost nothing at rest on a phone', () => {
  // The header's target/frames/size line is desktop-only and gone on a short
  // fold: it was a third header row on a 360 px screen.
  assert.match(workspace, /hidden text-xs text-content-muted sm:inline \[@media\(max-height:500px\)\]:hidden/)
  // A rail with ONE chip navigates to the section you are already looking at.
  assert.equal((workspace.match(/activePanels\.length > 1/g) || []).length, 2,
    'both rails (mobile chips and desktop sub-list) must hide a lone destination')
  assert.ok(!workspace.includes('activePanels.length > 0'),
    'a single destination must not draw a rail')
})

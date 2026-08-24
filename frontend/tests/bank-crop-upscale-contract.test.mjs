/* ✂ Crop / ✨ Upscale & improve in the Bank — wiring contract.
 *
 * Asked for by nofaceman on Discord (backed by mr.arrow). The risk this file
 * guards is NOT "the panel renders": it is the two ways this feature can ship
 * broken while every other test stays green.
 *
 *   1. THE EDIT IS INVISIBLE. `/thumb` answers with max-age=3600, so an image
 *      whose URL did not move keeps showing its pre-crop pixels for an hour —
 *      which is exactly what "the button did nothing" looks like. The version
 *      key has to be asked for by every surface that shows a bank image, and it
 *      has to carry the edit GENERATION (a re-crop changes the pixels under an
 *      unchanged `edit_method`).
 *   2. THE BANK PROMISE IS THE DATASET PROMISE. A dataset improve creates a
 *      separate candidate you review; a bank improve REPLACES what the bank
 *      shows. Copying the dataset's wording here would promise a validation step
 *      that does not exist, before a run that costs GPU-minutes per image.
 *
 * Rendering is real (support/mountJsx.mjs): a ReferenceError in any branch this
 * file reaches becomes a failed test rather than a white screen on a bank.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import './support/mountJsx.mjs'
import { WHATS_NEW } from '../src/whatsNew.js'
import { WHATS_NEW_ARCHIVE } from '../src/whatsNewArchive.js'

// The entry under test may have moved to the archive since it shipped
// (see whatsNew.js, rule "Keep the list tidy") — search the union.
const ALL_WHATS_NEW = [...WHATS_NEW, ...WHATS_NEW_ARCHIVE]
import { getHelpTopic } from '../src/help/helpRegistry.js'
import { BANK_PASSES, BANK_PASS_ORDER } from '../src/components/bank/bankPasses.js'
import { JOB_LABELS } from '../src/components/bank/bankPassRun.js'

const { default: BankEditPanel } =
  await import('../src/components/bank/BankEditPanel.jsx')
const { CapabilitiesProvider } = await import('../src/context/CapabilitiesContext.jsx')
const { ToastProvider } = await import('../src/components/common/Toast.jsx')
const { MemoryRouter } = await import('react-router')

const read = readSource
const tile = read('src/components/bank/BankTile.jsx')
const lightbox = read('src/components/bank/BankReviewLightbox.jsx')
// Review handlers live in useReviewLightbox since hook wave 5.
const workspace = read('src/components/bank/BankWorkspace.jsx')
  + read('src/components/bank/useReviewLightbox.js')
const passes = read('src/components/bank/BankPassesPanel.jsx')

/** The panel probes capabilities, toasts, and carries a ❓ HelpBadge that
 *  navigates — so it only renders inside the three providers its host is
 *  mounted in. (The badge's useNavigate() is what makes the Router mandatory:
 *  this harness found that in one run, which is the point of mounting rather
 *  than regex-ing.) */
const inApp = (node) =>
  renderToStaticMarkup(createElement(MemoryRouter, null,
    createElement(ToastProvider, null,
      createElement(CapabilitiesProvider, null, node))))

const payload = (cropped = 0, improved = 0, todo = 4) => ({
  counts: { total: 4, keep: 2, pending: 2, reject: 0, cropped, improved },
  pass_scopes: {
    improve: {
      todo: { keep: todo, pending: 0, reject: 0 },
      all: { keep: todo, pending: 0, reject: 0 },
    },
  },
})

const panel = (props = {}) => inApp(createElement(BankEditPanel, {
  bankId: 1, live: false, payload: payload(), selectedIds: [], onChanged: () => {},
  ...props,
}))

/* --- the cache key, on every surface that shows a bank image -------------- */

test('every bank image surface asks for the version key, edit generation included', () => {
  // The grid tile and the review lightbox are the two places a bank image is
  // rendered. Both must go through the shared helper — an inlined `?r=` would
  // serve the pre-crop image out of the browser cache for an hour.
  for (const [name, src] of [['tile', tile], ['lightbox', lightbox]]) {
    assert.match(src, /from '\.\/bankEdits\.js'/, `${name} does not read bankEdits`)
    assert.match(src, /imageVersionQuery\(/, `${name} builds its own cache key`)
  }
  assert.match(tile, /\/thumb\/\$\{img\.id\}\$\{imageVersionQuery\(img\)\}/)
  assert.match(lightbox, /\/file\/\$\{id\}\$\{imageVersionQuery\(img\)\}/)
  // The crop editor loads the SAME resolved image the server will cut, so the
  // box it returns means the same thing on both ends.
  assert.match(lightbox, /CropModal[\s\S]{0,400}\/file\/\$\{cropId\}\$\{imageVersionQuery/)
})

test('a crop made in Review re-keys the tile behind it', () => {
  // The grid is only refetched when Review CLOSES, so a tile left on the old
  // generation keeps serving the pre-crop thumbnail from cache underneath.
  assert.match(workspace, /const onReviewEdited = \(imageId, state\)/)
  assert.match(workspace, /onEdited=\{onReviewEdited\}/)
  assert.match(workspace, /edit_generation: state\?\.edit_generation \?\? 0/)
})

/* --- ✂ Crop lives where an image is big enough to draw on ---------------- */

test('✂ Crop is offered in Review, decides nothing, and can be reverted there', () => {
  assert.match(lightbox, /const cropCurrent = useCallback/)
  assert.match(lightbox, /\/api\/bank\/\$\{bankId\}\/image\/\$\{target\}\/crop/)
  assert.match(lightbox, /const revertCurrent = useCallback/)
  assert.match(lightbox, /\/api\/bank\/\$\{bankId\}\/edits\/revert/)
  // Neither action may advance the session: a badly framed image is fixed and
  // THEN judged — the same rule the quarter turn follows.
  const start = lightbox.indexOf('const applyEdit')
  const lane = lightbox.slice(start, lightbox.indexOf('// Moving forward without judging'))
  assert.ok(!/setSession/.test(lane), 'an edit must never advance the review')
  assert.match(lightbox, /C crop/)                  // printed, not folklore
})

/* --- ✨ Upscale & improve is a pass, and says what it really does --------- */

test('the ✨ pass is in the catalog, named in refusals, and out of the pass row', () => {
  const spec = BANK_PASSES.improve
  assert.ok(spec, '✨ improve is missing from BANK_PASSES')
  assert.equal(spec.endpoint, 'improve')
  assert.ok(spec.scopes && spec.selection, 'a GPU pass must be aimable')
  // No "run it again" tick box: an improved image leaves the pool and comes
  // back through ↩ Revert, so a second render is never a quiet side effect.
  assert.equal(spec.redo, null)
  assert.ok(spec.binCost, 'the bin is offered with no stated cost')
  // Named when it holds the bank, so another pass refuses in the user's words.
  assert.equal(JOB_LABELS.improve, '✨ Upscale & improve')
  // Its button lives on the ✂ Edits panel — a second copy in the pass row would
  // read as two different actions (the rule the cleaning levels already follow).
  assert.ok(!BANK_PASS_ORDER.includes('improve'))
  assert.match(passes, /<BankEditPanel/)
})

test('the caveats say the three things that cost the user something', () => {
  const text = BANK_PASSES.improve.caveats.join(' ')
  assert.match(text, /never written to/i)           // your files
  assert.match(text, /no candidate to validate/i)   // it REPLACES the image
  assert.match(text, /pass over those images again/i) // the measurements go
})

/* --- the panel renders, in the states that matter ------------------------- */

test('a bank with nothing edited stays collapsed, and offers no ↩ Revert', () => {
  // Editing is an occasional errand, not the reason the page is open — and a
  // ↩ Revert on a bank that has never been edited can only ever say "0".
  const html = panel()
  assert.match(html, /✂ Edits/)
  assert.match(html, /nothing edited yet/)
  assert.match(html, /aria-expanded="false"/)
  assert.doesNotMatch(html, /↩ Revert/)
})

test('a bank that HAS edits opens itself, with both engines and the ↩ Revert', () => {
  // This is also how the open branch gets EXECUTED at all: no event ever fires
  // in this harness, so a state nothing renders is a state nothing protects.
  const html = panel({ payload: payload(12, 4) })
  assert.match(html, /aria-expanded="true"/)
  assert.match(html, /12 cropped · 4 improved/)
  assert.match(html, /↩ Revert all \(16\)/)
  // Klein is always offered; SeedVR2 only once this install can run it — the
  // shared rule, read from utils/improveEngines.js rather than re-decided here.
  assert.match(html, /✨ Klein/)
  assert.doesNotMatch(html, /SeedVR2/)
  // The promise that is NOT the dataset's: no candidate to validate.
  assert.match(html, /replaces what this Bank shows/)
  assert.doesNotMatch(html, /candidate/)
  // ✂ Crop is not a button here — it needs a full-size image — so the panel has
  // to SAY where it is. A feature nobody can find reads as one never shipped.
  assert.match(html, /▶ Review/)
  assert.match(html, /press <b>C<\/b>/)
  // And it quotes the pool it would run on, from the server's own table.
  assert.match(html, /4 image\(s\) in this scope/)
})

test('a selection retargets ↩ Revert at the selection, not the whole bank', () => {
  const html = panel({ payload: payload(12, 4), selectedIds: [1, 2, 3] })
  assert.match(html, /↩ Revert selection \(3\)/)
  assert.doesNotMatch(html, /Revert all/)
})

test('the feature is announced and documented', () => {
  const entry = ALL_WHATS_NEW.find((e) => e.id === '2026-08-16-bank-crop-and-upscale')
  assert.ok(entry, "What's new entry for the bank crop/upscale is missing")
  assert.equal(entry.to, '/bank')
  assert.match(entry.blurb, /nofaceman/)            // credit where it is due
  // The distinguishing fact, in the entry itself: this crop resamples nothing.
  assert.match(entry.blurb, /resampled/i)
  for (const id of ['action-bank-crop', 'action-bank-improve', 'action-bank-revert-edits']) {
    const topic = getHelpTopic(id)
    assert.ok(topic, `help topic ${id} is missing`)
    assert.equal(topic.guide.anchor, 'crop-and-upscale-inside-a-bank')
  }
})

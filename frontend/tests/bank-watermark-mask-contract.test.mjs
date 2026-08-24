/* 🚩 Bank watermark mask — wiring contract (node --test cannot parse JSX, so the
 * properties that would silently un-ship the feature are asserted as text).
 *
 * Reported by Qeeyana (Reddit). The risk this file guards is NOT "the dialog
 * renders": it is the Bank quietly growing its own second mask editor, and the
 * mask being drawn in a coordinate space the cleaner does not use.
 */
import assert from 'node:assert/strict'
import { readSource } from './support/readSource.mjs'
import test from 'node:test'
import { WHATS_NEW } from '../src/whatsNew.js'
import { WHATS_NEW_ARCHIVE } from '../src/whatsNewArchive.js'

// The entry under test moved to the archive when it shipped long ago
// (see whatsNew.js, rule "Keep the list tidy") — search the union.
const ALL_WHATS_NEW = [...WHATS_NEW, ...WHATS_NEW_ARCHIVE]
import { getHelpTopic } from '../src/help/helpRegistry.js'

const read = readSource
const dialog = read('src/components/bank/BankWatermarkMaskDialog.jsx')
const lightbox = read('src/components/bank/BankReviewLightbox.jsx')

test('the Bank mounts the DATASET editor — it never grows a second one', () => {
  assert.match(dialog, /import WatermarkRegionEditor from '\.\.\/dataset\/WatermarkRegionEditor'/)
  assert.match(dialog, /<WatermarkRegionEditor/)
  // The geometry helpers come from the one shared module too.
  assert.match(dialog, /from '\.\.\/\.\.\/utils\/watermarkRegions\.js'/)
  // No hand-rolled drag/normalise maths sneaking back in on the bank side.
  assert.doesNotMatch(dialog, /getBoundingClientRect|onPointerMove/)
})

test('the mask is drawn on the SOURCE pixels the cleaner works on', () => {
  // The bank's watermark lane (scan, crop, inpaint) reads abs_image_path — the
  // unrotated source. Drawing on the rotated view would put every zone in the
  // wrong space, silently.
  assert.match(dialog, /\/api\/bank\/\$\{bankId\}\/file\/\$\{image\.id\}\?original=1/)
})

test('every edit is persisted to the bank route, and a failure says so', () => {
  assert.match(dialog, /putJson\(`\/api\/bank\/\$\{bankId\}\/image\/\$\{image\.id\}\/watermark-regions`/)
  const failure = dialog.match(/\.catch\(\(e\) => \{([\s\S]*?)\}\)/)[1]
  assert.match(failure, /setSave\(\{ status: 'failed'/)
  assert.doesNotMatch(failure, /onSaved/)          // a failed save is not a save
  assert.match(dialog, /is NOT saved/)             // and the user is told
})

test('the review lightbox offers the editor on any image it can still act on', () => {
  assert.match(lightbox, /canEditMask\(img\) && \(/)
  // The label is COMPUTED, never hardcoded: on an image the detector cleared or
  // never saw there is no box to "edit", and promising one is how the miss stayed
  // unanswerable. What each state reads is asserted on the helper's VALUES in
  // src/components/bank/bankWatermarkMask.test.js.
  assert.match(lightbox, /🚩 \{maskButtonLabel\(img\)\}/)
  assert.doesNotMatch(lightbox, /🚩 Edit mask/)
  // M is the Bank's OWN key, read off the same event once the shared review
  // grammar (K/R/S, ← , Esc) has declined it — and still behind the same
  // "does this field own the keystroke?" guard.
  assert.match(lightbox, /toLowerCase\(\) === 'm' && canEditMask\(img\)/)
  assert.match(lightbox, /M watermark mask/)       // printed, not folklore
})

test('the keyboard cannot decide on an image while an editor is open', () => {
  // K/R/S are one keystroke from a decision; the editor must own the keyboard.
  // Widened when the ✂ crop editor joined the mask editor in this lightbox: the
  // property being guarded was never "the mask specifically", it is that NO
  // open editor leaves the decision keys live underneath it. Both ids are named
  // so a third editor cannot be added without this line being read again.
  assert.match(lightbox, /if \(maskId != null \|\| cropId != null\) return/)
  assert.match(lightbox, /useFocusTrap\(dialogRef, maskId == null && cropId == null\)/)
})

test('the feature is announced and documented', () => {
  const entry = ALL_WHATS_NEW.find((e) => e.id === '2026-07-28-bank-watermark-mask-editing')
  assert.ok(entry, "What's new entry for the bank mask editor is missing")
  // Archived → no in-app target, by doctrine (whatsNew.js, "Keep the list tidy").
  assert.equal(entry.to, undefined)
  assert.match(entry.blurb, /Qeeyana/)             // credit where it is due
  const topic = getHelpTopic('bank-edit-watermark-mask')
  assert.ok(topic, 'help topic bank-edit-watermark-mask is missing')
  assert.equal(topic.guide.anchor, 'fix-a-watermark-mask-or-mark-one-the-scan-missed')
})

test('the dialog is usable with a thumb at 400 px', () => {
  // 44 px targets and a wrapping control row — drawing a rectangle by hand on a
  // phone is the hard case this feature has to survive.
  assert.match(dialog, /min-h-11/)
  assert.match(dialog, /flex-wrap/)
  assert.match(dialog, /\[container-type:size\]/)  // the photo caps to its cell
})

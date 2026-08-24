/**
 * Contract for the 🎨 Medium and ⤢ Angle chip rows.
 *
 * These two facets are the app's first CLASSIFIER-BACKED filters, and the
 * measurements behind them are weak in places that matter (see bankMedium.js).
 * So what is pinned here is not layout — it is the honesty rules that a future
 * rewrite would quietly drop:
 *   • a bucket id is a stored query value and may never be renamed;
 *   • "not measured" is never rendered as a verdict;
 *   • the limits sentence appears whenever there is a limit to state;
 *   • the backfill is never offered when there is nothing to backfill, and
 *     never without its price.
 */
import test from 'node:test'
import { readSource } from './support/readSource.mjs'
import assert from 'node:assert/strict'

import {
  ANGLE_BUCKETS, ANGLE_FRONTAL_MAX, ANGLE_PROFILE_MIN, MEDIUM_BUCKETS,
  angleBadge, angleOfYaw, angleReadiness, angleTitle, mediumLimits, mediumTitle,
  shownBuckets,
} from '../src/components/bank/bankMedium.js'
import { BANK_SORTS, bankSortOptions } from '../src/utils/gridSort.js'

const read = readSource

test('bucket ids are the stored contract with the server', () => {
  // These exact strings are query-string values AND, for medium, a database
  // column's contents. Renaming one silently empties a user's saved view.
  assert.deepEqual(MEDIUM_BUCKETS.map((b) => b.id),
    ['photo', 'anime', 'render3d', 'illustration', 'unsure'])
  assert.deepEqual(ANGLE_BUCKETS.map((b) => b.id),
    ['frontal', 'three_quarter', 'profile', 'behind'])
  // …and the SERVER agrees, read from its own source rather than restated here.
  const svc = read('../backend/app/services/image_bank_service.py')
  assert.match(svc, /^MEDIUMS = \('photo', 'anime', 'render3d', 'illustration'\)$/m)
  assert.match(svc, /^ANGLES = \('frontal', 'three_quarter', 'profile', 'behind'\)$/m)
  assert.match(svc, /^ANGLE_FRONTAL_MAX = 20\.0$/m)
  assert.match(svc, /^ANGLE_PROFILE_MIN = 60\.0$/m)
})

test('the two angle cuts match the server, so a chip and the grid cannot drift', () => {
  assert.equal(ANGLE_FRONTAL_MAX, 20)
  assert.equal(ANGLE_PROFILE_MIN, 60)
})

test('every bucket carries a tooltip, and the hard ones name their limit', () => {
  for (const b of MEDIUM_BUCKETS) assert.ok(mediumTitle(b.id), b.id)
  for (const b of ANGLE_BUCKETS) assert.ok(angleTitle(b.id), b.id)
  // The two verdicts that are measurably weakest must SAY so where the user
  // hovers, not only in the guide.
  assert.match(mediumTitle('anime'), /cosplay/i)
  assert.match(angleTitle('profile'), /detector/i)
  assert.match(angleTitle('behind'), /Framing/i)
})

test('an unmeasured yaw is never dressed up as a frontal shot', () => {
  assert.equal(angleOfYaw(null), null)
  assert.equal(angleOfYaw(undefined), null)
  assert.equal(angleOfYaw(NaN), null)
  assert.equal(angleOfYaw('0'), null)
  assert.equal(angleBadge({ face_yaw: null }), null)
  assert.equal(angleBadge({}), null)
})

test('yaw buckets are absolute and use the measured cuts', () => {
  assert.equal(angleOfYaw(0), 'frontal')
  assert.equal(angleOfYaw(19.9), 'frontal')
  assert.equal(angleOfYaw(-19.9), 'frontal')       // a turn is a turn either way
  assert.equal(angleOfYaw(20), 'three_quarter')
  assert.equal(angleOfYaw(-45), 'three_quarter')
  assert.equal(angleOfYaw(59.9), 'three_quarter')
  assert.equal(angleOfYaw(60), 'profile')
  assert.equal(angleOfYaw(-78.8), 'profile')       // the max seen in calibration
  assert.equal(angleBadge({ face_yaw: -74 }).id, 'profile')
})

test('a chip you are filtering on never disappears under the cursor', () => {
  const counts = { photo: 12, anime: 0, render3d: 0, illustration: 0, unsure: 3 }
  assert.deepEqual(shownBuckets(MEDIUM_BUCKETS, counts, null).map((b) => b.id),
    ['photo', 'unsure'])
  assert.deepEqual(shownBuckets(MEDIUM_BUCKETS, counts, 'anime').map((b) => b.id),
    ['photo', 'anime', 'unsure'])
  assert.deepEqual(shownBuckets(MEDIUM_BUCKETS, {}, null), [])
})

test('the medium row states its limits from THIS bank, not from a constant', () => {
  assert.equal(mediumLimits({}, 0), null)          // nothing measured, nothing to say
  const note = mediumLimits(
    { photo: 100, anime: 2, render3d: 0, illustration: 0, unsure: 40 }, 150)
  assert.match(note, /40 of 142 came back/)        // the real numbers
  assert.match(note, /cosplay/i)                   // the anime caveat, since anime > 0
  assert.match(note, /8 image\(s\) have no ✨ Score embedding/)
  // No anime verdicts ⇒ no cosplay warning: we do not scare people about a pile
  // they do not have.
  assert.doesNotMatch(
    mediumLimits({ photo: 10, anime: 0, render3d: 0, illustration: 0, unsure: 1 }, 11),
    /cosplay/i)
  // A real PILE of unsure earns the measured explanation of what it holds
  // (sampled by eye on a 36 921-image bank: cosplay/filtered photos held back
  // rather than mislabelled) — a handful does not need the theory.
  assert.match(
    mediumLimits({ photo: 33634, anime: 0, render3d: 0, illustration: 0, unsure: 3269 }, 36912),
    /mostly real photos it would not certify/)
  assert.doesNotMatch(
    mediumLimits({ photo: 100, anime: 0, render3d: 0, illustration: 0, unsure: 40 }, 140),
    /would not certify/)
})

test('the angle backfill is offered only when there is something to measure, and priced', () => {
  const none = angleReadiness({ counts: { angle_measured: 40, angle_backfillable: 0 },
    angles: { frontal: 30, three_quarter: 10 } })
  assert.equal(none.offer, null)

  const some = angleReadiness({
    counts: { angle_measured: 0, angle_backfillable: 7766, framing_classified: 10 },
    angles: {}, angle_backfill_minutes: 259,
  })
  assert.equal(some.offer.count, 7766)
  assert.equal(some.offer.minutes, 259)
  // The two things a user must know BEFORE clicking an hours-long job: why it
  // exists at all, and what it costs.
  assert.match(some.offer.why, /did not keep it/)
  assert.match(some.offer.why, /about 259 minute/)
})

test('the angle row says which pass is missing instead of showing an empty bucket', () => {
  const noPass = angleReadiness({ counts: {}, angles: {} })
  assert.match(noPass.note, /Person groups/)
  const noFace = angleReadiness({
    faces_scanned: 12,
    counts: { angle_measured: 0, angle_backfillable: 0 },
    angles: {},
  })
  assert.match(noFace.note, /12 images were face-checked/)
  assert.match(noFace.note, /no measurable head angle/)
  assert.doesNotMatch(noFace.note, /Run|Person groups/)
  assert.equal(noFace.offer, null)
  const noFraming = angleReadiness({
    counts: { angle_measured: 500, framing_classified: 0 },
    angles: { frontal: 400, behind: 0 },
  })
  assert.match(noFraming.note, /Framing/)
})

test('both measures are sortable, and gated on their OWN readiness count', () => {
  const ids = BANK_SORTS.map((s) => s.id)
  for (const id of ['yaw_desc', 'yaw_asc', 'medium_conf_desc', 'medium_conf_asc']) {
    assert.ok(ids.includes(id), id)
  }
  // A bank full of faces but with no measured angle must NOT offer a working
  // angle sort — gating it on `faces` would have done exactly that.
  const opts = bankSortOptions({ scanned: 10, scored: 10, faces: 7766,
    angle_measured: 0, medium_classified: 0 })
  const byId = Object.fromEntries(opts.map((o) => [o.id, o]))
  assert.equal(byId.yaw_desc.disabled, true)
  assert.match(byId.yaw_desc.label, /measure head angles/)
  assert.equal(byId.medium_conf_asc.disabled, true)
  assert.equal(byId.face_desc.disabled, false)     // faces themselves ARE ready
  const ready = bankSortOptions({ scanned: 10, scored: 10, faces: 10,
    angle_measured: 10, medium_classified: 10 })
  assert.equal(ready.find((o) => o.id === 'yaw_desc').disabled, false)
})

test('the two facets travel under their OWN payload keys', () => {
  // The regression this guards: a facet folded into `q`/`exclude`/`flag` silently
  // narrows the other thing that key already meant.
  const src = read('../frontend/src/components/bank/BankWorkspace.jsx')
  assert.match(src, /if \(f\.medium\) params\.medium = f\.medium/)
  assert.match(src, /if \(f\.angle\) params\.angle = f\.angle/)
  const routes = read('../backend/app/routes/bank.py')
  assert.match(routes, /medium=args\.get\('medium'\) or None/)
  assert.match(routes, /angle=args\.get\('angle'\) or None/)
  // …and the curation selectors read the SAME two keys, so a pick matches the
  // grid it was made from.
  assert.match(routes, /'medium': data\.get\('medium'\) or None/)
  assert.match(routes, /'angle': data\.get\('angle'\) or None/)
})

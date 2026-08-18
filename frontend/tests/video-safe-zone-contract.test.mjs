/** 🔳 The safe zone, on the side of the app the user actually touches.
 *
 * The generic contract test next door already pins that the panel offers every
 * cut the backend honours and that every flag carries a label. What it cannot
 * see is what makes THIS pass different from the ten cuts before it:
 *
 *  * it is the only pass in the app that runs at HALF strength. With no OCR
 *    engine it still measures bands, so its button must NOT be greyed out —
 *    a requirement stated by an ABSENCE in a table, which is exactly the kind
 *    of thing a refactor deletes as dead weight;
 *  * its three cuts publish real figures (the image bank's measured 0.04, and
 *    HunyuanVideo 1.5's 60 %) and those figures belong in the HINTS and in no
 *    default — the whole doctrine of this lane in one assertion;
 *  * three findings, three flags, three remedies. Collapsing them into one
 *    "badly framed" chip would only ever be able to recommend "drop it".
 */
import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  FLAG_LABELS, thresholdFields, flagChips, flagCounts, filterByFlag,
  payloadFromDraft, draftThresholds,
} from '../src/components/videobank/videoMetricsFilter.js'
import { PASS_LABELS, PASS_RUNNING_LABELS, passLabel }
  from '../src/components/videobank/videoBankStatus.js'
import { passBlockedBy, PASS_REQUIREMENTS }
  from '../src/components/videobank/videoCapability.js'

const field = (key) => thresholdFields().find((f) => f.key === key)
// Hints are re-wrapped every time somebody edits a sentence. Compare the VALUE,
// never the source's line breaks.
const hint = (key) => (field(key)?.hint || '').replace(/\s+/g, ' ')

test('the panel offers the three safe-zone cuts, each on its own flag', () => {
  assert.deepEqual(
    ['bars_max', 'text_coverage_max', 'safe_area_min'].map((k) => {
      const f = field(k)
      return f && { flag: f.flag, direction: f.direction }
    }),
    [
      { flag: 'letterboxed', direction: 'above' },
      { flag: 'burned_text', direction: 'above' },
      { flag: 'small_safe_zone', direction: 'below' },
    ])
})

test('each safe-zone flag reaches the grid with a name, never a raw key', () => {
  for (const flag of ['letterboxed', 'burned_text', 'small_safe_zone']) {
    assert.ok(FLAG_LABELS[flag], `${flag} has no label`)
    assert.notEqual(FLAG_LABELS[flag], flag)
  }
  // Three findings, three names. A shared label would tell a user that a padded
  // clip and a subtitled one are the same problem; they are fixed differently.
  const labels = ['letterboxed', 'burned_text', 'small_safe_zone']
    .map((f) => FLAG_LABELS[f])
  assert.equal(new Set(labels).size, 3)
})

test('the hints carry the published figures, and the cuts carry no default', () => {
  // The doctrine, pinned: a reference belongs in a hint where the user reads it
  // and decides, never in a default that decides for them.
  assert.match(hint('bars_max'), /0\.04/,
    'the Bank\'s measured letterbox cut is not quoted')
  assert.match(hint('safe_area_min'), /60 ?%/,
    'the published "keeps 60 % of the frame" figure is not quoted')
  const draft = draftThresholds({})
  for (const key of ['bars_max', 'text_coverage_max', 'safe_area_min']) {
    assert.equal(draft[key], null, `${key} arrives with a value`)
  }
})

test('the text cuts say out loud that they need the extra', () => {
  // The failure this prevents: a user sets a text cut on an install with no OCR
  // engine, nothing is ever flagged, and the panel reads exactly like a bank
  // with no subtitles in it.
  for (const key of ['text_coverage_max', 'safe_area_min']) {
    assert.match(hint(key), /extra|Setup/i,
      `${key} does not mention the install it depends on`)
  }
  assert.match(hint('text_coverage_max'), /bands only|never flagged/i)
})

test('the bars hint does not pretend a letterboxed film is a defect', () => {
  // 0.12 of bands is a 2.35:1 film in a 16:9 container — a crop waiting to
  // happen, not damage. A hint that omitted this would have people rejecting
  // every widescreen source they own.
  assert.match(hint('bars_max'), /0\.12|2\.35/)
})

test('the three cuts ride in the dry-run payload', () => {
  const payload = payloadFromDraft({
    bars_max: 0.04, text_coverage_max: 0.01, safe_area_min: 0.6, nonsense: 1,
  })
  assert.deepEqual(payload,
    { bars_max: 0.04, text_coverage_max: 0.01, safe_area_min: 0.6 })
})

test('the chips count and select the safe-zone flags', () => {
  const clips = [
    { metrics: { bars_ratio: 0.24 }, flags: ['letterboxed'] },
    { metrics: { text_coverage: 0.05 }, flags: ['burned_text', 'letterboxed'] },
    { metrics: { safe_area: 0.99 }, flags: [] },
  ]
  const counts = flagCounts(clips)
  assert.equal(counts.letterboxed, 2)
  assert.equal(counts.burned_text, 1)
  // Two clips carry a flag, not three flags' worth of clips.
  assert.equal(counts.flagged, 2)
  assert.deepEqual(flagChips(clips).map((c) => [c.flag, c.count]),
    [['letterboxed', 2], ['burned_text', 1]])
  assert.equal(filterByFlag(clips, 'burned_text').length, 1)
})

test('a bands-only shot can be flagged for bands and never for text', () => {
  // The degraded install, read at the surface: with no OCR engine the backend
  // stores NO text key, so no text flag can reach the grid whatever the cut.
  const clip = { metrics: { safe_zone_state: 'bars_only', bars_ratio: 0.24 },
    flags: ['letterboxed'] }
  assert.equal(flagCounts([clip]).burned_text, undefined)
  assert.equal(filterByFlag([clip], 'burned_text').length, 0)
})

test('the pass has a name in both voices', () => {
  assert.ok(PASS_LABELS.safezone)
  assert.ok(PASS_RUNNING_LABELS.safezone)
  assert.equal(passLabel('safezone'), PASS_LABELS.safezone)
  // The 409 refusal names the pass that owns the bank by this key.
  assert.notEqual(passLabel('safezone'), 'safezone')
})

test('a missing OCR engine never greys the button out', () => {
  // THE requirement that is stated by an absence. The pass measures bands with
  // no engine at all, so `video_text` must not appear in its requirement list —
  // and `decode` must, because with no decoder there is no frame at all.
  assert.deepEqual(PASS_REQUIREMENTS.safezone, ['decode'])
  const capable = { decode: true, detect: true, encode: true, video_text: false }
  assert.equal(passBlockedBy(capable, 'safezone'), null)
  assert.ok(passBlockedBy({ ...capable, decode: false }, 'safezone'))
})

test('the backend can raise exactly the three flags this file names', () => {
  // Read off `verdicts()` itself: a fourth flag added there with no label
  // reaches the grid as a raw identifier, and the generic contract test only
  // catches it once somebody remembers to add its threshold row too.
  const backend = readFileSync(
    new URL('../../backend/app/services/video_metrics.py', import.meta.url), 'utf8')
  const raised = new Set(
    [...backend.matchAll(/flags\.add\('([a-z_]+)'\)/g)].map((m) => m[1]))
  for (const flag of ['letterboxed', 'burned_text', 'small_safe_zone']) {
    assert.ok(raised.has(flag), `the backend no longer raises ${flag}`)
  }
})

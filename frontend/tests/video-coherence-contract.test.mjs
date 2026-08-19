import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import {
  FLAG_LABELS, thresholdFields, flagChips,
} from '../src/components/videobank/videoMetricsFilter.js'

// 🔗 The coherence cut in the panel, held to what the backend actually honours
// and to what the calibration actually found. The hints in this table are the
// only place a user meets those numbers, so they are pinned like behaviour.

const backend = readFileSync(
  new URL('../../backend/app/services/video_metrics.py', import.meta.url), 'utf8')
const guide = readFileSync(
  new URL('../../docs/guide/using-the-app.md', import.meta.url), 'utf8')

const field = () => thresholdFields().find((f) => f.key === 'coherence_floor')

test('the coherence cut has a panel row that feeds a labelled flag', () => {
  const f = field()
  assert.ok(f, 'coherence_floor has no row in the thresholds panel')
  assert.equal(f.flag, 'missed_cut')
  // A FLOOR: low similarity is the suspicious side. Written as 'above' it would
  // flag every ordinary shot in the bank and clear every double one.
  assert.equal(f.direction, 'below')
  assert.ok(FLAG_LABELS.missed_cut)
})

test('the backend raises exactly the flag the panel row promises', () => {
  assert.match(backend, /flags\.add\('missed_cut'\)/)
  // And it reads the SPAN, not the minimum — the aggregate the calibration
  // chose. Reading coherence_min here would flag shots whose middle frame
  // merely wandered.
  assert.match(backend, /scores\.get\('coherence_span'\)/)
  assert.ok(!backend.includes("scores.get('coherence_min')"),
    'the verdict must read the span, not the minimum')
})

test('the flag is named for the remedy, not for the number', () => {
  // "Low coherence" would name the measurement and tell the user nothing about
  // what to do with the clip; the remedy is to split it.
  assert.match(FLAG_LABELS.missed_cut, /cut/i)
  assert.doesNotMatch(FLAG_LABELS.missed_cut, /coherence|similarity/i)
})

test('the coherence chip is not sold as the stillness chip', () => {
  // They sit at OPPOSITE ends of the same similarity and only one of them is on
  // this scale at all. Identical-looking wording would send a user to the wrong
  // cut for a still bank.
  assert.notEqual(FLAG_LABELS.missed_cut, FLAG_LABELS.still)
  assert.doesNotMatch(FLAG_LABELS.missed_cut, /move|still|motion/i)
})

test('the hint states how reliable it is, in numbers, before anyone sets it', () => {
  const { hint } = field()
  // A ranking, not a verdict — and the figures that make that concrete.
  assert.match(hint, /0\.80/)
  assert.match(hint, /third/i)
  assert.match(hint, /one honest shot in seven/i)
})

test('the hint warns that a long shot scores lower with no cut in it', () => {
  // The confound the calibration found (Spearman -0.41 against duration). A user
  // who meets it on their first long take with no warning reads the pass as
  // broken.
  const { hint } = field()
  assert.match(hint, /long/i)
  assert.match(hint, /0\.84/)
})

test('the hint says what an unmeasured shot does, like every other cut here', () => {
  const { hint } = field()
  assert.match(hint, /never flagged/i)
  assert.match(hint, /Find scenes/)
})

test('the panel offers no stillness cut on this number', () => {
  // The refuted half of Panda-70M's rule. A test on an ABSENCE, because the
  // absence is the finding: this number tracks shot length rather than motion,
  // and a second row here would put a measured-and-rejected claim in front of
  // users.
  const keys = thresholdFields().map((f) => f.key)
  assert.equal(keys.filter((k) => k.startsWith('coherence_')).length, 1)
  for (const flag of Object.keys(FLAG_LABELS)) {
    assert.ok(!/slideshow|false_?movement|no_?change/.test(flag),
      `${flag} re-adds the stillness claim this pass refuted`)
  }
})

test('the chip only offers itself once some shot carries the flag', () => {
  const none = flagChips([{ metrics: {}, flags: [] }])
  assert.ok(!none.some((c) => c.flag === 'missed_cut'))
  const some = flagChips([{ metrics: {}, flags: ['missed_cut'] },
                          { metrics: {}, flags: [] }])
  const chip = some.find((c) => c.flag === 'missed_cut')
  assert.equal(chip.count, 1)
  assert.equal(chip.label, FLAG_LABELS.missed_cut)
})

test('the Guide section the help topic points at actually explains this', () => {
  // The registry validates that the ANCHOR exists; nothing validates that the
  // chapter says anything about the pass. It is an H3 inside the quality H2, so
  // the anchor alone cannot be the check.
  assert.match(guide, /^###\s+🔗 /m)
  assert.match(guide, /scene coherence/i)
  assert.match(guide, /Cut inside the shot/)
  // And it must carry the honest limit rather than only the feature.
  assert.match(guide, /ranking/i)
})

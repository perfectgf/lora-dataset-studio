import test from 'node:test'
import assert from 'node:assert/strict'

import {
  canRecut, dryRunSummary, effectiveThreshold, parseThreshold, recutSummary,
  sweepRows, thresholdLabel, transitionChip,
} from './videoShotCuts.js'

/** 🎬 Find shots — the numbers and the sentences, with no React around them.
 *
 * The threshold is the one control in this lane where an empty field and a zero
 * mean opposite things: empty is "inherit", zero cuts on every single frame. Half
 * of these tests exist to keep those two apart on every path a value can take
 * between the input box and the server.
 *
 * The rest hold the sentences honest. A preview that says "3 shots" while three
 * of the bank's files could not be answered for is worse than no preview at all.
 */

// --- the threshold field -------------------------------------------------------

test('an empty field means INHERIT, never zero', () => {
  assert.equal(parseThreshold(''), null)
  assert.equal(parseThreshold('   '), null)
  assert.equal(parseThreshold(null), null)
})

test('a typed zero is a real threshold and survives as one', () => {
  assert.equal(parseThreshold('0'), 0)
})

test('a value outside 0..1 is refused with the sentence the user needs', () => {
  assert.throws(() => parseThreshold('7'), /between 0 and 1/)
  assert.throws(() => parseThreshold('-1'), /between 0 and 1/)
})

test('anything that is not a number is refused rather than sent as NaN', () => {
  assert.throws(() => parseThreshold('high'), /between 0 and 1/)
})

test('the label says when a value is inherited, and what it inherits', () => {
  assert.equal(thresholdLabel(null, 0.5), 'Default (0.50)')
  assert.equal(thresholdLabel(0.75, 0.5), '0.75')
  assert.equal(thresholdLabel(0, 0.5), '0.00')
})

test('a file falls back to its bank, and the bank to the global default', () => {
  assert.equal(effectiveThreshold({ shot_threshold: 0.4 }, 0.8, 0.5), 0.4)
  assert.equal(effectiveThreshold({ shot_threshold: null }, 0.8, 0.5), 0.8)
  assert.equal(effectiveThreshold({ shot_threshold: null }, null, 0.5), 0.5)
})

test('a file threshold of zero is not treated as unset', () => {
  assert.equal(effectiveThreshold({ shot_threshold: 0 }, 0.8, 0.5), 0)
})

// --- the preview ---------------------------------------------------------------

test('the preview marks the threshold currently in force', () => {
  const rows = sweepRows({
    rows: [{ threshold: 0.5, shots: 12 }, { threshold: 0.8, shots: 4 }],
    current: 0.5,
  })
  assert.deepEqual(rows.map((r) => r.current), [true, false])
})

test('each row says how it differs from the one in force', () => {
  // "4" means nothing on its own; "8 fewer than now" is the decision.
  const rows = sweepRows({
    rows: [{ threshold: 0.5, shots: 12 }, { threshold: 0.8, shots: 4 }],
    current: 0.5,
  })
  assert.equal(rows[1].delta, -8)
  assert.match(rows[1].deltaLabel, /8 fewer/)
})

test('the row in force says so instead of saying "0 fewer"', () => {
  const rows = sweepRows({ rows: [{ threshold: 0.5, shots: 12 }], current: 0.5 })
  assert.match(rows[0].deltaLabel, /in force|current/i)
})

test('a preview with nothing in force still reads', () => {
  const rows = sweepRows({ rows: [{ threshold: 0.5, shots: 12 }] })
  assert.equal(rows[0].current, false)
  assert.equal(rows[0].deltaLabel, '')
})

test('the summary names the files it could NOT answer for', () => {
  const note = dryRunSummary({ rows: [], sources: 4, skipped: 3 })
  assert.match(note, /4 files/)
  assert.match(note, /3/)
  assert.match(note, /Find shots/)
})

test('…and stays quiet about them when every file answered', () => {
  const note = dryRunSummary({ rows: [], sources: 4, skipped: 0 })
  assert.match(note, /4 files/)
  assert.ok(!/could not/i.test(note), note)
})

test('a preview over an empty bank says so rather than showing nothing', () => {
  assert.match(dryRunSummary({ rows: [], sources: 0, skipped: 0 }), /no file/i)
})

test('files left out because they are single takes are named, not "skipped"', () => {
  // They are not missing anything. The count leaves them out precisely because
  // the re-cut will too — folding them into "could not be counted" would offer
  // a fix for something that is working as asked.
  const note = dryRunSummary({ rows: [], sources: 2, skipped: 0, single_shot: 1 })
  assert.match(note, /single take/i)
  assert.ok(!/could not be counted/i.test(note), note)
})

// --- what a re-cut did ---------------------------------------------------------

test('the re-cut summary reports what it left alone, not only what it did', () => {
  const note = recutSummary({ sources: 5, clips: 210, skipped: 2, single_shot: 1 })
  assert.match(note, /210 shots/)
  assert.match(note, /5 files/)
  assert.match(note, /2/)
  assert.match(note, /single take/i)
})

test('a clean re-cut does not invent caveats', () => {
  const note = recutSummary({ sources: 5, clips: 210, skipped: 0, single_shot: 0 })
  assert.ok(!/single take/i.test(note), note)
  assert.ok(!/could not/i.test(note), note)
})

// --- which files can be re-cut instantly ---------------------------------------

test('a file with a cached vector can be re-cut instantly', () => {
  assert.equal(canRecut({ has_probs: true, probe_state: 'ok' }), true)
})

test('a file detected before the cache existed cannot, and is not blamed', () => {
  assert.equal(canRecut({ has_probs: false, probe_state: 'ok' }), false)
})

test('an unreadable file cannot be re-cut either', () => {
  assert.equal(canRecut({ has_probs: true, probe_state: 'unreadable' }), false)
})

// --- the transition chip -------------------------------------------------------

test('a dissolve gets a chip carrying its width in frames', () => {
  const chip = transitionChip({ transition: { start: null, end: { kind: 'dissolve', width: 18 } } })
  assert.equal(chip.label, 'dissolve 18f')
})

test('a hard cut gets no chip, because a chip on everything says nothing', () => {
  assert.equal(transitionChip({ transition: { start: { kind: 'cut', width: 1 }, end: null } }), null)
})

test('a shot dissolving at both ends is named once, by its widest edge', () => {
  const chip = transitionChip({
    transition: { start: { kind: 'dissolve', width: 9 }, end: { kind: 'dissolve', width: 22 } },
  })
  assert.equal(chip.label, 'dissolve 22f')
  assert.match(chip.title, /both ends/i)
})

test('a clip nobody measured gets no chip rather than a guessed one', () => {
  assert.equal(transitionChip({ transition: null }), null)
  assert.equal(transitionChip({}), null)
})

test('the chip explains that the label is a reading, not a verdict', () => {
  const chip = transitionChip({ transition: { start: null, end: { kind: 'dissolve', width: 18 } } })
  assert.match(chip.title, /dissolve/i)
  assert.ok(chip.title.length > chip.label.length, 'the tooltip must explain')
})

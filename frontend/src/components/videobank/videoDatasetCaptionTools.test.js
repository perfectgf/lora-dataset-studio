import test from 'node:test'
import assert from 'node:assert/strict'

import {
  captionEditPlan, captionEditConfirmation, captionEditProgressLabel,
  captionEditReport, CAPTION_OPS,
} from './videoDatasetCaptionTools.js'

const clip = (id, caption) => ({ id, filename: `clip_000${id}.mp4`, caption })

// ---- the plan is what stops 300 pointless disk writes ------------------------

test('a no-op is not in the plan', () => {
  const clips = [clip(1, 'a woman walks'), clip(2, 'a man runs')]
  // Replacing a word with itself changes nothing, so nothing is sent.
  assert.deepEqual(captionEditPlan(clips, { kind: 'replace', find: 'woman', replace: 'woman' }), [])
  // A term nobody says is the same story.
  assert.deepEqual(captionEditPlan(clips, { kind: 'replace', find: 'zebra', replace: 'x' }), [])
})

test('the plan carries before AND after, so the screen can show the diff', () => {
  const plan = captionEditPlan([clip(1, 'a woman walks')],
    { kind: 'replace', find: 'woman', replace: 'man' })
  assert.deepEqual(plan, [{
    id: 1, filename: 'clip_0001.mp4', before: 'a woman walks', after: 'a man walks',
  }])
})

test('an unknown operation plans nothing rather than guessing one', () => {
  const clips = [clip(1, 'a woman walks')]
  assert.deepEqual(captionEditPlan(clips, { kind: 'shout', text: 'x' }), [])
  assert.deepEqual(captionEditPlan(clips, null), [])
  // And every op the UI can offer is one the planner really handles.
  for (const kind of CAPTION_OPS) {
    const op = kind === 'replace'
      ? { kind, find: 'woman', replace: 'person' }
      : { kind, text: 'cinematic' }
    assert.ok(captionEditPlan(clips, op).length > 0, `${kind} planned nothing`)
  }
})

// ---- find/replace ------------------------------------------------------------

test('replace is case-insensitive and hits every occurrence', () => {
  const plan = captionEditPlan([clip(1, 'Woman beside a woman')],
    { kind: 'replace', find: 'woman', replace: 'man' })
  assert.equal(plan[0].after, 'man beside a man')
})

test('the find field is literal text, not a regular expression', () => {
  // Without escaping, "a.b" would match "axb" and a "(" would throw.
  const plan = captionEditPlan([clip(1, 'shot a.b here'), clip(2, 'shot axb here')],
    { kind: 'replace', find: 'a.b', replace: 'Z' })
  assert.deepEqual(plan.map((p) => p.id), [1])
  assert.equal(plan[0].after, 'shot Z here')
  assert.doesNotThrow(() => captionEditPlan([clip(3, 'a (close) shot')],
    { kind: 'replace', find: '(close)', replace: 'wide' }))
})

test('whole-word mode does not eat a word that merely contains the term', () => {
  const clips = [clip(1, 'a man walks'), clip(2, 'a woman walks')]
  const loose = captionEditPlan(clips, { kind: 'replace', find: 'man', replace: 'person' })
  assert.deepEqual(loose.map((p) => p.id), [1, 2])          // "woman" contains "man"
  const strict = captionEditPlan(clips,
    { kind: 'replace', find: 'man', replace: 'person', wholeWord: true })
  assert.deepEqual(strict.map((p) => p.id), [1])
  assert.equal(strict[0].after, 'a person walks')
})

test('an empty replacement removes the term without leaving a bare comma', () => {
  const plan = captionEditPlan([clip(1, 'a woman, blurry, on a beach')],
    { kind: 'replace', find: 'blurry', replace: '' })
  assert.equal(plan[0].after, 'a woman, on a beach')
  const head = captionEditPlan([clip(2, 'blurry, a woman')],
    { kind: 'replace', find: 'blurry', replace: '' })
  assert.equal(head[0].after, 'a woman')
  const tail = captionEditPlan([clip(3, 'a woman, blurry')],
    { kind: 'replace', find: 'blurry', replace: '' })
  assert.equal(tail[0].after, 'a woman')
})

// ---- prefix / suffix ---------------------------------------------------------

test('a prefix reaches the SILENT clips, which is the point of having one', () => {
  const plan = captionEditPlan([clip(1, ''), clip(2, 'a woman walks')],
    { kind: 'prefix', text: 'sks_style' })
  assert.deepEqual(plan.map((p) => p.after), ['sks_style', 'sks_style, a woman walks'])
})

test('a suffix does NOT invent a caption out of an empty one', () => {
  const plan = captionEditPlan([clip(1, ''), clip(2, 'a woman walks')],
    { kind: 'suffix', text: 'cinematic' })
  assert.deepEqual(plan.map((p) => p.id), [2])
  assert.equal(plan[0].after, 'a woman walks, cinematic')
})

test('running a prefix pass twice does not double it', () => {
  const once = captionEditPlan([clip(1, 'a woman walks')], { kind: 'prefix', text: 'sks' })
  assert.equal(once[0].after, 'sks, a woman walks')
  const twice = captionEditPlan([clip(1, once[0].after)], { kind: 'prefix', text: 'sks' })
  assert.deepEqual(twice, [])
  // Same guarantee at the other end.
  const suffixed = captionEditPlan([clip(1, 'a woman walks, cinematic')],
    { kind: 'suffix', text: 'cinematic' })
  assert.deepEqual(suffixed, [])
})

test('empty text plans nothing at all', () => {
  const clips = [clip(1, 'a woman walks')]
  assert.deepEqual(captionEditPlan(clips, { kind: 'prefix', text: '   ' }), [])
  assert.deepEqual(captionEditPlan(clips, { kind: 'suffix', text: '' }), [])
  assert.deepEqual(captionEditPlan(clips, { kind: 'replace', find: '', replace: 'x' }), [])
})

// ---- what the user is told ---------------------------------------------------

test('the confirm names the count and says the .txt files are rewritten', () => {
  assert.equal(captionEditConfirmation([], { kind: 'replace' }), null)
  const text = captionEditConfirmation([{}, {}, {}], { kind: 'replace' })
  assert.match(text, /3 captions will be rewritten/)
  assert.match(text, /\.txt/)
  assert.match(captionEditConfirmation([{}], { kind: 'prefix' }), /1 caption will gain the prefix/)
  assert.match(captionEditConfirmation([{}], { kind: 'suffix' }), /gain the suffix/)
})

test('the progress line counts what is really being written', () => {
  assert.equal(captionEditProgressLabel(3, 12), 'Rewriting captions - 3 of 12...'
    .replace('-', '—').replace('...', '…'))
})

test('a failed sidecar is REPORTED, never rounded off into the success', () => {
  assert.match(captionEditReport({ changed: 5 }), /5 captions rewritten/)
  assert.match(captionEditReport({ changed: 5 }), /\.txt files included/)
  assert.equal(captionEditReport({}), 'Nothing was changed.')
})

test('a request that threw and a sidecar that would not write are DIFFERENT news', () => {
  // The server commits the row before it tries the sidecar, so the two outcomes
  // are opposites: one left the app showing text the trainer will not read, the
  // other moved nothing at all. Saying "still hold their previous text" about
  // both was true of one and a lie about the other.
  const sidecar = captionEditReport({ changed: 4, sidecarFailed: 1 })
  assert.match(sidecar, /4 captions rewritten/)
  assert.match(sidecar, /saved in the app but their \.txt could NOT be written/)
  assert.match(sidecar, /training will read the previous text/)
  assert.doesNotMatch(sidecar, /could not be saved at all/)

  const threw = captionEditReport({ changed: 4, failed: 1 })
  assert.match(threw, /1 could not be saved at all and still hold their previous text/)
  assert.doesNotMatch(threw, /\.txt could NOT be written/)

  // And all three at once still reads as three separate facts.
  const both = captionEditReport({ changed: 2, sidecarFailed: 1, failed: 3 })
  assert.match(both, /2 captions rewritten/)
  assert.match(both, /1 saved in the app/)
  assert.match(both, /3 could not be saved at all/)
})

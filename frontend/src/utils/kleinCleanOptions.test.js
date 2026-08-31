import assert from 'node:assert/strict'
import test from 'node:test'
import {
  CLEAN_MAX_MP_DEFAULT, CLEAN_MAX_MP_MAX, CLEAN_MAX_MP_MIN, CLEAN_MP_CHOICES,
  CLEAN_OUTPUT_MODES, CLEAN_PROMPT_DEFAULT, cleanPromptText, clampMaxMp, formatMp,
  maxMpChoices, mpNote, normalizeOutput, outputNote, sentPromptLine,
} from './kleinCleanOptions.js'

/* These clamps MIRROR the backend (watermark_klein.clean_prompt / clean_max_mp /
 * clean_output_mode). The backend re-resolves whatever it is handed, so nothing here is
 * a security boundary — what these tests protect is the screen never showing a value
 * the pass would silently change under the user, which is how a dial stops being
 * trusted. backend/tests/test_klein_clean_options.py pins the same numbers. */

test('a blank prompt means the shipped default, never an empty instruction', () => {
  assert.equal(CLEAN_PROMPT_DEFAULT, 'remove watermark')
  for (const blank of ['', '   ', undefined, null, 7, {}]) {
    assert.equal(cleanPromptText(blank), 'remove watermark')
  }
  // trimmed, not reformatted: blanks around a typed instruction are typing, not intent
  assert.equal(cleanPromptText('  erase every logo  '), 'erase every logo')
})

test('the processing size is clamped to the range the backend supports', () => {
  assert.deepEqual([CLEAN_MAX_MP_MIN, CLEAN_MAX_MP_DEFAULT, CLEAN_MAX_MP_MAX],
    [0.5, 2, 4])
  assert.equal(clampMaxMp(12), 4)
  assert.equal(clampMaxMp(0.01), 0.5)
  assert.equal(clampMaxMp(3), 3)
  assert.equal(clampMaxMp('1.5'), 1.5)      // config.json hand-edited as a string
  for (const junk of [undefined, null, 'abc', NaN, Infinity, [], {}]) {
    assert.equal(clampMaxMp(junk), 2, `${String(junk)} should read as the default`)
  }
  // Number(true) is 1 — a stray `true` silently becoming a 1 MP cap is unfindable.
  assert.equal(clampMaxMp(true), 2)
  assert.equal(clampMaxMp(false), 2)
})

test('the offered sizes read as less / as before / more', () => {
  assert.deepEqual(CLEAN_MP_CHOICES, [1, 1.5, 2, 3, 4])
  assert.ok(CLEAN_MP_CHOICES.includes(CLEAN_MAX_MP_DEFAULT))
  assert.deepEqual(maxMpChoices(2), CLEAN_MP_CHOICES)
  /* A hand-edited value the list does not offer is KEPT, not snapped: a select that
     silently showed 2 MP while the pass ran at 2.5 would be the one lie this panel
     exists to remove. */
  assert.deepEqual(maxMpChoices(2.5), [1, 1.5, 2, 2.5, 3, 4])
  assert.deepEqual(maxMpChoices(99), [1, 1.5, 2, 3, 4])   // clamped to 4, already listed
})

test('the write-back mode falls back to the one that cannot resize anybody’s files', () => {
  assert.deepEqual(CLEAN_OUTPUT_MODES.map((m) => m.id), ['original', 'render'])
  assert.equal(normalizeOutput('  RENDER '), 'render')
  for (const junk of ['', 'native', 'full', undefined, null, 3]) {
    assert.equal(normalizeOutput(junk), 'original')
  }
})

test('the size note states BOTH what it buys and what it costs', () => {
  const note = mpNote(4)
  assert.match(note, /4 MP/)
  assert.match(note, /detail/i, 'the note does not say what a larger size buys')
  assert.match(note, /VRAM/i)
  assert.match(note, /time/i)
  // The half people get wrong: raising the cap does nothing for a small photo.
  assert.match(note, /never enlarged|nothing is ever enlarged/i)
  assert.match(mpNote(1.5), /1\.5 MP/)
})

test('the render mode says the file changes dimensions, in those words', () => {
  const render = outputNote('render')
  assert.match(render, /CHANGES DIMENSIONS/,
    'a user who discovers this after a batch has no undo but ↩ Restore original')
  assert.match(render, /smaller/i)
  assert.match(render, /Restore original/)
  const original = outputNote('original')
  assert.match(original, /never\s+changes the shape/i)
  assert.doesNotMatch(original, /CHANGES DIMENSIONS/)
  // An unknown mode describes the SAFE behaviour, which is also the one it resolves to.
  assert.equal(outputNote('nonsense'), original)
})

test('formatMp writes a number a user would type', () => {
  assert.equal(formatMp(2), '2')
  assert.equal(formatMp(1.5), '1.5')
  assert.equal(formatMp(4), '4')
})

test('the prompt line quotes what is sent', () => {
  assert.equal(sentPromptLine('erase every logo'), 'Sent to Klein: “erase every logo”')
  assert.equal(sentPromptLine(''), 'Sent to Klein: “remove watermark”')
})

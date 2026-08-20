import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CUSTOM_SHOT_PREVIEW_CHARS, customShotLabel, createCustomShot,
  editCustomShot, customShotDraft,
} from './customShots.js';

const SHOTS = [
  { id: 'custom_1', label: '✨ sitting on a bench', prompt: 'sitting on a bench', framing: 'body' },
  { id: 'custom_2', label: '🔞 lying on a bed', prompt: 'lying on a bed', framing: 'body', nsfw: true },
  { id: 'custom_3', label: '✨ close portrait', prompt: 'close portrait', framing: 'face' },
];

test('the label is the register emoji plus the head of the prompt', () => {
  assert.equal(customShotLabel('sitting on a bench'), '✨ sitting on a bench');
  assert.equal(customShotLabel('sitting on a bench', true), '🔞 sitting on a bench');
  const long = 'a'.repeat(CUSTOM_SHOT_PREVIEW_CHARS + 25);
  assert.equal(customShotLabel(long), `✨ ${'a'.repeat(CUSTOM_SHOT_PREVIEW_CHARS)}`);
});

test('an edit keeps the id, the position and the register, and re-derives the label', () => {
  const next = editCustomShot(SHOTS, 'custom_2', { prompt: 'kneeling on a rug', framing: 'bust' });
  assert.notEqual(next, SHOTS);                       // a real change = a new array
  assert.deepEqual(next.map((s) => s.id), ['custom_1', 'custom_2', 'custom_3']);
  const edited = next[1];
  assert.equal(edited.id, 'custom_2');                // selection and presets survive
  assert.equal(edited.prompt, 'kneeling on a rug');
  assert.equal(edited.framing, 'bust');
  assert.equal(edited.nsfw, true);                    // an edit re-words, it does not re-file
  assert.equal(edited.label, '🔞 kneeling on a rug'); // the label follows the prompt
  assert.deepEqual(next[0], SHOTS[0]);                // the neighbours are untouched
  assert.deepEqual(next[2], SHOTS[2]);
});

test('an edit that changes nothing usable returns the SAME array', () => {
  // Unknown id, empty prompt, whitespace-only prompt, and a no-op draft: four
  // ways the panel can call this on a keystroke, none of which may rerender.
  assert.equal(editCustomShot(SHOTS, 'nope', { prompt: 'x', framing: 'body' }), SHOTS);
  assert.equal(editCustomShot(SHOTS, 'custom_1', { prompt: '', framing: 'body' }), SHOTS);
  assert.equal(editCustomShot(SHOTS, 'custom_1', { prompt: '   ', framing: 'body' }), SHOTS);
  assert.equal(
    editCustomShot(SHOTS, 'custom_1', { prompt: 'sitting on a bench', framing: 'body' }), SHOTS);
});

test('an edit trims, and an unknown framing keeps the one already stored', () => {
  const next = editCustomShot(SHOTS, 'custom_1', { prompt: '  walking away  ', framing: 'sideways' });
  assert.equal(next[0].prompt, 'walking away');
  assert.equal(next[0].framing, 'body');
});

test('a fresh card carries the register, a trimmed prompt and a derived label', () => {
  const shot = createCustomShot({ prompt: '  on a motorbike ', framing: 'back', nsfw: true, id: 'custom_9' });
  assert.deepEqual(shot, {
    id: 'custom_9', label: '🔞 on a motorbike', prompt: 'on a motorbike', framing: 'back', nsfw: true,
  });
  // A safe card has no `nsfw` key at all — the stored shape predates the flag
  // and the preset validator only accepts `true` or absent.
  assert.equal('nsfw' in createCustomShot({ prompt: 'on a bench', framing: 'body' }), false);
  assert.equal(createCustomShot({ prompt: '   ', framing: 'body' }), null);
  assert.equal(createCustomShot({ prompt: 'x', framing: 'nonsense' }).framing, 'body');
});

test('the draft is what the editor opens with, or null when the card is gone', () => {
  assert.deepEqual(customShotDraft(SHOTS, 'custom_3'), { prompt: 'close portrait', framing: 'face' });
  assert.equal(customShotDraft(SHOTS, 'custom_404'), null);
  assert.equal(customShotDraft(null, 'custom_1'), null);
});

// Subject-type helper contract. Mirrors the backend list (face_variations
// .SUBJECT_TYPES) and locks the per-type framing relabels + default-preset pick.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  SUBJECT_TYPES, SUBJECT_TYPE_LABELS, SUBJECT_TYPE_HINTS,
  normalizeSubjectType, framingHeaders, framingLabel, defaultPresetKey,
} from './subjectTypes.js';

test('the five subject types match the backend and all have a label + hint', () => {
  assert.deepEqual(SUBJECT_TYPES, ['human', 'animal', 'creature', 'object', 'other']);
  for (const st of SUBJECT_TYPES) {
    assert.ok(SUBJECT_TYPE_LABELS[st], `label for ${st}`);
    assert.ok(SUBJECT_TYPE_HINTS[st], `hint for ${st}`);
  }
});

test('normalizeSubjectType falls back to human for anything unknown', () => {
  assert.equal(normalizeSubjectType('animal'), 'animal');
  assert.equal(normalizeSubjectType('person'), 'human');
  assert.equal(normalizeSubjectType(''), 'human');
  assert.equal(normalizeSubjectType(undefined), 'human');
});

test('every subject type relabels the four internal framings', () => {
  for (const st of SUBJECT_TYPES) {
    const h = framingHeaders(st);
    for (const fr of ['face', 'bust', 'body', 'back']) {
      assert.ok(h[fr], `${st}.${fr}`);
    }
  }
  // Human keeps the historical wording; animals never say "Bust".
  assert.equal(framingLabel('human', 'bust'), 'Bust');
  assert.equal(framingLabel('animal', 'face'), 'Head');
  assert.equal(framingLabel('animal', 'bust'), 'Half-body');
  assert.equal(framingLabel('object', 'body'), 'Full');
});

test('defaultPresetKey picks the right preset per subject type', () => {
  const human = { balanced_25: [], zimage_12: [], body_emphasis: [] };
  assert.equal(defaultPresetKey(human, 'human'), 'balanced_25');
  assert.equal(defaultPresetKey(human, 'human', { bodyFidelity: true }), 'body_emphasis');
  // Non-human: the single balanced preset (first key) is selected.
  assert.equal(defaultPresetKey({ animal_balanced: [] }, 'animal'), 'animal_balanced');
  assert.equal(defaultPresetKey({}, 'object'), null);
});

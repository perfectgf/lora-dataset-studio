import test from 'node:test';
import assert from 'node:assert/strict';
import {
  KIND_LABELS, normalizeKindLabel, kindSwitchSummary,
} from './datasetKindSwitch.js';

test('normalizeKindLabel folds unknown/empty to character (mirrors the server)', () => {
  assert.equal(normalizeKindLabel('style'), 'style');
  assert.equal(normalizeKindLabel('Concept'), 'concept');
  assert.equal(normalizeKindLabel('character'), 'character');
  assert.equal(normalizeKindLabel(''), 'character');
  assert.equal(normalizeKindLabel(null), 'character');
  assert.equal(normalizeKindLabel('slider'), 'character');
});

test('every kind has a label', () => {
  assert.deepEqual(Object.keys(KIND_LABELS).sort(), ['character', 'concept', 'style']);
});

test('no-op switch returns null', () => {
  assert.equal(kindSwitchSummary('character', 'character'), null);
  assert.equal(kindSwitchSummary('', 'character'), null);       // both fold to character
  assert.equal(kindSwitchSummary('concept', 'Concept'), null);  // case-insensitive
});

test('character → style: hides build panels, drops the activation trigger + fidelity', () => {
  const s = kindSwitchSummary('character', 'style');
  assert.equal(s.from, 'character');
  assert.equal(s.to, 'style');
  const joined = s.changes.join(' ');
  assert.match(joined, /aesthetic\/style unspoken/);            // caption strategy
  assert.match(joined, /hidden/);                                // reference/generate gone
  assert.match(joined, /No activation trigger/);                 // trigger role
  assert.match(joined, /fidelity setting no longer applies/);    // fidelity cleared
  // Nothing is required and nothing is deleted.
  assert.ok(s.preserved.some((p) => /stay exactly as they are/.test(p)));
});

test('character → concept: requires a concept description, no style-trigger copy', () => {
  const s = kindSwitchSummary('character', 'concept');
  const joined = s.changes.join(' ');
  assert.match(joined, /concept description is required/);
  assert.match(joined, /recurring concept unspoken/);
  assert.doesNotMatch(joined, /No activation trigger/);          // concept still has one
  assert.ok(s.preserved.some((p) => /concept description is remembered/.test(p)));
});

test('style → character: trigger field returns, build panels come back', () => {
  const s = kindSwitchSummary('style', 'character');
  const joined = s.changes.join(' ');
  assert.match(joined, /trigger word returns/);
  assert.match(joined, /become available/);
  assert.match(joined, /identity/);                              // caption strategy
});

test('recaption nudge tracks whether captions exist', () => {
  assert.equal(kindSwitchSummary('character', 'style', { hasCaptions: true }).recaption, true);
  assert.equal(kindSwitchSummary('character', 'style', { hasCaptions: false }).recaption, false);
  assert.equal(kindSwitchSummary('character', 'style').recaption, false);
});

test('run history is always flagged as preserved (runs are keyed by family+trigger, not kind)', () => {
  for (const [a, b] of [['character', 'concept'], ['concept', 'style'], ['style', 'character']]) {
    const s = kindSwitchSummary(a, b);
    assert.ok(s.preserved.some((p) => /training runs and checkpoints keep their identity/.test(p)),
      `${a}→${b} should preserve run identity`);
  }
});

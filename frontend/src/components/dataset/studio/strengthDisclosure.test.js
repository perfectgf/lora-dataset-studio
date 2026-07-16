import test from 'node:test';
import assert from 'node:assert/strict';
import { BASE_STRENGTH_MAX, hasExtendedSelection } from './strengthDisclosure.js';
import { STRENGTH_CHOICES, STRENGTH_CHOICES_EXTENDED } from './constants.js';

test('extended strength choices reach 4.0, are all above the base range, sorted, no overlap', () => {
  assert.equal(BASE_STRENGTH_MAX, 2.0);
  // Base row tops out at exactly 2.0.
  assert.equal(Math.max(...STRENGTH_CHOICES), 2.0);
  assert.ok(STRENGTH_CHOICES.every((s) => s <= BASE_STRENGTH_MAX));
  // Extended row is strictly above the base max and reaches the 4.0 server ceiling.
  assert.ok(STRENGTH_CHOICES_EXTENDED.every((s) => s > BASE_STRENGTH_MAX));
  assert.equal(Math.max(...STRENGTH_CHOICES_EXTENDED), 4.0);
  // Ascending, no duplicate between the two rows.
  const sorted = [...STRENGTH_CHOICES_EXTENDED].sort((a, b) => a - b);
  assert.deepEqual(STRENGTH_CHOICES_EXTENDED, sorted);
  const base = new Set(STRENGTH_CHOICES);
  assert.ok(STRENGTH_CHOICES_EXTENDED.every((s) => !base.has(s)));
});

test('hasExtendedSelection is false for base-only selections (extended row stays collapsible)', () => {
  assert.equal(hasExtendedSelection([]), false);
  assert.equal(hasExtendedSelection(null), false);
  assert.equal(hasExtendedSelection(undefined), false);
  assert.equal(hasExtendedSelection([0, 0.7, 1.0]), false);
  assert.equal(hasExtendedSelection([2.0]), false);   // 2.0 is the top base chip, not extended
});

test('hasExtendedSelection force-opens the extended row when an above-2.0 value is selected', () => {
  assert.equal(hasExtendedSelection([2.25]), true);
  assert.equal(hasExtendedSelection([0.7, 1.0, 3.5]), true);   // reloaded recent prompt w/ extended
  assert.equal(hasExtendedSelection([4.0]), true);
  // Robust to a persisted off-grid value above the base ceiling (never hide it).
  assert.equal(hasExtendedSelection([2.1]), true);
});

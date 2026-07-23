import test from 'node:test';
import assert from 'node:assert/strict';

import { extraRefCropSource } from './extraRefs.js';

const EXTRAS = ['local_datasetrefx_aa.webp', 'local_datasetrefx_bb.webp'];
const SOURCES = ['local_datasetrefxorig_aa.webp', 'local_datasetrefxorig_bb.webp'];

test('opens the kept full-frame original of the right extra', () => {
  assert.equal(extraRefCropSource(EXTRAS, SOURCES, EXTRAS[1]), SOURCES[1]);
});

test('falls back to the extra itself when no original is kept for it', () => {
  // Legacy extra: the backend sends the extra's own name as its crop source.
  assert.equal(extraRefCropSource(EXTRAS, [EXTRAS[0], SOURCES[1]], EXTRAS[0]), EXTRAS[0]);
  // Payload predating the field at all (older tab, cached response).
  assert.equal(extraRefCropSource(EXTRAS, undefined, EXTRAS[0]), EXTRAS[0]);
  assert.equal(extraRefCropSource(EXTRAS, [null, null], EXTRAS[0]), EXTRAS[0]);
});

test('returns null for anything that is not one of this dataset extras', () => {
  // The caller opens NO editor rather than guessing a path — the backend applies
  // the same membership rule as the authoritative guard.
  assert.equal(extraRefCropSource(EXTRAS, SOURCES, '../ref.webp'), null);
  assert.equal(extraRefCropSource(EXTRAS, SOURCES, undefined), null);
  assert.equal(extraRefCropSource(undefined, undefined, EXTRAS[0]), null);
});

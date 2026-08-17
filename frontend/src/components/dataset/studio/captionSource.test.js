import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BANK, DATASET, captionSourceBody, emptyDrawMessage, lockedLabel,
  normaliseSource, sourceMeta,
} from './captionSource.js';

/* --- the stored choice, which must survive the arrival of banks ----------- */

test('a choice stored BEFORE banks existed still reads as a dataset', () => {
  // studioCaptionDataset_v1 holds {id, name} and no kind. Reading that as a
  // bank would repoint a locked choice at whatever bank shares the number.
  const legacy = normaliseSource({ id: 7, name: 'portraits' });
  assert.equal(legacy.kind, DATASET);
  assert.deepEqual(captionSourceBody(legacy), { dataset_id: 7 });
});

test('an unknown kind falls back to dataset rather than guessing', () => {
  assert.equal(normaliseSource({ id: 1, name: 'x', kind: 'video' }).kind, DATASET);
  assert.equal(normaliseSource({ id: 1, name: 'x', kind: BANK }).kind, BANK);
});

test('junk is refused instead of producing a source that cannot be drawn from', () => {
  for (const raw of [null, undefined, {}, { id: 0 }, { id: -3 }, { id: 1.5 }, { id: 'x' }]) {
    assert.equal(normaliseSource(raw), null);
    assert.equal(captionSourceBody(raw), null);
  }
});

test('a nameless source still gets a name, and it names its kind', () => {
  assert.equal(normaliseSource({ id: 4 }).name, 'Dataset #4');
  assert.equal(normaliseSource({ id: 4, kind: BANK }).name, 'Bank #4');
  assert.equal(normaliseSource({ id: 4, name: '   ' }).name, 'Dataset #4');
});

/* --- the request, whose dataset half must not have moved ----------------- */

test('the dataset request is byte-identical to the one that shipped before', () => {
  assert.deepEqual(captionSourceBody({ id: 12, name: 'a', kind: DATASET }), { dataset_id: 12 });
  // Exactly one key: sending both is what the route refuses outright.
  assert.deepEqual(Object.keys(captionSourceBody({ id: 12, name: 'a' })), ['dataset_id']);
});

test('a bank asks for a bank, and never for both', () => {
  const body = captionSourceBody({ id: 3, name: 'dump', kind: BANK });
  assert.deepEqual(body, { bank_id: 3 });
  assert.deepEqual(Object.keys(body), ['bank_id']);
});

/* --- what the picker and the chip say ------------------------------------ */

test('a bank row counts what is KEPT — the only pile the draw reads', () => {
  assert.equal(sourceMeta({ keep: 40, total: 900 }, BANK), '40 kept · of 900');
  // A dataset keeps its existing wording, unchanged.
  assert.equal(sourceMeta({ kind: 'character', images_total: 50 }, DATASET),
    'character · 50 images');
  assert.equal(sourceMeta({ kind: 'style', images_total: 1 }, DATASET), 'style · 1 image');
  assert.equal(sourceMeta({}, DATASET), '');
});

test('the locked chip names the KIND, because the names collide in practice', () => {
  // A bank routinely shares its name with the dataset it promotes into.
  assert.equal(lockedLabel({ id: 1, name: 'Elise' }), 'Dataset: Elise');
  assert.equal(lockedLabel({ id: 2, name: 'Elise', kind: BANK }), 'Bank: Elise');
  assert.equal(lockedLabel(null), '');
});

test('an empty draw is explained in the source own noun', () => {
  assert.match(emptyDrawMessage({ id: 1, name: 'a', kind: BANK }), /This bank/);
  assert.match(emptyDrawMessage({ id: 1, name: 'a' }), /This dataset/);
});

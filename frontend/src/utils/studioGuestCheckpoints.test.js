import assert from 'node:assert/strict';
import test from 'node:test';
import {
  MAX_GUEST_CHECKPOINTS, GUEST_LABEL_PREFIX, guestLabel, guestStem,
  normalizeGuestCheckpoints, chosenCheckpoints, addGuestCheckpoint,
  removeGuestCheckpoint,
} from './studioGuestCheckpoints.js';

test('guestLabel prefixes the basename, not the folder', () => {
  assert.equal(guestStem('krea\\foo.safetensors'), 'foo');
  assert.equal(guestLabel('other-char.safetensors'), `${GUEST_LABEL_PREFIX}other-char`);
});

test('normalizeGuestCheckpoints dedupes, drops empties, caps at 16', () => {
  assert.deepEqual(normalizeGuestCheckpoints(null), []);
  assert.deepEqual(normalizeGuestCheckpoints([
    { filename: 'a.safetensors' },
    { filename: 'a.safetensors' },
    { filename: '' },
    'b.safetensors',
  ]), [
    { filename: 'a.safetensors', label: `${GUEST_LABEL_PREFIX}a` },
    { filename: 'b.safetensors', label: `${GUEST_LABEL_PREFIX}b` },
  ]);
  const many = Array.from({ length: 40 }, (_, i) => ({ filename: `g${i}.safetensors` }));
  assert.equal(normalizeGuestCheckpoints(many).length, MAX_GUEST_CHECKPOINTS);
});

test('chosenCheckpoints keeps mine and guests independent; canvas pin wins', () => {
  const mine = ['z image\\lora_a_000002000.safetensors'];
  const guests = [{ filename: 'theirs.safetensors', label: `${GUEST_LABEL_PREFIX}theirs` }];
  assert.deepEqual(chosenCheckpoints({
    mineFns: mine, selCps: null, guests, selGuests: null,
  }), [...mine, 'theirs.safetensors']);
  assert.deepEqual(chosenCheckpoints({
    mineFns: mine, selCps: [], guests, selGuests: ['theirs.safetensors'],
  }), ['theirs.safetensors']);
  assert.deepEqual(chosenCheckpoints({
    mineFns: mine, selCps: mine, guests, selGuests: [],
  }), mine);
  assert.deepEqual(chosenCheckpoints({
    mineFns: mine, selCps: null, guests, selGuests: null, pinned: ['x'],
  }), ['x']);
});

test('addGuestCheckpoint refuses mine, dupes and the cap', () => {
  const mine = ['mine.safetensors'];
  assert.deepEqual(addGuestCheckpoint([], 'mine.safetensors', mine), []);
  const one = addGuestCheckpoint([], 'a.safetensors', mine);
  assert.equal(one.length, 1);
  assert.deepEqual(addGuestCheckpoint(one, 'a.safetensors', mine), one);
  const full = Array.from({ length: MAX_GUEST_CHECKPOINTS }, (_, i) => (
    { filename: `g${i}.safetensors`, label: `x${i}` }));
  assert.equal(addGuestCheckpoint(full, 'overflow.safetensors').length, MAX_GUEST_CHECKPOINTS);
});

test('removeGuestCheckpoint drops one file', () => {
  const cur = [
    { filename: 'a.safetensors', label: 'A' },
    { filename: 'b.safetensors', label: 'B' },
  ];
  assert.deepEqual(removeGuestCheckpoint(cur, 'a.safetensors'), [cur[1]]);
});

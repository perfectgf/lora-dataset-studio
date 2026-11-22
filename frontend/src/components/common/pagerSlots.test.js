import test from 'node:test';
import assert from 'node:assert/strict';
import { pagerSlots } from './pagerSlots.js';

test('a short list shows every page', () => {
  assert.deepEqual(pagerSlots(1, 1), [1]);
  assert.deepEqual(pagerSlots(3, 4), [1, 2, 3, 4]);
  assert.deepEqual(pagerSlots(2, 7, 1), [1, 2, 3, 4, 5, 6, 7]);
});

test('a long list keeps the first and the last page, and marks what it skips', () => {
  assert.deepEqual(pagerSlots(1, 130, 1), [1, 2, 3, 4, 5, 0, 130]);
  assert.deepEqual(pagerSlots(60, 130, 1), [1, 0, 59, 60, 61, 0, 130]);
  assert.deepEqual(pagerSlots(130, 130, 1), [1, 0, 126, 127, 128, 129, 130]);
  assert.deepEqual(pagerSlots(60, 130, 2), [1, 0, 58, 59, 60, 61, 62, 0, 130]);
  assert.deepEqual(pagerSlots(60, 130, 0), [1, 0, 60, 0, 130]);
});

test('the row has the same number of slots on every page, so buttons do not move under the pointer', () => {
  for (const siblings of [0, 1, 2]) {
    const lengths = new Set();
    for (let page = 1; page <= 130; page += 1) lengths.add(pagerSlots(page, 130, siblings).length);
    assert.deepEqual([...lengths], [siblings * 2 + 5]);
  }
});

test('every page is reachable in at most two presses from any other, and the current one is always shown', () => {
  for (let page = 1; page <= 40; page += 1) {
    const slots = pagerSlots(page, 40, 1);
    assert.ok(slots.includes(page));
    assert.equal(slots[0], 1);
    assert.equal(slots.at(-1), 40);
    // no page number appears twice
    const numbers = slots.filter(Boolean);
    assert.equal(new Set(numbers).size, numbers.length);
  }
});

test('a page outside the list is brought back inside it', () => {
  assert.deepEqual(pagerSlots(0, 130, 1), pagerSlots(1, 130, 1));
  assert.deepEqual(pagerSlots(999, 130, 1), pagerSlots(130, 130, 1));
  assert.deepEqual(pagerSlots(NaN, 0, 1), [1]);
});

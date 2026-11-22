import test from 'node:test';
import assert from 'node:assert/strict';
import { GRID_PAGE_SIZE, clampPage, pageCount, pageSlice } from './gridPaging.js';

const list = (n) => Array.from({ length: n }, (_, i) => ({ id: i + 1 }));

test('an empty list is one empty page, not zero pages', () => {
  assert.equal(pageCount(0), 1);
  const view = pageSlice([], 0);
  assert.deepEqual(view.items, []);
  assert.equal(view.pages, 1);
  assert.equal(view.from, 0);
  assert.equal(view.to, 0);
  assert.equal(view.paged, false);
});

test('a list that fits on one page hides the pager', () => {
  const view = pageSlice(list(GRID_PAGE_SIZE), 0);
  assert.equal(view.paged, false);
  assert.equal(view.items.length, GRID_PAGE_SIZE);
  assert.equal(view.pages, 1);
});

test('one image over the page size starts paging', () => {
  const view = pageSlice(list(GRID_PAGE_SIZE + 1), 0);
  assert.equal(view.paged, true);
  assert.equal(view.pages, 2);
});

test('pages slice the list in order, with a 1-based inclusive label', () => {
  const images = list(6211);
  const first = pageSlice(images, 0, 500);
  assert.equal(first.items.length, 500);
  assert.equal(first.items[0].id, 1);
  assert.equal(first.items[499].id, 500);
  assert.equal(first.from, 1);
  assert.equal(first.to, 500);

  const third = pageSlice(images, 2, 500);
  assert.equal(third.items[0].id, 1001);
  assert.equal(third.from, 1001);
  assert.equal(third.to, 1500);

  const last = pageSlice(images, 12, 500);
  assert.equal(last.items.length, 211);
  assert.equal(last.from, 6001);
  assert.equal(last.to, 6211);
  assert.equal(last.pages, 13);
});

test('every image appears exactly once across the pages', () => {
  const images = list(1234);
  const seen = [];
  for (let p = 0; p < pageCount(1234, 100); p++) {
    seen.push(...pageSlice(images, p, 100).items.map((i) => i.id));
  }
  assert.equal(seen.length, 1234);
  assert.deepEqual(seen, images.map((i) => i.id));
});

test('a page past the end lands on the last real page, never on a blank grid', () => {
  // A filter or a delete shrinks the list under the page the user stands on.
  const view = pageSlice(list(120), 9, 100);
  assert.equal(view.page, 1);
  assert.equal(view.items.length, 20);
  assert.equal(clampPage(9, 120, 100), 1);
  assert.equal(clampPage(-3, 120, 100), 0);
});

test('clampPage survives the values a shrinking list produces', () => {
  assert.equal(clampPage(0, 0), 0);
  assert.equal(clampPage(5, 0), 0);
  assert.equal(clampPage(NaN, 6211), 0);
  assert.equal(clampPage(undefined, 6211), 0);
  assert.equal(clampPage(2.7, 6211, 500), 2);
});

test('a non-array list degrades to an empty page instead of throwing', () => {
  const view = pageSlice(undefined, 3);
  assert.deepEqual(view.items, []);
  assert.equal(view.total, 0);
  assert.equal(view.page, 0);
});

test('a page stays under the smooth ceiling and ends on a whole row', () => {
  // The ceiling is a perf fact, not a cosmetic one: 6 211 tiles rendered at once
  // measured ~41 ms/frame, 500 and under measured 6 ms. Below it the size is
  // chosen for the reader — a page of a few screens — and must divide by every
  // column count the grid uses (S/M/L at each breakpoint), or a full page ends
  // on a ragged row that looks like the end of the set.
  assert.ok(GRID_PAGE_SIZE <= 500);
  assert.ok(GRID_PAGE_SIZE >= 24, 'a page smaller than this turns review into page-turning');
  for (const columns of [2, 3, 4, 6, 8]) {
    assert.equal(GRID_PAGE_SIZE % columns, 0, `${GRID_PAGE_SIZE} tiles do not fill rows of ${columns}`);
  }
});

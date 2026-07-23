import test from 'node:test';
import assert from 'node:assert/strict';

import {
  GRID_STATUS_FILTERS,
  DEFAULT_GRID_STATUS_FILTER,
  isGridStatusFilter,
  normalizeGridStatusFilter,
  filterImagesByStatus,
  gridStatusFilterCounts,
} from './gridStatusFilter.js';
import { filterImages } from './tagFilter.js';
import { isSmallImageRescueRow } from './smallImageRescue.js';

const img = (over) => ({ id: over.id, status: 'pending', filename: `${over.id}.png`, ...over });

// A small dataset that mirrors the real mix: decided + undecided + Klein improve
// candidates + one unresolved rescue pair.
const IMAGES = [
  img({ id: 1, status: 'keep', caption: 'smile, portrait' }),
  img({ id: 2, status: 'keep', caption: 'standing' }),
  img({ id: 3, status: 'reject', caption: 'blurry' }),
  img({ id: 4, status: 'failed', filename: null }),
  img({ id: 5, status: 'pending', caption: 'smile, closeup' }),          // awaiting ✓/✕
  img({ id: 6, status: 'pending', caption: 'closeup' }),                 // awaiting ✓/✕
  img({ id: 7, status: 'pending', filename: null }),                     // still generating
  img({ id: 8, status: 'pending', derivation_kind: 'klein_image_improve', parent_image_id: 1 }),
  img({ id: 9, status: 'pending', derivation_kind: 'klein_small_image' }),   // unresolved pair
  img({ id: 10, status: 'pending', derivation_kind: 'small_image_source' }), // unresolved pair
];
const UNRESOLVED = new Set([9, 10]);
const ids = (list) => list.map((i) => i.id);

test('the filter ids are stable and "all" is the default', () => {
  assert.deepEqual(GRID_STATUS_FILTERS.map((f) => f.id),
    ['all', 'undecided', 'kept', 'rejected', 'improve']);
  assert.equal(DEFAULT_GRID_STATUS_FILTER, 'all');
  assert.ok(isGridStatusFilter('undecided'));
  assert.ok(!isGridStatusFilter('pending'));
  assert.equal(normalizeGridStatusFilter('nope'), 'all');   // stale localStorage
  assert.equal(normalizeGridStatusFilter(undefined), 'all');
  assert.equal(normalizeGridStatusFilter('kept'), 'kept');
});

test('"all" is a no-op that keeps the very same array', () => {
  assert.equal(filterImagesByStatus(IMAGES, 'all', { unresolvedRescueIds: UNRESOLVED }), IMAGES);
});

test('Undecided keeps only the pending images that HAVE a file, rescue pairs excluded', () => {
  const out = filterImagesByStatus(IMAGES, 'undecided', { unresolvedRescueIds: UNRESOLVED });
  // 7 has no file yet (still generating), 9/10 are an unresolved rescue pair,
  // 8 is a pending improve candidate WITH a file → it counts as awaiting ✓/✕.
  assert.deepEqual(ids(out), [5, 6, 8]);
});

test('Undecided matches the workspace "awaiting ✓/✕" count exactly', () => {
  const badge = IMAGES.filter((i) => i.status === 'pending' && i.filename
    && !UNRESOLVED.has(i.id)).length;
  assert.equal(filterImagesByStatus(IMAGES, 'undecided', { unresolvedRescueIds: UNRESOLVED }).length,
    badge);
});

test('Kept / Rejected / Improve candidates each isolate their own subset', () => {
  const opts = { unresolvedRescueIds: UNRESOLVED };
  assert.deepEqual(ids(filterImagesByStatus(IMAGES, 'kept', opts)), [1, 2]);
  assert.deepEqual(ids(filterImagesByStatus(IMAGES, 'rejected', opts)), [3, 4]);
  assert.deepEqual(ids(filterImagesByStatus(IMAGES, 'improve', opts)), [8]);
});

test('without the unresolved set, rescue rows are held back rather than guessed', () => {
  assert.deepEqual(ids(filterImagesByStatus(IMAGES, 'undecided')), [5, 6, 8]);
});

test('the decision filter composes with the caption tag filter', () => {
  const undecided = filterImagesByStatus(IMAGES, 'undecided', { unresolvedRescueIds: UNRESOLVED });
  const both = filterImages(undecided, { includes: ['smile'], mode: 'booru' });
  assert.deepEqual(ids(both), [5]);
  // Order must not matter: tags first then decisions gives the same set.
  const reversed = filterImagesByStatus(
    filterImages(IMAGES, { includes: ['smile'], mode: 'booru' }),
    'undecided', { unresolvedRescueIds: UNRESOLVED },
  );
  assert.deepEqual(ids(reversed), ids(both));
});

test('select all follows the filtered subset (same rule DatasetGrid applies)', () => {
  const selectable = (list) => list.filter((i) => i.filename && !isSmallImageRescueRow(i));
  assert.equal(selectable(IMAGES).length, 6);   // "select all (6)" without a filter
  const undecided = filterImagesByStatus(IMAGES, 'undecided', { unresolvedRescueIds: UNRESOLVED });
  assert.equal(selectable(undecided).length, 3);
  const improve = filterImagesByStatus(IMAGES, 'improve', { unresolvedRescueIds: UNRESOLVED });
  assert.equal(selectable(improve).length, 1);
});

test('counts cover every entry and "all" is the total', () => {
  const counts = gridStatusFilterCounts(IMAGES, { unresolvedRescueIds: UNRESOLVED });
  assert.deepEqual(counts, { all: 10, undecided: 3, kept: 2, rejected: 2, improve: 1 });
});

test('empty / missing input never throws', () => {
  assert.deepEqual(filterImagesByStatus(null, 'kept'), []);
  assert.deepEqual(filterImagesByStatus(undefined, 'all'), []);
  assert.deepEqual(filterImagesByStatus([], 'improve'), []);
});

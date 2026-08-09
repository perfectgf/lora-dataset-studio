import test from 'node:test';
import assert from 'node:assert/strict';
import { matchesDatasetQuery, statusLabel } from './canvasFilterBar.js';

test('every status the board can report has a name of its own', () => {
  assert.equal(statusLabel('active'), 'Active');
  assert.equal(statusLabel('completed'), 'Completed');
  assert.equal(statusLabel('error'), 'Errors');
  // A run whose lineage could not be read is a filterable state, not a gap —
  // calling it "Other" would hide that it is a real thing on the board.
  assert.equal(statusLabel('unknown'), 'Unknown');
  assert.equal(statusLabel(undefined), 'Unknown');
});

const DS = { id: 1, name: 'Ada Portrait', families: ['krea', 'zimage'] };

test('an empty query shows the whole list rather than hiding it', () => {
  // A picker that shows nothing until you type is a list you cannot browse.
  assert.equal(matchesDatasetQuery(DS, ''), true);
  assert.equal(matchesDatasetQuery(DS, '   '), true);
  assert.equal(matchesDatasetQuery(DS, null), true);
});

test('the picker matches on the name, case-insensitively and mid-word', () => {
  assert.equal(matchesDatasetQuery(DS, 'ada'), true);
  assert.equal(matchesDatasetQuery(DS, 'PORTRAIT'), true);
  assert.equal(matchesDatasetQuery(DS, 'trait'), true);
  assert.equal(matchesDatasetQuery(DS, 'bea'), false);
});

test('…and on the model family, which is what the old three-column list answered by eye', () => {
  assert.equal(matchesDatasetQuery(DS, 'krea'), true);
  assert.equal(matchesDatasetQuery(DS, 'zim'), true);
  assert.equal(matchesDatasetQuery({ id: 2, name: 'Bea' }, 'krea'), false);
});

test('a dataset with no name or no families never throws', () => {
  assert.equal(matchesDatasetQuery({}, 'x'), false);
  assert.equal(matchesDatasetQuery(null, 'x'), false);
  assert.equal(matchesDatasetQuery(null, ''), true);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  TRAINED, NOT_TRAINED, normalizeTileSize, normalizeCollapsedMap,
  datasetKind, datasetMatches, kindsPresent, isTrained, groupDatasets,
} from './datasetLibrary.js';

test('tile size preference only accepts S/M/L and defaults to M', () => {
  assert.equal(normalizeTileSize('S'), 'S');
  assert.equal(normalizeTileSize('M'), 'M');
  assert.equal(normalizeTileSize('L'), 'L');
  assert.equal(normalizeTileSize('XL'), 'M');
  assert.equal(normalizeTileSize(null), 'M');
  assert.equal(normalizeTileSize(undefined), 'M');
});

test('collapsed-sections map survives malformed storage payloads', () => {
  assert.deepEqual(normalizeCollapsedMap('{"sdxl":1}'), { sdxl: 1 });
  assert.deepEqual(normalizeCollapsedMap(''), {});
  assert.deepEqual(normalizeCollapsedMap(null), {});
  assert.deepEqual(normalizeCollapsedMap('not json'), {});
  assert.deepEqual(normalizeCollapsedMap('[1,2]'), {});
  assert.deepEqual(normalizeCollapsedMap('"str"'), {});
});

test('dataset kind applies the server default (missing/empty → character)', () => {
  assert.equal(datasetKind({ kind: 'style' }), 'style');
  assert.equal(datasetKind({ kind: 'Concept' }), 'concept');
  assert.equal(datasetKind({ kind: '' }), 'character');
  assert.equal(datasetKind({}), 'character');
});

test('search matches name or trigger word, case-insensitive', () => {
  const d = { name: 'Emma', trigger_word: 'zchar_emma' };
  assert.equal(datasetMatches(d, 'emm'), true);
  assert.equal(datasetMatches(d, 'ZCHAR'), true);
  assert.equal(datasetMatches(d, '  emma  '), true);
  assert.equal(datasetMatches(d, 'nope'), false);
  assert.equal(datasetMatches(d, ''), true);
  assert.equal(datasetMatches({ name: null, trigger_word: null }, 'x'), false);
});

test('kind chip filter composes with search', () => {
  const d = { name: 'Ink', trigger_word: '', kind: 'style' };
  assert.equal(datasetMatches(d, '', 'style'), true);
  assert.equal(datasetMatches(d, '', 'character'), false);
  assert.equal(datasetMatches(d, 'ink', 'style'), true);
  assert.equal(datasetMatches(d, 'nope', 'style'), false);
  // 'all' disables the chip filter entirely.
  assert.equal(datasetMatches(d, '', 'all'), true);
});

test('kindsPresent lists kinds in canonical order (chips need >= 2)', () => {
  assert.deepEqual(kindsPresent([]), []);
  assert.deepEqual(kindsPresent([{ kind: 'style' }, { kind: '' }, { kind: 'style' }]),
    ['character', 'style']);
  assert.deepEqual(kindsPresent([{ kind: 'style' }, { kind: 'concept' }, { kind: 'character' }]),
    ['character', 'concept', 'style']);
});

test('groupDatasets splits into Trained then Not trained yet, keeping input order', () => {
  const groups = groupDatasets([
    { id: 1, trained_families: ['sdxl'] },
    { id: 2, trained_families: [] },
    { id: 3, trained_families: ['zimage'] },
    { id: 4 },   // missing trained_families → treated as none
  ]);
  assert.deepEqual(groups.map((g) => g.family), [TRAINED[0], NOT_TRAINED[0]]);
  assert.deepEqual(groups[0].items.map((d) => d.id), [1, 3]);
  assert.deepEqual(groups[1].items.map((d) => d.id), [2, 4]);
  assert.equal(groups[0].label, TRAINED[1]);
  assert.equal(groups[0].emoji, TRAINED[2]);
  assert.equal(groups.at(-1).label, NOT_TRAINED[1]);
  assert.equal(groups.at(-1).emoji, NOT_TRAINED[2]);
});

test('one section per dataset: multi-family and unknown families are still just "Trained"', () => {
  // The family detail lives in the per-tile badges, not in the sections.
  const groups = groupDatasets([
    { id: 1, trained_families: ['flux', 'zimage'] },
    { id: 2, trained_families: ['future-family'] },
  ]);
  assert.deepEqual(groups.map((g) => g.family), [TRAINED[0]]);
  assert.deepEqual(groups[0].items.map((d) => d.id), [1, 2]);
});

test('grouping ignores train_type entirely — only real training runs place a dataset', () => {
  const groups = groupDatasets([
    { id: 1, train_type: 'sdxl', trained_families: ['zimage'] },
    { id: 2, train_type: 'flux', trained_families: [] },
  ]);
  assert.deepEqual(groups.map((g) => g.family), [TRAINED[0], NOT_TRAINED[0]]);
  assert.deepEqual(groups[0].items.map((d) => d.id), [1]);
  assert.deepEqual(groups.at(-1).items.map((d) => d.id), [2]);
});

test('empty sections are dropped (all-trained / none-trained libraries)', () => {
  assert.deepEqual(groupDatasets([{ id: 1, trained_families: ['sdxl'] }])
    .map((g) => g.family), [TRAINED[0]]);
  assert.deepEqual(groupDatasets([{ id: 1, trained_families: [] }])
    .map((g) => g.family), [NOT_TRAINED[0]]);
});

test('isTrained: any real trained family counts, absence/malformed does not', () => {
  assert.equal(isTrained({ trained_families: ['sdxl'] }), true);
  assert.equal(isTrained({ trained_families: ['future-family'] }), true);
  assert.equal(isTrained({ trained_families: [] }), false);
  assert.equal(isTrained({}), false);
  assert.equal(isTrained(null), false);
});

test('groupDatasets on an empty library returns no sections', () => {
  assert.deepEqual(groupDatasets([]), []);
});

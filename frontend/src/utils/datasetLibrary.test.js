import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FAMILY_ORDER, OTHER_FAMILY, normalizeTileSize, normalizeCollapsedMap,
  datasetKind, datasetMatches, kindsPresent, groupDatasets,
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

test('groupDatasets keeps FAMILY_ORDER, drops empty sections, defaults zimage', () => {
  const ds = [
    { id: 1, train_type: 'sdxl' },
    { id: 2, train_type: 'zimage' },
    { id: 3 },                       // missing train_type → zimage
    { id: 4, train_type: 'sdxl' },
  ];
  const groups = groupDatasets(ds);
  assert.deepEqual(groups.map((g) => g.family), ['zimage', 'sdxl']);
  assert.deepEqual(groups[0].items.map((d) => d.id), [2, 3]);
  assert.deepEqual(groups[1].items.map((d) => d.id), [1, 4]);
  // Section metadata comes straight from FAMILY_ORDER (label + emoji).
  const sdxl = FAMILY_ORDER.find(([fam]) => fam === 'sdxl');
  assert.equal(groups[1].label, sdxl[1]);
  assert.equal(groups[1].emoji, sdxl[2]);
});

test('unknown train_type lands in a trailing Other section, never vanishes', () => {
  const groups = groupDatasets([
    { id: 1, train_type: 'zimage' },
    { id: 2, train_type: 'future-family' },
  ]);
  assert.deepEqual(groups.map((g) => g.family), ['zimage', OTHER_FAMILY[0]]);
  assert.deepEqual(groups[1].items.map((d) => d.id), [2]);
  assert.equal(groups.at(-1).label, OTHER_FAMILY[1]);
});

test('groupDatasets on an empty library returns no sections', () => {
  assert.deepEqual(groupDatasets([]), []);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FAMILY_ORDER, OTHER_FAMILY, NOT_TRAINED, normalizeTileSize, normalizeCollapsedMap,
  datasetKind, datasetMatches, kindsPresent, primaryFamily, groupDatasets,
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

test('groupDatasets sections by primary trained family, in FAMILY_ORDER, dropping empties', () => {
  const ds = [
    { id: 1, trained_families: ['sdxl'] },
    { id: 2, trained_families: ['zimage'] },
    { id: 3, trained_families: ['sdxl'] },
  ];
  const groups = groupDatasets(ds);
  assert.deepEqual(groups.map((g) => g.family), ['zimage', 'sdxl']);
  assert.deepEqual(groups[0].items.map((d) => d.id), [2]);
  assert.deepEqual(groups[1].items.map((d) => d.id), [1, 3]); // input order kept
  // Section metadata comes straight from FAMILY_ORDER (label + emoji).
  const sdxl = FAMILY_ORDER.find(([fam]) => fam === 'sdxl');
  assert.equal(groups[1].label, sdxl[1]);
  assert.equal(groups[1].emoji, sdxl[2]);
});

test('a multi-family dataset lands in ONE section: primary by FAMILY_ORDER, not array order', () => {
  // The server returns trained_families sorted alphabetically (['flux','zimage']);
  // FAMILY_ORDER puts zimage first, so the section is zimage even though flux is
  // first in the array. The secondary family stays a badge — no second section.
  const groups = groupDatasets([{ id: 1, trained_families: ['flux', 'zimage'] }]);
  assert.deepEqual(groups.map((g) => g.family), ['zimage']);
  assert.deepEqual(groups[0].items.map((d) => d.id), [1]);
  assert.equal(groups.length, 1);
});

test('a dataset with no trained family lands in the trailing "Not trained yet" section', () => {
  const groups = groupDatasets([
    { id: 1, trained_families: ['zimage'] },
    { id: 2, trained_families: [] },
    { id: 3 },   // missing trained_families → treated as none
  ]);
  assert.deepEqual(groups.map((g) => g.family), ['zimage', NOT_TRAINED[0]]);
  assert.deepEqual(groups.at(-1).items.map((d) => d.id), [2, 3]);
  assert.equal(groups.at(-1).label, NOT_TRAINED[1]);
  assert.equal(groups.at(-1).emoji, NOT_TRAINED[2]);
});

test('grouping ignores train_type entirely — only real training runs place a dataset', () => {
  // train_type says sdxl but the only trained LoRA is zimage → zimage section;
  // an untrained dataset stays in "Not trained yet" whatever its train_type is.
  const groups = groupDatasets([
    { id: 1, train_type: 'sdxl', trained_families: ['zimage'] },
    { id: 2, train_type: 'flux', trained_families: [] },
  ]);
  assert.deepEqual(groups.map((g) => g.family), ['zimage', NOT_TRAINED[0]]);
  assert.deepEqual(groups[0].items.map((d) => d.id), [1]);
  assert.deepEqual(groups.at(-1).items.map((d) => d.id), [2]);
});

test('trained only in a family this build does not know → Other, before Not trained yet', () => {
  const groups = groupDatasets([
    { id: 1, trained_families: ['zimage'] },
    { id: 2, trained_families: ['future-family'] },
    { id: 3, trained_families: [] },
  ]);
  assert.deepEqual(groups.map((g) => g.family), ['zimage', OTHER_FAMILY[0], NOT_TRAINED[0]]);
  assert.deepEqual(groups[1].items.map((d) => d.id), [2]);
  assert.equal(groups[1].label, OTHER_FAMILY[1]);
});

test('full section order: known families, then Other, then Not trained yet (always last)', () => {
  const groups = groupDatasets([
    { id: 1, trained_families: [] },
    { id: 2, trained_families: ['weird'] },
    { id: 3, trained_families: ['flux2klein'] },
    { id: 4, trained_families: ['zimage'] },
  ]);
  assert.deepEqual(groups.map((g) => g.family),
    ['zimage', 'flux2klein', OTHER_FAMILY[0], NOT_TRAINED[0]]);
});

test('primaryFamily: first FAMILY_ORDER match, else Other, else Not trained', () => {
  assert.equal(primaryFamily({ trained_families: ['sdxl'] }), 'sdxl');
  assert.equal(primaryFamily({ trained_families: ['flux', 'zimage'] }), 'zimage');
  assert.equal(primaryFamily({ trained_families: ['unknown'] }), OTHER_FAMILY[0]);
  assert.equal(primaryFamily({ trained_families: [] }), NOT_TRAINED[0]);
  assert.equal(primaryFamily({}), NOT_TRAINED[0]);
  assert.equal(primaryFamily(null), NOT_TRAINED[0]);
});

test('groupDatasets on an empty library returns no sections', () => {
  assert.deepEqual(groupDatasets([]), []);
});

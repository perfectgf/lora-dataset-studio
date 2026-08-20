import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BANK_SORTS, bankSortOptions, bankSortGroups, bankSortStorageKey,
  loadBankSort, saveBankSort, normalizeBankSort,
  DATASET_SORTS, datasetSortOptions, sortDatasetImages, normalizeDatasetSort,
} from './gridSort.js';
import { filterImagesByStatus } from './gridStatusFilter.js';

// id → the dataset row shape the grid actually receives. `framing` is left off
// by default on purpose: that is what a row looks like before the 📐 classify
// pass, and it is the case every sort has to survive.
const img = (id, face_score, status = 'keep', framing = undefined) => ({
  id, face_score, status, filename: `${id}.png`,
  ...(framing === undefined ? {} : { framing }),
});

// Deliberately unordered scores + two rows the face pass never reached.
const IMAGES = [
  img(5, 0.41), img(4, null), img(3, 0.88), img(2, 0.62), img(1, undefined),
];
const order = (list) => list.map((i) => i.id);

// ---- dataset: the central assertion ---------------------------------------

test('face similarity sorts both ways and sinks the unscored to the END', () => {
  // Closest first: 0.88 > 0.62 > 0.41, then the two unscored — never at the top.
  assert.deepEqual(order(sortDatasetImages(IMAGES, 'face_desc')), [3, 2, 5, 4, 1]);
  // Least alike first: the unscored are STILL last, not first — a "worst first"
  // sort that opened on the un-measured pile would be worse than no sort at all.
  assert.deepEqual(order(sortDatasetImages(IMAGES, 'face_asc')), [5, 2, 3, 4, 1]);
  // Ties (both unscored) fall back to the default newest-first order.
  assert.deepEqual(order(sortDatasetImages([img(1, null), img(9, null)], 'face_desc')),
    [9, 1]);
});

test('default (and an unknown id) is a no-op that keeps the same array', () => {
  assert.equal(sortDatasetImages(IMAGES, 'default'), IMAGES);
  assert.equal(sortDatasetImages(IMAGES, 'bogus-from-an-old-localStorage'), IMAGES);
  assert.equal(normalizeDatasetSort('bogus'), 'default');
  assert.equal(normalizeDatasetSort('face_asc'), 'face_asc');
});

test('sorting never mutates the input', () => {
  const before = order(IMAGES);
  sortDatasetImages(IMAGES, 'face_desc');
  assert.deepEqual(order(IMAGES), before);
});

// ---- dataset: composition with the existing filters ------------------------

test('sort composes with the decision filter — no image lost, none invented', () => {
  const mixed = [
    img(5, 0.41, 'keep'), img(4, 0.95, 'reject'), img(3, 0.88, 'keep'),
    img(2, null, 'keep'), img(1, 0.62, 'reject'),
  ];
  const kept = filterImagesByStatus(mixed, 'kept');
  // Filter → sort and sort → filter agree, and both keep exactly the kept set.
  const filterThenSort = sortDatasetImages(kept, 'face_desc');
  const sortThenFilter = filterImagesByStatus(sortDatasetImages(mixed, 'face_desc'), 'kept');
  assert.deepEqual(order(filterThenSort), [3, 5, 2]);
  assert.deepEqual(order(sortThenFilter), order(filterThenSort));
  assert.deepEqual(new Set(order(filterThenSort)), new Set(order(kept)));
  // The rejected rows stay out even though one of them has the best score.
  assert.ok(!order(filterThenSort).includes(4));
});

test('"select all" follows the active sort — it takes the list the grid shows', () => {
  // DatasetGrid derives BOTH its tiles and its select-all from the same `images`
  // prop, so whatever the workspace sorted+filtered IS the selection universe.
  const shown = sortDatasetImages(filterImagesByStatus(IMAGES, 'kept'), 'face_asc');
  const selectAll = shown.filter((i) => i.filename).map((i) => i.id);
  assert.deepEqual(selectAll, order(shown));
  assert.deepEqual(selectAll, [5, 2, 3, 4, 1]);
});

// ---- menus: an option whose data does not exist is disabled WITH the reason -

test('dataset menu greys out an entry until the pass it reads has run', () => {
  // Nothing measured at all: every entry but Newest first is offered and dead,
  // each naming the pass that would bring it to life.
  const byId = (list) => Object.fromEntries(list.map((o) => [o.id, o]));
  const none = byId(datasetSortOptions([img(1, null), img(2, undefined)]));
  assert.equal(none.default.disabled, false);
  for (const id of ['face_desc', 'face_asc']) {
    assert.equal(none[id].disabled, true);
    assert.match(none[id].label, /Analyze faces/, 'the reason must be IN the label');
    assert.match(none[id].title, /No image has a face score yet/);
  }
  for (const id of ['shot', 'shot_face']) {
    assert.equal(none[id].disabled, true);
    assert.match(none[id].label, /Classify framing/, 'the reason must be IN the label');
    assert.match(none[id].title, /No image has a shot type yet/);
  }

  // One scored image is enough for the similarity sorts — and NOT enough for the
  // grouping, which reads a different column entirely.
  const scored = byId(datasetSortOptions(IMAGES));
  assert.equal(scored.face_desc.disabled, false);
  assert.equal(scored.shot.disabled, true);
  assert.match(scored.shot.label, /Classify framing/);

  // The reverse: classified but never scored. The grouping alone works; the one
  // that ALSO reads the face score names the face pass, not the framing pass.
  const framed = byId(datasetSortOptions([img(1, null, 'keep', 'face'), img(2, null, 'keep', 'body')]));
  assert.equal(framed.shot.disabled, false);
  assert.equal(framed.shot_face.disabled, true);
  assert.match(framed.shot_face.label, /Analyze faces/,
    'an entry needing two passes names the first one still missing');

  const all = byId(datasetSortOptions([img(1, 0.5, 'keep', 'face')]));
  assert.deepEqual(Object.values(all).map((o) => o.disabled), [false, false, false, false, false]);
  assert.deepEqual(Object.keys(all), DATASET_SORTS.map((s) => s.id));
});

// ---- shot type: a CATEGORICAL order, asked for by .samexit (Discord) ---------

// A set that mixes the four kinds, out of order, plus one image the classify
// pass never reached and one it could not call.
const MIXED = [
  img(1, 0.30, 'keep', 'back'), img(2, 0.90, 'keep', 'face'), img(3, 0.10, 'keep', 'unknown'),
  img(4, 0.50, 'keep', 'body'), img(5, 0.70, 'keep', 'face'), img(6, null, 'keep', 'bust'),
  img(7, 0.20, 'keep'),
];

test('shot type puts every kind in one run, in the order the app speaks them', () => {
  // face, face, bust, body, back — then the two with no shot type, at the END.
  assert.deepEqual(order(sortDatasetImages(MIXED, 'shot')), [5, 2, 6, 4, 1, 7, 3]);
  // Inside a kind, the fallback is the default newest-first (5 before 2), NOT
  // the face score — 2 scores higher and still comes second.
  assert.equal(sortDatasetImages(MIXED, 'shot')[0].id, 5);
});

test('shot type then similarity ranks inside each kind, unscored last in its own run', () => {
  // face: 2 (0.90) then 5 (0.70) — the reverse of the newest-first order above.
  // bust: 6 alone, unscored. body: 4. back: 1. Then the unclassified pair.
  assert.deepEqual(order(sortDatasetImages(MIXED, 'shot_face')), [2, 5, 6, 4, 1, 7, 3]);
  const twoFaces = [img(8, null, 'keep', 'face'), img(9, 0.4, 'keep', 'face')];
  // An unscored row sinks inside ITS OWN kind, never out of the group.
  assert.deepEqual(order(sortDatasetImages(twoFaces, 'shot_face')), [9, 8]);
});

test('grouping never adds, drops or duplicates an image', () => {
  for (const id of ['shot', 'shot_face']) {
    const out = sortDatasetImages(MIXED, id);
    assert.equal(out.length, MIXED.length);
    assert.deepEqual([...order(out)].sort((a, b) => a - b), [1, 2, 3, 4, 5, 6, 7]);
    assert.notEqual(out, MIXED, 'a real order returns a new array, never mutates');
    assert.deepEqual(order(MIXED), [1, 2, 3, 4, 5, 6, 7], 'the input is untouched');
  }
});

test('bank menu greys out what the passes have not produced yet', () => {
  const fresh = bankSortOptions({ scanned: 0, scored: 0 });
  const byId = Object.fromEntries(fresh.map((o) => [o.id, o]));
  assert.equal(byId.default.disabled, false);
  for (const id of ['res_desc', 'res_asc', 'sharp_desc', 'sharp_asc']) {
    assert.equal(byId[id].disabled, true, `${id} should be disabled`);
    assert.match(byId[id].label, /Scan quality/);
  }
  for (const id of ['aesthetic_desc', 'aesthetic_asc']) {
    assert.equal(byId[id].disabled, true, `${id} should be disabled`);
    assert.match(byId[id].label, /Score/);
  }
  // Scanned but not scored: everything the scan measures is usable, everything
  // the ✨ Score pass measures is not — one pass, one whole section of the menu.
  const scanned = Object.fromEntries(
    bankSortOptions({ scanned: 120, scored: 0, faces: 0 }).map((o) => [o.id, o.disabled]));
  for (const id of ['res_desc', 'size_asc', 'sharp_desc', 'noise_asc', 'flat_desc',
    'detail_asc', 'bars_desc', 'jpeg_asc']) {
    assert.equal(scanned[id], false, `${id} should be usable after a scan`);
  }
  for (const id of ['aesthetic_desc', 'nsfw_asc', 'face_desc']) {
    assert.equal(scanned[id], true, `${id} needs its own pass`);
  }
  // Everything measured: nothing greyed out.
  assert.ok(bankSortOptions({ scanned: 120, scored: 120, faces: 120 })
    .every((o) => !o.disabled));
  // Payload not loaded yet — "I don't know" is never rendered as "you can't".
  assert.ok(bankSortOptions(null).every((o) => !o.disabled));
  assert.ok(bankSortOptions(undefined).every((o) => !o.disabled));
});

test('bank sort ids are the ones the server accepts (never rename: they are stored)', () => {
  // Mirrors image_bank_service._SORT_KEYS × (desc, asc), in menu order. A rename
  // here silently breaks a remembered order in someone's localStorage AND stops
  // matching the server, which would drop back to the default without a word.
  assert.deepEqual(BANK_SORTS.map((s) => s.id), [
    'default',
    'res_desc', 'res_asc', 'size_desc', 'size_asc',
    'aesthetic_desc', 'aesthetic_asc', 'nsfw_desc', 'nsfw_asc',
    'sharp_desc', 'sharp_asc', 'noise_desc', 'noise_asc',
    'flat_desc', 'flat_asc', 'detail_desc', 'detail_asc',
    'bars_desc', 'bars_asc', 'jpeg_desc', 'jpeg_asc',
    'face_desc', 'face_asc',
    'yaw_desc', 'yaw_asc',
    'medium_conf_desc', 'medium_conf_asc',
  ]);
  // Every measure is offered BOTH ways — the point of the widening. A one-way
  // sort is the gap the chips already had.
  const ids = new Set(BANK_SORTS.map((s) => s.id));
  for (const id of ids) {
    if (id === 'default') continue;
    const [, key, dir] = id.match(/^(.+)_(desc|asc)$/);
    assert.ok(ids.has(`${key}_${dir === 'asc' ? 'desc' : 'asc'}`), `${id} has no opposite`);
  }
  // Every entry says WHY you would click it, and no title is a restatement of
  // the arrow — the menu is now long enough that labels alone stop helping.
  for (const s of BANK_SORTS) assert.ok(s.title && s.title.length > 12, s.id);
});

// ---- the menu is grouped by the pass that measures it ----------------------

test('bank menu groups by pass, keeps every option, and heads with Default', () => {
  const groups = bankSortGroups({ scanned: 1, scored: 1, faces: 1 });
  assert.deepEqual(groups[0], {
    group: '',
    options: [{ id: 'default', group: '', label: 'Default', disabled: false,
      title: groups[0].options[0].title }],
  });
  assert.deepEqual(groups.map((g) => g.group),
    ['', '📁 File', '✨ Score', '🔎 Scan quality', '🎭 Faces', '🎨 Medium']);
  // Grouping is a RE-ARRANGEMENT: no option invented, none lost, none duplicated.
  const flat = groups.flatMap((g) => g.options.map((o) => o.id));
  assert.deepEqual(flat.slice().sort(), BANK_SORTS.map((s) => s.id).slice().sort());
  assert.equal(new Set(flat).size, flat.length);
});

// ---- the chosen order is remembered PER BANK -------------------------------

/** The two localStorage methods the helpers touch, over a plain Map. */
const fakeStore = (init = {}) => {
  const map = new Map(Object.entries(init));
  return {
    map,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => map.set(k, v),
    removeItem: (k) => map.delete(k),
  };
};

test('a bank remembers its own order, and one bank never speaks for another', () => {
  const store = fakeStore();
  assert.equal(loadBankSort(7, store), 'default');
  saveBankSort(7, 'sharp_asc', store);
  assert.equal(loadBankSort(7, store), 'sharp_asc');
  // Bank 8 is a different bank: the order that suits a 9 000-image dump is not
  // the one that suits a hand-picked set.
  assert.equal(loadBankSort(8, store), 'default');
  assert.equal(bankSortStorageKey(7), 'lds.bank.7.sort');
});

test('going back to Default FORGETS the preference instead of storing it', () => {
  const store = fakeStore();
  saveBankSort(3, 'nsfw_desc', store);
  saveBankSort(3, 'default', store);
  assert.equal(store.map.size, 0, 'no key left behind for the default order');
  assert.equal(loadBankSort(3, store), 'default');
});

test('a stored order from another build degrades to Default, never to a 500', () => {
  // The value outlives the build that wrote it: a sort this version dropped, a
  // hand-edited key, a corrupted profile. All of them read as "no preference".
  for (const bogus of ['face_score_desc', 'sharp', '', 'null', '{"a":1}']) {
    assert.equal(loadBankSort(1, fakeStore({ 'lds.bank.1.sort': bogus })), 'default');
  }
  assert.equal(normalizeBankSort('sharp_asc'), 'sharp_asc');
  assert.equal(normalizeBankSort(undefined), 'default');
  // A storage that throws on every access (private mode) is not a crash.
  const hostile = { getItem() { throw new Error('denied') },
    setItem() { throw new Error('denied') }, removeItem() { throw new Error('denied') } };
  assert.equal(loadBankSort(1, hostile), 'default');
  assert.equal(saveBankSort(1, 'res_asc', hostile), 'res_asc');
});

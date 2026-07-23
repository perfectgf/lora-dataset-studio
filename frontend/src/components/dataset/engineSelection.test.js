import test from 'node:test';
import assert from 'node:assert/strict';
import {
  canonicalEngines, readEngines, writeEngines, readMode, writeMode, primaryEngine,
  distributeVariations, engineBatches, kleinQueuesBehindApi, totalImages, estimateCost,
  billingEngines, generateBlockedReason, STORAGE_ENGINES, STORAGE_PRIMARY, STORAGE_MODE,
} from './engineSelection.js';

/** Minimal localStorage stand-in. `boom` makes every access throw, the private-
 *  browsing case the real code must survive. */
function fakeStorage(seed = {}, { boom = false } = {}) {
  const data = { ...seed };
  return {
    data,
    getItem(k) { if (boom) throw new Error('denied'); return k in data ? data[k] : null; },
    setItem(k, v) { if (boom) throw new Error('denied'); data[k] = String(v); },
  };
}

const shots = (n) => Array.from({ length: n }, (_, i) => ({ label: `s${i}`, prompt: `p${i}` }));

test('canonicalEngines keeps known ids, de-duplicates and re-orders', () => {
  assert.deepEqual(canonicalEngines(['chatgpt', 'klein']), ['klein', 'chatgpt']);
  assert.deepEqual(canonicalEngines(['chatgpt', 'chatgpt']), ['chatgpt']);
  assert.deepEqual(canonicalEngines(['midjourney', null, 7, 'klein']), ['klein']);
  assert.deepEqual(canonicalEngines([]), []);
  assert.deepEqual(canonicalEngines('klein'), []);      // not a list
  assert.deepEqual(canonicalEngines(undefined), []);
});

test('readEngines: a legacy single-string profile reads as a one-engine selection', () => {
  // THE compatibility case: an existing user upgrades and must keep generating
  // exactly as before, with no migration step.
  assert.deepEqual(readEngines(fakeStorage({ [STORAGE_PRIMARY]: 'chatgpt' })), ['chatgpt']);
  assert.deepEqual(readEngines(fakeStorage({ [STORAGE_PRIMARY]: 'klein' })), ['klein']);
});

test('readEngines: falls back to the historic default, never crashes', () => {
  assert.deepEqual(readEngines(fakeStorage()), ['nanobanana']);
  assert.deepEqual(readEngines(fakeStorage({ [STORAGE_PRIMARY]: 'gone-engine' })), ['nanobanana']);
  assert.deepEqual(readEngines(fakeStorage({ [STORAGE_ENGINES]: '{oops' })), ['nanobanana']);
  assert.deepEqual(readEngines(fakeStorage({}, { boom: true })), ['nanobanana']);
  assert.deepEqual(readEngines(null), ['nanobanana']);
});

test('readEngines: the list key wins and an empty list is a real state', () => {
  const s = fakeStorage({ [STORAGE_ENGINES]: '["chatgpt","klein"]', [STORAGE_PRIMARY]: 'nanobanana' });
  assert.deepEqual(readEngines(s), ['klein', 'chatgpt']);
  // Everything unchecked must NOT silently resurrect the legacy engine.
  assert.deepEqual(readEngines(fakeStorage({ [STORAGE_ENGINES]: '[]', [STORAGE_PRIMARY]: 'klein' })), []);
});

test('writeEngines mirrors the primary onto the legacy key', () => {
  const s = fakeStorage();
  writeEngines(s, ['chatgpt', 'klein']);
  assert.equal(s.data[STORAGE_ENGINES], '["klein","chatgpt"]');
  assert.equal(s.data[STORAGE_PRIMARY], 'klein');   // regenerate + ✎ modal read this
});

test('writeEngines keeps the last known engine when the selection is emptied', () => {
  const s = fakeStorage({ [STORAGE_PRIMARY]: 'chatgpt' });
  writeEngines(s, []);
  assert.equal(s.data[STORAGE_ENGINES], '[]');
  assert.equal(s.data[STORAGE_PRIMARY], 'chatgpt');  // regenerate still has an engine
  assert.doesNotThrow(() => writeEngines(fakeStorage({}, { boom: true }), ['klein']));
});

test('mode: persisted, defaults to split, rejects junk', () => {
  assert.equal(readMode(fakeStorage()), 'split');
  assert.equal(readMode(fakeStorage({ [STORAGE_MODE]: 'all' })), 'all');
  assert.equal(readMode(fakeStorage({ [STORAGE_MODE]: 'wat' })), 'split');
  assert.equal(readMode(fakeStorage({}, { boom: true })), 'split');
  const s = fakeStorage();
  assert.equal(writeMode(s, 'all'), 'all');
  assert.equal(s.data[STORAGE_MODE], 'all');
  assert.equal(writeMode(s, 'nope'), 'split');
});

test('primaryEngine follows the canonical order', () => {
  assert.equal(primaryEngine(['chatgpt', 'klein']), 'klein');
  assert.equal(primaryEngine(['chatgpt', 'nanobanana']), 'nanobanana');
  assert.equal(primaryEngine([]), null);
});

test('split mode: every shot goes to exactly one engine, 25/3 → 9+8+8', () => {
  const batches = distributeVariations(shots(25), ['klein', 'nanobanana', 'chatgpt'], 'split');
  assert.deepEqual(batches.map((b) => b.variations.length), [9, 8, 8]);
  const seen = batches.flatMap((b) => b.variations.map((v) => v.label));
  assert.equal(seen.length, 25);                       // nothing duplicated
  assert.equal(new Set(seen).size, 25);                // nothing lost
});

test('all mode: every engine renders every shot', () => {
  const batches = distributeVariations(shots(25), ['klein', 'nanobanana', 'chatgpt'], 'all');
  assert.deepEqual(batches.map((b) => b.variations.length), [25, 25, 25]);
});

test('one engine is byte-for-byte the old behaviour, whatever the mode', () => {
  const s = shots(4);
  for (const mode of ['split', 'all']) {
    const batches = distributeVariations(s, ['chatgpt'], mode);
    assert.equal(batches.length, 1);
    assert.equal(batches[0].generator, 'chatgpt');
    assert.deepEqual(batches[0].variations, s);
  }
});

test('degenerate inputs produce no batch at all', () => {
  assert.deepEqual(distributeVariations(shots(3), [], 'split'), []);
  assert.deepEqual(distributeVariations([], ['klein'], 'all'), []);
  // More engines than shots: no empty batch is ever sent to the server.
  const batches = distributeVariations(shots(2), ['klein', 'nanobanana', 'chatgpt'], 'split');
  assert.deepEqual(batches.map((b) => b.generator), ['klein', 'nanobanana']);
});

test('engineBatches dispatches the API engines before the local GPU one', () => {
  const order = engineBatches(shots(6), ['klein', 'nanobanana', 'chatgpt'], 'all')
    .map((b) => b.generator);
  assert.deepEqual(order, ['nanobanana', 'chatgpt', 'klein']);
  // Klein alone still goes out (nothing to wait behind).
  assert.deepEqual(engineBatches(shots(2), ['klein'], 'split').map((b) => b.generator), ['klein']);
});

test('kleinQueuesBehindApi only when the GPU shares the run with an API', () => {
  assert.equal(kleinQueuesBehindApi(['klein', 'chatgpt']), true);
  assert.equal(kleinQueuesBehindApi(['klein']), false);
  assert.equal(kleinQueuesBehindApi(['nanobanana', 'chatgpt']), false);
});

test('totalImages: split keeps the count, all multiplies it', () => {
  assert.equal(totalImages(25, ['klein', 'nanobanana', 'chatgpt'], 'split'), 25);
  assert.equal(totalImages(25, ['klein', 'nanobanana', 'chatgpt'], 'all'), 75);
  assert.equal(totalImages(25, ['chatgpt'], 'all'), 25);
  assert.equal(totalImages(10, ['nanobanana', 'chatgpt'], 'all', 2), 40);
  assert.equal(totalImages(10, [], 'all'), 0);
  assert.equal(totalImages(0, ['chatgpt'], 'all'), 0);
});

test('estimateCost: Klein is free and only pays for its own share in split', () => {
  // 24 shots over klein + chatgpt: 12 each, only the 12 ChatGPT ones bill.
  assert.equal(estimateCost(24, ['klein', 'chatgpt'], 'split').toFixed(2), (12 * 0.17).toFixed(2));
  // Same engines in all mode: ChatGPT renders all 24.
  assert.equal(estimateCost(24, ['klein', 'chatgpt'], 'all').toFixed(2), (24 * 0.17).toFixed(2));
  // Klein only → free whatever the mode.
  assert.equal(estimateCost(30, ['klein'], 'all'), 0);
  // The ChatGPT subscription lane spends quota, not dollars.
  assert.equal(estimateCost(10, ['chatgpt'], 'all', { gptViaSub: true }), 0);
  assert.equal(estimateCost(10, ['nanobanana', 'chatgpt'], 'all', { gptViaSub: true }).toFixed(2),
    (10 * 0.15).toFixed(2));
  // The multiplier bills too.
  assert.equal(estimateCost(4, ['nanobanana'], 'all', { multiplier: 3 }).toFixed(2),
    (12 * 0.15).toFixed(2));
  // Uneven split: 25 shots / 3 engines → nanobanana 8, chatgpt 8, klein 9 (free).
  assert.equal(estimateCost(25, ['klein', 'nanobanana', 'chatgpt'], 'split').toFixed(2),
    (8 * 0.15 + 8 * 0.17).toFixed(2));
  assert.equal(estimateCost(10, [], 'all'), 0);
});

test('billingEngines names only the lanes that really charge', () => {
  assert.deepEqual(billingEngines(['klein', 'nanobanana', 'chatgpt']), ['nanobanana', 'chatgpt']);
  assert.deepEqual(billingEngines(['klein', 'chatgpt'], { gptViaSub: true }), []);
  assert.deepEqual(billingEngines(['klein']), []);
});

test('generateBlockedReason: no silent empty batch, and the server cap is explained', () => {
  assert.match(generateBlockedReason({ engines: [], shotCount: 5, mode: 'split' }), /at least one engine/);
  assert.match(generateBlockedReason({ engines: ['klein'], shotCount: 0, mode: 'split' }), /at least one shot/);
  assert.equal(generateBlockedReason({ engines: ['klein'], shotCount: 5, mode: 'split' }), null);
  // 25 shots × 3 engines = 75 > 60: refused HERE, before the click, not by a
  // half-dispatched batch on the server.
  const over = generateBlockedReason({
    engines: ['klein', 'nanobanana', 'chatgpt'], shotCount: 25, mode: 'all', maxFanout: 60 });
  assert.match(over, /75 images/);
  assert.match(over, /switch to Split/);
  assert.equal(generateBlockedReason({
    engines: ['klein', 'nanobanana', 'chatgpt'], shotCount: 25, mode: 'split', maxFanout: 60 }), null);
  // Unknown cap (older server / probe failed) → never blocks the user.
  assert.equal(generateBlockedReason({
    engines: ['klein', 'nanobanana', 'chatgpt'], shotCount: 25, mode: 'all' }), null);
});

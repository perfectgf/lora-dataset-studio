/* The ⚖ compare dialog's pure rules, executed — plus the parity contract:
 * one component, BOTH surfaces, or the dialog quietly becomes dataset-only the
 * way features on this repo have before. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { DEFAULT_COMPARE_CAP, compareSeed, defaultTicked, toggleTicked } from './kleinCompare.js';

test('the default ticks start from the current pick and stay a sitting-through count', () => {
  const choices = ['a', 'b', 'c', 'd', 'e'];
  // No pick: the first CAP models.
  assert.deepEqual(defaultTicked(choices, null), ['a', 'b', 'c']);
  // A pick further down the list is IN, first — comparing without the current
  // model answers a question nobody asked.
  assert.deepEqual(defaultTicked(choices, 'd'), ['d', 'a', 'b']);
  assert.equal(defaultTicked(choices, 'd').length, DEFAULT_COMPARE_CAP);
  // A stored pick that left the disk is not resurrected.
  assert.deepEqual(defaultTicked(['a', 'b'], 'gone'), ['a', 'b']);
  assert.deepEqual(defaultTicked([], null), []);
});

test('toggling is symmetric and never duplicates', () => {
  assert.deepEqual(toggleTicked(['a'], 'b'), ['a', 'b']);
  assert.deepEqual(toggleTicked(['a', 'b'], 'b'), ['a']);
  assert.deepEqual(toggleTicked(toggleTicked(['a'], 'b'), 'b'), ['a']);
});

test('the seed is a JSON-safe 32-bit integer', () => {
  for (let i = 0; i < 64; i += 1) {
    const s = compareSeed();
    assert.ok(Number.isInteger(s) && s >= 0 && s <= 0xffffffff);
  }
});

/* --- parity: one dialog, two mounts, each arming its own authority --------- */
test('both surfaces mount the compare dialog against their own route', () => {
  const read = (rel) => readFileSync(new URL(`../../${rel}`, import.meta.url), 'utf8');
  const ds = read('components/dataset/DatasetWorkspace.jsx');
  const bank = read('components/bank/BankWatermarkPanel.jsx');
  for (const [name, src] of [['dataset', ds], ['bank', bank]]) {
    assert.match(src, /<KleinCompareDialog/, `the ${name} surface lost the ⚖ dialog`);
    assert.match(src, /Compare models…/, `the ${name} surface lost the ⚖ button`);
  }
  assert.match(ds, /watermarks\/klein-compare/);
  assert.match(bank, /watermark\/klein-compare/);
  // Two DIFFERENT adopt semantics, both deliberate: the dataset saves its one
  // stored authority; the bank arms a per-run override and stores nothing.
  assert.match(ds, /\/klein-model`, \{ klein_model: model \}/,
    'adopting on the dataset no longer saves the dataset pick');
  assert.match(bank, /setKleinRunModel\(model\)/,
    'adopting on the bank no longer arms the per-run override');
  assert.match(bank, /klein_model: kleinRunModel/,
    'the bank launch no longer carries the armed override');
});

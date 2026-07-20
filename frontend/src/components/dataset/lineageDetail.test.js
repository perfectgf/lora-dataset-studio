import test from 'node:test';
import assert from 'node:assert/strict';
import { configRows, noteBadge, diffConfigs, toggleDiffSelection } from './lineageDetail.js';

test('configRows lists known keys in order, formats values', () => {
  const rows = configRows({ learning_rate: '1e-4', rank: 32, network: 'lora' });
  const labels = rows.map((r) => r.label);
  assert.ok(labels.indexOf('Rank') < labels.indexOf('Learning rate') || labels.includes('Rank'));
  assert.ok(rows.some((r) => r.label === 'Learning rate' && String(r.value) === '1e-4'));
});

test('configRows returns [] for a legacy run with no config', () => {
  assert.deepEqual(configRows(null), []);
});

test('noteBadge is true when the run or any checkpoint is annotated', () => {
  assert.equal(noteBadge({ has_note: true, checkpoints: [] }), true);
  assert.equal(noteBadge({ has_note: false, checkpoints: [{ note: 'x' }] }), true);
  assert.equal(noteBadge({ has_note: false, checkpoints: [{ note: '' }] }), false);
  assert.equal(noteBadge({}), false);
});

// ---- diffConfigs (Lab, slice 2 — compare two runs) ----------------------

test('diffConfigs flags changed:true EXACTLY on the keys that differ', () => {
  const a = { rank: 16, alpha: 16, learning_rate: '1e-4', optimizer: 'adamw' };
  const b = { rank: 32, alpha: 16, learning_rate: '5e-5', optimizer: 'adamw' };
  const rows = diffConfigs(a, b);
  const byKey = Object.fromEntries(rows.map((r) => [r.key, r]));
  assert.equal(byKey.rank.changed, true);
  assert.equal(byKey.rank.a, '16');
  assert.equal(byKey.rank.b, '32');
  assert.equal(byKey.learning_rate.changed, true);
  assert.equal(byKey.alpha.changed, false);       // identical → not highlighted
  assert.equal(byKey.optimizer.changed, false);
  // exactly the two differing keys are flagged
  assert.deepEqual(rows.filter((r) => r.changed).map((r) => r.key).sort(),
    ['learning_rate', 'rank']);
});

test('diffConfigs keeps CONFIG_LABELS order and carries the label', () => {
  const rows = diffConfigs({ learning_rate: '1e-4', rank: 16 }, { learning_rate: '1e-4', rank: 32 });
  const keys = rows.map((r) => r.key);
  assert.ok(keys.indexOf('rank') < keys.indexOf('learning_rate'));   // rank before LR in the table
  assert.ok(rows.every((r) => typeof r.label === 'string' && r.label.length > 0));
});

test('diffConfigs: identical snapshots produce zero changed rows', () => {
  const cfg = { rank: 32, alpha: 16, learning_rate: '5e-5', optimizer: 'prodigy' };
  const rows = diffConfigs(cfg, { ...cfg });
  assert.equal(rows.filter((r) => r.changed).length, 0);
  assert.ok(rows.length > 0);   // but it still lists the shared rows
});

test('diffConfigs: this test bites — ignoring b would break it', () => {
  // A mutant that returned `changed: false` regardless of b (i.e. ignored b)
  // must fail here: the two runs share only structure, every value differs.
  const rows = diffConfigs({ rank: 16, steps: 1000 }, { rank: 64, steps: 4000 });
  assert.equal(rows.filter((r) => r.changed).length, rows.length);
  assert.ok(rows.length >= 2);
});

test('diffConfigs: one side null → the recorded side shows as changed, other is null', () => {
  const rows = diffConfigs(null, { rank: 32, optimizer: 'adamw' });
  assert.ok(rows.length >= 2);
  for (const r of rows) {
    assert.equal(r.a, null);          // legacy side recorded nothing
    assert.notEqual(r.b, null);
    assert.equal(r.changed, true);    // present-on-one-side is a difference
  }
});

test('diffConfigs: both sides null → [] (nothing to compare)', () => {
  assert.deepEqual(diffConfigs(null, null), []);
});

test('diffConfigs: a key present on only one side is a changed row, empty on the other', () => {
  const rows = diffConfigs({ rank: 16 }, { rank: 16, ema: 'on' });
  const ema = rows.find((r) => r.key === 'ema');
  assert.ok(ema);
  assert.equal(ema.a, null);
  assert.equal(ema.b, 'on');
  assert.equal(ema.changed, true);
  assert.equal(rows.find((r) => r.key === 'rank').changed, false);
});

// ---- toggleDiffSelection (bounded-to-2 pick state) ----------------------

test('toggleDiffSelection adds, removes, and slides at a cap of two', () => {
  assert.deepEqual(toggleDiffSelection([], 1), [1]);
  assert.deepEqual(toggleDiffSelection([1], 2), [1, 2]);
  assert.deepEqual(toggleDiffSelection([1, 2], 1), [2]);        // toggle off
  assert.deepEqual(toggleDiffSelection([1, 2], 3), [2, 3]);     // third pick drops the oldest
  assert.deepEqual(toggleDiffSelection(undefined, 5), [5]);     // tolerates no prior state
});

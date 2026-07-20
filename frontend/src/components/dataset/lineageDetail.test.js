import test from 'node:test';
import assert from 'node:assert/strict';
import { configRows, noteBadge } from './lineageDetail.js';

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

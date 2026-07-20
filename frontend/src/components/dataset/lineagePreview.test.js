import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  checkpointKey, toggleCheckpointSelection, selectedCheckpointRefs,
  describePreviewSelection, parseSeedInput,
  checkpointDeployed, lineageImportPayload,
} from './lineagePreview.js';

const pills = new Map([
  ['7:500', { record_id: 7, step: 500, testable: true }],
  ['7:1000', { record_id: 7, step: 1000, testable: true }],
  ['9:1500', { record_id: 9, step: 1500, testable: false }],   // not deployed
]);

test('checkpointKey joins record and step', () => {
  assert.equal(checkpointKey(7, 500), '7:500');
});

test('toggleCheckpointSelection adds then removes without mutating', () => {
  const a = new Set();
  const b = toggleCheckpointSelection(a, '7:500');
  assert.deepEqual([...b], ['7:500']);
  assert.equal(a.size, 0);                       // original untouched
  const c = toggleCheckpointSelection(b, '7:500');
  assert.equal(c.size, 0);
});

test('selectedCheckpointRefs keeps only testable picks, as {record_id, step}', () => {
  const sel = new Set(['7:500', '9:1500', '7:1000']);
  const refs = selectedCheckpointRefs(sel, pills);
  assert.deepEqual(refs, [{ record_id: 7, step: 500 }, { record_id: 7, step: 1000 }]);
});

test('describePreviewSelection: nothing selected → disabled with hint', () => {
  const d = describePreviewSelection(new Set(), pills);
  assert.equal(d.enabled, false);
  assert.match(d.hint, /Check one or more/);
});

test('describePreviewSelection: only undeployed → disabled, deploy hint', () => {
  const d = describePreviewSelection(new Set(['9:1500']), pills);
  assert.equal(d.enabled, false);
  assert.equal(d.testableCount, 0);
  assert.match(d.hint, /deployed/);
});

test('describePreviewSelection: mixed → enabled, skip hint', () => {
  const d = describePreviewSelection(new Set(['7:500', '9:1500']), pills);
  assert.equal(d.enabled, true);
  assert.equal(d.testableCount, 1);
  assert.equal(d.undeployedCount, 1);
  assert.match(d.hint, /1 not-deployed checkpoint will be skipped/);
});

test('describePreviewSelection: all testable → enabled, no hint', () => {
  const d = describePreviewSelection(new Set(['7:500', '7:1000']), pills);
  assert.equal(d.enabled, true);
  assert.equal(d.hint, null);
});

test('checkpointDeployed: true only when the pill is testable', () => {
  assert.equal(checkpointDeployed({ testable: true }), true);
  assert.equal(checkpointDeployed({ testable: false }), false);
  assert.equal(checkpointDeployed({}), false);
  assert.equal(checkpointDeployed(null), false);
});

test('lineageImportPayload: cloud node carries cloud_run_id + run family/variant/base', () => {
  const node = { source: 'cloud', run_id: 42, train_type: 'flux', variant: 'turbo', base_model: '' };
  const pill = { step: 500, filename: 'lora_000500.safetensors' };
  assert.deepEqual(lineageImportPayload(node, pill), {
    filename: 'lora_000500.safetensors',
    base_model: '', train_type: 'flux', variant: 'turbo', cloud_run_id: 42,
  });
});

test('lineageImportPayload: local node has no cloud_run_id', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'sd_xl_base.safetensors' };
  const pill = { step: 1000, filename: 'lora_001000.safetensors' };
  const body = lineageImportPayload(node, pill);
  assert.equal('cloud_run_id' in body, false);
  assert.deepEqual(body, {
    filename: 'lora_001000.safetensors',
    base_model: 'sd_xl_base.safetensors', train_type: 'sdxl', variant: 'base',
  });
});

test('lineageImportPayload: null when nothing to deploy', () => {
  assert.equal(lineageImportPayload(null, { filename: 'x' }), null);
  assert.equal(lineageImportPayload({ source: 'local' }, null), null);
  assert.equal(lineageImportPayload({ source: 'local' }, { step: 1 }), null);   // no filename
  // a cloud node with no resolved run has nothing importable
  assert.equal(lineageImportPayload({ source: 'cloud', run_id: null }, { filename: 'x' }), null);
});

test('parseSeedInput: blank → null (engine picks), int → number, junk → error', () => {
  assert.deepEqual(parseSeedInput(''), { seed: null });
  assert.deepEqual(parseSeedInput('  '), { seed: null });
  assert.deepEqual(parseSeedInput('42'), { seed: 42 });
  assert.ok(parseSeedInput('-1').error);
  assert.ok(parseSeedInput('1.5').error);
  assert.ok(parseSeedInput('abc').error);
});

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  checkpointKey, toggleCheckpointSelection, selectedCheckpointRefs,
  describePreviewSelection, parseSeedInput,
  checkpointDeployed, lineageImportPayload,
  lineageDeletePayload, checkpointDeletable, checkpointIsBestSettings,
  describeCheckpointDelete,
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

// --- 🗑 Delete this save (graph popover) -----------------------------------

test('lineageDeletePayload: a cloud pill carries its cloud_run_id', () => {
  const node = { source: 'cloud', run_id: 42, train_type: 'zimage', variant: 'base', base_model: '' };
  const body = lineageDeletePayload(node, { step: 2000, filename: 'e2000.safetensors' });
  assert.equal(body.cloud_run_id, 42);
  assert.equal(body.filename, 'e2000.safetensors');
});

test('lineageDeletePayload: a local pill carries base/family/variant, no run id', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'sd_xl_base.safetensors' };
  const body = lineageDeletePayload(node, { step: 1000, filename: 'lora_001000.safetensors' });
  assert.equal('cloud_run_id' in body, false);
  assert.deepEqual(body, {
    filename: 'lora_001000.safetensors',
    base_model: 'sd_xl_base.safetensors', train_type: 'sdxl', variant: 'base',
  });
});

test('lineageDeletePayload: null rather than a body that would hit the wrong file', () => {
  assert.equal(lineageDeletePayload(null, { filename: 'x' }), null);
  assert.equal(lineageDeletePayload({ source: 'local' }, { step: 1 }), null);   // no filename
  // a cloud node with no resolved run would be deleted as a LOCAL file — refuse
  assert.equal(lineageDeletePayload({ source: 'cloud', run_id: null }, { filename: 'x' }), null);
});

test('checkpointDeletable: gone pills and in-flight cloud runs offer nothing', () => {
  const local = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'b' };
  assert.equal(checkpointDeletable(local, { step: 1, filename: 'a.safetensors' }), true);
  assert.equal(checkpointDeletable(local, { step: 1, filename: 'a.safetensors', present: false }), false);
  const training = { source: 'cloud', run_id: 7, status: 'training', train_type: 'zimage' };
  assert.equal(checkpointDeletable(training, { step: 1, filename: 'a.safetensors' }), false);
  const done = { source: 'cloud', run_id: 7, status: 'done', train_type: 'zimage' };
  assert.equal(checkpointDeletable(done, { step: 1, filename: 'a.safetensors' }), true);
});

test('checkpointIsBestSettings matches on the basename, both sides', () => {
  const pill = { filename: 'lora_001000.safetensors' };
  assert.equal(checkpointIsBestSettings(pill, 'loras/zimage/lora_001000.safetensors'), true);
  const winPath = ['C:', 'loras', 'lora_001000.safetensors'].join(String.fromCharCode(92));
  assert.equal(checkpointIsBestSettings(pill, winPath), true);
  assert.equal(checkpointIsBestSettings(pill, 'lora_002000.safetensors'), false);
  assert.equal(checkpointIsBestSettings(pill, null), false);   // pin unknown → no false alarm
});

test('describeCheckpointDelete says trash + ComfyUI copy kept, and warns on ★ best settings', () => {
  const node = { source: 'local', train_type: 'sdxl' };
  const pill = { step: 1000, filename: 'lora_001000.safetensors' };
  const plain = describeCheckpointDelete(node, pill);
  assert.equal(plain.isBest, false);
  assert.match(plain.message, /trash/i);
  assert.match(plain.message, /Settings/);
  assert.match(plain.message, /imported into ComfyUI stays/);
  assert.doesNotMatch(plain.message, /BEST SETTINGS/);

  const best = describeCheckpointDelete(node, pill, { bestSettingsLora: 'loras/lora_001000.safetensors' });
  assert.equal(best.isBest, true);
  assert.match(best.message, /★ BEST SETTINGS/);
  assert.match(best.message, /trash/i);   // the reassurance survives the warning
});

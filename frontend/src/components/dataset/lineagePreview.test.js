import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  checkpointKey, toggleCheckpointSelection, selectedCheckpointRefs,
  describePreviewSelection, parseSeedInput,
  checkpointDeployed, lineageImportPayload,
  lineageDeletePayload, checkpointDeleteTarget, checkpointIsBestSettings,
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

test('checkpointIsBestSettings matches on the basename, both sides', () => {
  const pill = { filename: 'lora_001000.safetensors' };
  assert.equal(checkpointIsBestSettings(pill, 'loras/zimage/lora_001000.safetensors'), true);
  const winPath = ['C:', 'loras', 'lora_001000.safetensors'].join(String.fromCharCode(92));
  assert.equal(checkpointIsBestSettings(pill, winPath), true);
  assert.equal(checkpointIsBestSettings(pill, 'lora_002000.safetensors'), false);
  assert.equal(checkpointIsBestSettings(pill, null), false);   // pin unknown → no false alarm
});


test('checkpointIsBestSettings also matches the DEPLOYED name of a deployed pill', () => {
  const pill = { filename: 'lora_001000.safetensors', testable: true,
    deployed_filename: 'z image/lora_nova_000001000_rc90_v2.safetensors' };
  assert.equal(checkpointIsBestSettings(pill, 'lora_nova_000001000_rc90_v2.safetensors'), true);
});

test('checkpointDeleteTarget: a DEPLOYED pill aims at the ComfyUI copy', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'b' };
  const pill = { step: 1000, filename: 'lora_001000.safetensors', testable: true,
    deployed_filename: 'sdxl/lora_nova_000001000_rl7.safetensors' };
  const t = checkpointDeleteTarget(node, pill);
  assert.equal(t.kind, 'deployed');
  assert.equal(t.path, 'train/checkpoint/delete');          // the imported-LoRA route
  assert.deepEqual(t.body, { filename: 'sdxl/lora_nova_000001000_rl7.safetensors', train_type: 'sdxl' });
  assert.match(t.label, /ComfyUI/);
});

test('checkpointDeleteTarget: the SAME pill undeployed aims at the training save', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'b' };
  const pill = { step: 1000, filename: 'lora_001000.safetensors', testable: false };
  const t = checkpointDeleteTarget(node, pill);
  assert.equal(t.kind, 'save');
  assert.equal(t.path, 'train/run-checkpoint/delete');       // the RUN save route
  assert.equal(t.body.filename, 'lora_001000.safetensors');
  assert.match(t.label, /training save/i);
});

test('checkpointDeleteTarget: a cloud save carries cloud_run_id, a cloud deploy does not', () => {
  const node = { source: 'cloud', run_id: 42, status: 'done', train_type: 'zimage', variant: 'base', base_model: '' };
  const save = checkpointDeleteTarget(node, { step: 2000, filename: 'e2000.safetensors' });
  assert.equal(save.body.cloud_run_id, 42);
  const deployed = checkpointDeleteTarget(node, { step: 2000, filename: 'e2000.safetensors',
    testable: true, deployed_filename: 'z image/lds42_e2000_rc42.safetensors' });
  assert.equal(deployed.kind, 'deployed');
  assert.equal('cloud_run_id' in deployed.body, false);      // the loras folder has no run scope
});

test('checkpointDeleteTarget: nothing to delete → no action', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'b' };
  assert.equal(checkpointDeleteTarget(node, { step: 1, filename: 'a.safetensors', present: false }), null);
  // deployed but the deployed copy's own name is unknown → the route would reject it
  assert.equal(checkpointDeleteTarget(node, { step: 1, filename: 'a.safetensors', testable: true }), null);
  // a cloud run still syncing epochs down keeps its saves
  const training = { source: 'cloud', run_id: 7, status: 'training', train_type: 'zimage' };
  assert.equal(checkpointDeleteTarget(training, { step: 1, filename: 'a.safetensors' }), null);
});

test('describeCheckpointDelete names the target of the moment and what survives', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'b' };
  const deployed = describeCheckpointDelete(node,
    { step: 1000, filename: 'lora_001000.safetensors', testable: true,
      deployed_filename: 'sdxl/lora_nova_000001000_rl7.safetensors' });
  assert.equal(deployed.kind, 'deployed');
  assert.match(deployed.message, /REMOVE FROM COMFYUI/);
  assert.match(deployed.message, /training save in the run folder is KEPT/);
  assert.match(deployed.message, /frees no space/);
  assert.match(deployed.message, /trash/i);
  assert.doesNotMatch(deployed.message, /DELETE THE TRAINING SAVE/);

  const save = describeCheckpointDelete(node, { step: 1000, filename: 'lora_001000.safetensors' });
  assert.equal(save.kind, 'save');
  assert.match(save.message, /DELETE THE TRAINING SAVE/);
  assert.match(save.message, /isn't imported/);
  assert.match(save.message, /trash — recoverable until you empty it in Settings/);
  assert.doesNotMatch(save.message, /REMOVE FROM COMFYUI/);
});

test('describeCheckpointDelete warns on ★ best settings, wording per target', () => {
  const node = { source: 'local', train_type: 'sdxl', variant: 'base', base_model: 'b' };
  const pill = { step: 1000, filename: 'lora_001000.safetensors', testable: true,
    deployed_filename: 'sdxl/lora_001000.safetensors' };
  const best = describeCheckpointDelete(node, pill, { bestSettingsLora: 'loras/lora_001000.safetensors' });
  assert.equal(best.isBest, true);
  assert.match(best.message, /★ BEST SETTINGS/);
  assert.match(best.message, /saved combo will stop working/);
  const plain = describeCheckpointDelete(node, pill, { bestSettingsLora: 'loras/other.safetensors' });
  assert.equal(plain.isBest, false);
  assert.doesNotMatch(plain.message, /BEST SETTINGS/);
});

test('describeCheckpointDelete: null when there is nothing to delete', () => {
  assert.equal(describeCheckpointDelete({ source: 'local' }, { step: 1 }), null);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  SHOT_PRESETS_STORAGE_KEY,
  ShotPresetValidationError,
  applyShotPreset,
  deleteShotPreset,
  loadShotPresets,
  renameShotPreset,
  saveShotPreset,
} from './shotPresets.js';

const storage = (value) => ({ getItem: () => value });
const custom = { id: 'custom_1', label: 'Custom pose', prompt: 'pose', framing: 'body' };

test('malformed or unknown stored payloads are ignored', () => {
  assert.deepEqual(loadShotPresets(storage('{broken')), []);
  assert.deepEqual(loadShotPresets(storage(JSON.stringify({ version: 99, presets: [{}] }))), []);
  assert.equal(SHOT_PRESETS_STORAGE_KEY, 'datasetCustomPresetsV1');
});

test('save validates name, selection and duplicate names', () => {
  assert.throws(() => saveShotPreset([], '', ['a'], []), ShotPresetValidationError);
  assert.throws(() => saveShotPreset([], 'Empty', [], []), /selection/i);
  const saved = saveShotPreset([], 'My mix', ['a'], []);
  assert.throws(() => saveShotPreset(saved, ' my MIX ', ['b'], []), /already exists/i);
});

test('save snapshots selected custom shots and apply restores missing definitions', () => {
  const [preset] = saveShotPreset([], 'Portrait set', ['builtin_1', custom.id], [custom]);
  assert.deepEqual(preset.customShots, [custom]);
  assert.deepEqual(applyShotPreset(preset, []), {
    selectedIds: ['builtin_1', custom.id], customShots: [custom],
  });
});

test('rename validates duplicates and delete removes only the requested preset', () => {
  let presets = saveShotPreset([], 'One', ['a'], []);
  presets = saveShotPreset(presets, 'Two', ['b'], []);
  assert.throws(() => renameShotPreset(presets, presets[0].id, 'two'), /already exists/i);
  const renamed = renameShotPreset(presets, presets[0].id, 'First');
  assert.equal(renamed[0].name, 'First');
  assert.deepEqual(deleteShotPreset(renamed, renamed[0].id).map((item) => item.name), ['Two']);
});

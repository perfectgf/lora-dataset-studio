import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

import {
  DENSE_PATH_PAUSE_MS, denseChoiceSaver, denseChoiceTiming, densePickerSetters,
} from './denseChoice.js';

function picker({ variant = 'base', base = '', customBase = false } = {}) {
  const saved = [];
  const saver = denseChoiceSaver();
  const setters = densePickerSetters({
    variant, base, customRef: { current: customBase },
    setVariant: () => {}, setBase: () => {}, setCustomBase: () => {},
    saver,
    persist: async (selection, options) => { saved.push({ ...selection, ...options }); },
  });
  return { saved, saver, setters };
}

test('a pick is saved at once, a typed path only when it reads as a complete file', () => {
  assert.equal(denseChoiceTiming({ baseModel: '', customBase: false }), 'now');
  assert.equal(denseChoiceTiming({ baseModel: 'krea/model.safetensors', customBase: false }), 'now');
  for (const typed of ['', 'C', 'C:\\models\\kr', 'model.safetensors', 'models/model.safetensors']) {
    assert.equal(denseChoiceTiming({ baseModel: typed, customBase: true }), null, typed);
  }
  for (const typed of ['C:\\models\\krea.safetensors', 'D:/krea.SAFETENSORS', '/home/me/krea.safetensors']) {
    assert.equal(denseChoiceTiming({ baseModel: typed, customBase: true }), 'pause', typed);
  }
});

test('Raw, Turbo and an official pick reach the server, with the rest of the selection', async () => {
  const { saved, setters } = picker({ variant: 'base', base: 'krea/own.safetensors' });
  await setters.setVariant('turbo');
  assert.deepEqual(saved, [{ variant: 'turbo', baseModel: 'krea/own.safetensors', quiet: false }]);

  const leaving = picker({ variant: 'base', base: 'C:\\models\\krea.safetensors', customBase: true });
  leaving.setters.setCustomBase(false);   // the plugin's order for an official pick
  leaving.setters.setBase('');
  assert.deepEqual(leaving.saved, [{ variant: 'base', baseModel: '', quiet: false }]);
});

test('choosing Custom weights saves nothing until a complete path is typed', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { saved, setters } = picker();
  setters.setCustomBase(true);            // the plugin's order, in one event
  setters.setBase('');
  for (const typed of ['C', 'C:\\mo', 'C:\\models\\krea.safet']) setters.setBase(typed);
  t.mock.timers.tick(DENSE_PATH_PAUSE_MS * 2);
  assert.deepEqual(saved, []);

  setters.setBase('C:\\models\\krea.safetensors');
  t.mock.timers.tick(DENSE_PATH_PAUSE_MS - 1);
  assert.deepEqual(saved, []);
  t.mock.timers.tick(1);
  assert.deepEqual(saved, [{ variant: 'base', baseModel: 'C:\\models\\krea.safetensors', quiet: true }]);
});

test('a path edited again, or a later pick, cancels the save still waiting for its pause', (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const { saved, setters } = picker({ customBase: true });
  setters.setBase('C:\\models\\one.safetensors');
  setters.setBase('C:\\models\\one.safetensors.bak');
  t.mock.timers.tick(DENSE_PATH_PAUSE_MS);
  assert.deepEqual(saved, []);

  setters.setBase('C:\\models\\two.safetensors');
  setters.setCustomBase(false);
  setters.setBase('krea/listed.safetensors');
  t.mock.timers.tick(DENSE_PATH_PAUSE_MS);
  assert.deepEqual(saved.map((s) => s.baseModel), ['krea/listed.safetensors']);
});

test('one request at a time, and the last choice wins', async () => {
  const saver = denseChoiceSaver();
  const order = [];
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const first = saver.now(async () => { order.push('raw'); await gate; });
  saver.now(async () => { order.push('turbo'); });
  saver.now(async () => { order.push('raw again'); });
  assert.deepEqual(order, ['raw']);
  release();
  await first;
  assert.deepEqual(order, ['raw', 'raw again']);
});

test('the panel hands these setters to the picker and reports what it saved', () => {
  const panel = fs.readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');
  const hook = fs.readFileSync(new URL('../../hooks/useDataset.js', import.meta.url), 'utf8');
  assert.match(panel, /setVariant=\{densePicker\.setVariant\}/);
  assert.match(panel, /setBase=\{densePicker\.setBase\}/);
  assert.match(panel, /setCustomBase=\{densePicker\.setCustomBase\}/);
  assert.match(panel, /setDenseSelectionRevision\(\(n\) => n \+ 1\)/);
  // Switching mode must not let a waiting path turn full-model back on.
  assert.match(panel, /const onTrainingModeChange = async \(nextValue\) => \{\s+denseSaver\.drop\(\);/);
  assert.match(hook, /setDatasetTrainingMode = useCallback\(async \(trainingMode, selection = \{\}, \{ quiet = false \} = \{\}\)/);
});

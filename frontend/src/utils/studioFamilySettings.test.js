import assert from 'node:assert/strict';
import test from 'node:test';
import {
  defaultModelSelection, modelSelectionError, readComparisonAxis, showsNegativePrompt, supportsNegativePrompt,
  writeComparisonAxis,
} from './studioFamilySettings.js';

const storage = (initial = {}) => {
  const entries = new Map(Object.entries(initial));
  return { getItem: (key) => entries.get(key) ?? null,
    setItem: (key, value) => entries.set(key, value) };
};

test('server capability controls negative prompts, including explicit disable and future families', () => {
  assert.equal(supportsNegativePrompt({ negative_prompt: true }, 'anima'), true);
  assert.equal(supportsNegativePrompt({ negative_prompt: true }, 'qwenimage21'), true);
  assert.equal(supportsNegativePrompt({ negative_prompt: true }, 'future-family'), true);
  assert.equal(supportsNegativePrompt({ negative_prompt: false }, 'zimage'), false);
  assert.equal(supportsNegativePrompt(null, 'flux'), false);
  assert.equal(supportsNegativePrompt(null, 'zimage'), true);
});

test('a negative that only acts above CFG 1 is shown once a selected CFG exceeds it', () => {
  const krea = { negative_prompt: true, negative_prompt_needs_cfg: true };
  assert.equal(showsNegativePrompt(krea, 'krea', [1]), false);
  assert.equal(showsNegativePrompt(krea, 'krea', [1, 3]), true);
  assert.equal(showsNegativePrompt(krea, 'krea', [4]), true); // a Raw base's own default
  assert.equal(showsNegativePrompt(krea, 'krea', null), false);
  assert.equal(showsNegativePrompt({ negative_prompt: true }, 'zimage', [1]), true);
  assert.equal(showsNegativePrompt({ negative_prompt: false }, 'krea', [3]), false);
});

test('missing selected models are reported without substituting an installed model', () => {
  const models = [{ value: 'new-base.safetensors' }, { value: '' }];
  assert.match(modelSelectionError(models, ['old-base.safetensors']), /old-base\.safetensors/);
  assert.equal(modelSelectionError(models, ['new-base.safetensors', '']), null);
  assert.equal(modelSelectionError(null, ['old-base.safetensors']), null);
  assert.match(modelSelectionError([], ['old-base.safetensors']), /unavailable/);
});

test('a missing configured model is never replaced by the first installed model', () => {
  const models = [{ value: 'another-base.safetensors' }];
  const selected = defaultModelSelection(models, 'configured-base.safetensors');
  assert.deepEqual(selected, ['configured-base.safetensors']);
  assert.match(modelSelectionError(models, selected), /configured-base\.safetensors/);
  assert.deepEqual(defaultModelSelection([{ value: '' }], ''), ['']);
  assert.deepEqual(defaultModelSelection(models, null), ['another-base.safetensors']);
});

test('new families do not inherit legacy Turbo settings and keep separate comparison axes', () => {
  const saved = storage({ studioComp_steps: '[8]', studioComp_cfgs: '[1]' });
  for (const family of ['flux', 'anima', 'qwenimage21', 'moss']) {
    assert.equal(readComparisonAxis(saved, 'studioComp_steps', family), null);
    assert.equal(readComparisonAxis(saved, 'studioComp_cfgs', family), null);
  }
  writeComparisonAxis(saved, 'studioComp_steps', 'anima', [24, 32]);
  writeComparisonAxis(saved, 'studioComp_steps', 'qwenimage21', [40]);
  assert.deepEqual(readComparisonAxis(saved, 'studioComp_steps', 'anima'), [24, 32]);
  assert.deepEqual(readComparisonAxis(saved, 'studioComp_steps', 'qwenimage21'), [40]);
});

test('legacy preference migration belongs to one family and retains the previous storage keys', () => {
  const saved = storage({ studioComp_steps: '[12]', studioComp_cfgs: '[2]' });
  assert.deepEqual(readComparisonAxis(saved, 'studioComp_steps', 'zimage'), [12]);
  assert.deepEqual(readComparisonAxis(saved, 'studioComp_cfgs', 'zimage'), [2]);
  assert.equal(readComparisonAxis(saved, 'studioComp_steps', 'sdxl'), null);
  writeComparisonAxis(saved, 'studioComp_steps', 'zimage', null);
  assert.equal(readComparisonAxis(saved, 'studioComp_steps', 'zimage'), null);
  assert.equal(saved.getItem('studioComp_steps'), '[12]');
});

test('malformed saved axes and unavailable storage defer to server defaults', () => {
  assert.equal(readComparisonAxis(storage({ studioComp_steps_anima: '["8"]' }), 'studioComp_steps', 'anima'), null);
  assert.equal(readComparisonAxis(undefined, 'studioComp_steps', 'anima'), null);
  assert.doesNotThrow(() => writeComparisonAxis(undefined, 'studioComp_steps', 'anima', [30]));
});

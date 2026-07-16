import test from 'node:test';
import assert from 'node:assert/strict';
import {
  checkpointSelectionMatchesTraining,
  checkpointVariantLabel,
  checkpointVariantOptions,
  cloudTrainingLaunchPayload,
  defaultCheckpointBase,
  defaultCheckpointVariant,
  loraFolderLabel,
  normalizeCheckpointVariant,
  trainingRunSelection,
  trainFamilyLabel,
} from './checkpointBrowser.js';

test('results browser chooses the official base when a family provides one', () => {
  assert.equal(defaultCheckpointBase([{ value: 'custom', label: 'Custom' }, { value: '', label: 'Official' }]), '');
  assert.equal(defaultCheckpointBase([{ value: 'juggernaut', label: 'Juggernaut' }]), 'juggernaut');
  assert.equal(defaultCheckpointBase([]), '');
});

test('results selection only matches training when family, base and variant match', () => {
  assert.equal(checkpointSelectionMatchesTraining('krea', '', 'base', 'krea', '', 'base'), true);
  assert.equal(checkpointSelectionMatchesTraining('krea', '', 'base', 'zimage', '', 'base'), false);
  assert.equal(checkpointSelectionMatchesTraining('sdxl', 'a.safetensors', 'turbo', 'sdxl', 'b.safetensors', 'turbo'), false);
  assert.equal(checkpointSelectionMatchesTraining('zimage', '', 'turbo', 'zimage', '', 'base'), false);
});

test('checkpoint variants have safe family-aware defaults and labels', () => {
  assert.equal(defaultCheckpointVariant('zimage'), 'turbo');
  assert.equal(defaultCheckpointVariant('krea'), 'base');
  assert.equal(defaultCheckpointVariant('flux2klein'), '4b');
  assert.equal(normalizeCheckpointVariant('zimage', '4b'), 'turbo');
  assert.equal(checkpointVariantLabel('zimage', 'base'), 'Base · non-distilled');
  assert.equal(checkpointVariantLabel('krea', 'base'), 'Raw');
  assert.deepEqual(checkpointVariantOptions('flux2klein').map((item) => item.value), ['4b', '9b']);
});

test('run selection payload preserves the official empty base and variant', () => {
  assert.deepEqual(trainingRunSelection('', 'zimage', 'deturbo'), {
    base_model: '', train_type: 'zimage', variant: 'deturbo',
  });
  assert.deepEqual(trainingRunSelection(undefined, 'krea', 'base'), {
    train_type: 'krea', variant: 'base',
  });
});

test('cloud payload always sends the selected base to the server guard', () => {
  assert.deepEqual(cloudTrainingLaunchPayload({
    baseModel: '', variant: 'turbo', trainType: 'zimage', masked: false,
  }), {
    base_model: '', variant: 'turbo', train_type: 'zimage', masked: false,
  });
  assert.equal(cloudTrainingLaunchPayload({
    baseModel: 'custom/model.safetensors', variant: 'base', trainType: 'zimage',
  }).base_model, 'custom/model.safetensors');
});

test('family labels and ComfyUI folders stay tied to the results filter', () => {
  assert.equal(trainFamilyLabel('flux2klein'), 'FLUX.2 Klein');
  assert.equal(loraFolderLabel('flux2klein'), 'loras/flux2klein');
  assert.equal(trainFamilyLabel('unknown'), 'Z-Image');
  assert.equal(loraFolderLabel('unknown'), 'loras/z image');
});

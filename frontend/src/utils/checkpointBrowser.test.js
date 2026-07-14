import test from 'node:test';
import assert from 'node:assert/strict';
import {
  checkpointSelectionMatchesTraining,
  defaultCheckpointBase,
  loraFolderLabel,
  trainFamilyLabel,
} from './checkpointBrowser.js';

test('results browser chooses the official base when a family provides one', () => {
  assert.equal(defaultCheckpointBase([{ value: 'custom', label: 'Custom' }, { value: '', label: 'Official' }]), '');
  assert.equal(defaultCheckpointBase([{ value: 'juggernaut', label: 'Juggernaut' }]), 'juggernaut');
  assert.equal(defaultCheckpointBase([]), '');
});

test('results selection only matches training when both family and base match', () => {
  assert.equal(checkpointSelectionMatchesTraining('krea', '', 'krea', ''), true);
  assert.equal(checkpointSelectionMatchesTraining('krea', '', 'zimage', ''), false);
  assert.equal(checkpointSelectionMatchesTraining('sdxl', 'a.safetensors', 'sdxl', 'b.safetensors'), false);
});

test('family labels and ComfyUI folders stay tied to the results filter', () => {
  assert.equal(trainFamilyLabel('flux2klein'), 'FLUX.2 Klein');
  assert.equal(loraFolderLabel('flux2klein'), 'loras/flux2klein');
  assert.equal(trainFamilyLabel('unknown'), 'Z-Image');
  assert.equal(loraFolderLabel('unknown'), 'loras/z image');
});

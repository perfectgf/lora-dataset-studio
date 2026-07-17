import test from 'node:test';
import assert from 'node:assert/strict';
import { deriveSetupSteps, kleinMissingLabels, KLEIN_ASSET_LABELS } from './useSetupSteps.js';

const comfyStep = (comfyui) => deriveSetupSteps({ comfyui }).find((s) => s.id === 'comfyui');

test('Klein readiness needs the full trio, not just the UNET', () => {
  // UNET landed, but the backend still lists the text-encoder + VAE as missing:
  // the step must NOT go "ready", and hasKlein must be false so "Nothing to do"
  // and the disappearing-download-buttons bugs cannot fire.
  const step = comfyStep({
    reachable: true,
    models: { klein: ['flux-2-klein-9b-fp8.safetensors'] },
    klein_missing: ['klein_text_encoder', 'klein_vae', 'klein_lora'],
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial'); // reachable but incomplete
  assert.deepEqual(step.kleinMissing, ['klein_text_encoder', 'klein_vae', 'klein_lora']);
});

test('all three weights present -> ready even with the recommended LoRA still missing', () => {
  const step = comfyStep({
    reachable: true,
    models: { klein: ['flux-2-klein-9b-fp8.safetensors'] },
    klein_missing: ['klein_lora'], // recommended only, does not gate the engine
  });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
});

test('reachable with nothing missing -> ready', () => {
  const step = comfyStep({ reachable: true, klein_missing: [] });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
});

test('unreachable ComfyUI is "available" regardless of assets on disk', () => {
  const step = comfyStep({ reachable: false, klein_missing: [] });
  assert.equal(step.status, 'available');
  assert.equal(step.reachable, false);
});

test('legacy payload without klein_missing falls back to the UNET scan', () => {
  const withUnet = comfyStep({ reachable: true, models: { klein: ['a.safetensors'] } });
  assert.equal(withUnet.hasKlein, true);
  assert.deepEqual(withUnet.kleinMissing, []);

  const noUnet = comfyStep({ reachable: true, models: { klein: [] } });
  assert.equal(noUnet.hasKlein, false);
  assert.deepEqual(noUnet.kleinMissing, ['klein_model']);
});

test('kleinMissingLabels maps required assets to words in a stable order', () => {
  // Order is canonical (model, text encoder, VAE), independent of input order.
  assert.deepEqual(
    kleinMissingLabels(['klein_vae', 'klein_lora', 'klein_model']),
    ['model', 'VAE'],
  );
  assert.deepEqual(kleinMissingLabels(['klein_text_encoder']), ['text encoder']);
  assert.deepEqual(kleinMissingLabels([]), []);
  assert.deepEqual(kleinMissingLabels(undefined), []);
  // The recommended LoRA is never surfaced as a required gap.
  assert.deepEqual(kleinMissingLabels(['klein_lora']), []);
  assert.equal(KLEIN_ASSET_LABELS.klein_text_encoder, 'text encoder');
});

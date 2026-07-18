import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deriveSetupSteps, kleinMissingLabels, KLEIN_ASSET_LABELS,
  comfyuiDirVerdict, COMFYUI_SKIP_LOST, COMFYUI_SKIP_KEPT,
} from './useSetupSteps.js';

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

test('a present-but-INVALID required asset keeps the step from going green', () => {
  // The #help incident: the UNET file is on disk (so klein_missing is empty) but
  // it is really an HTML licence-gate page. The step must NOT go "ready" — otherwise
  // Setup stays green and the user hits the cryptic UNETLoader crash at generate.
  const step = comfyStep({
    reachable: true,
    klein_missing: [],
    klein_invalid: [{
      asset: 'klein_model', filename: 'flux-2-klein-9b-fp8.safetensors', blocking: true,
      verdict: 'html_or_text', reason: 'flux-2-klein-9b-fp8.safetensors is not a real model (looks like an HTML page …)',
    }],
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial'); // reachable but a required weight is broken
  assert.equal(step.kleinInvalid.length, 1);
});

test('an advisory too_small invalid does NOT gate readiness', () => {
  const step = comfyStep({
    reachable: true,
    klein_missing: [],
    klein_invalid: [{ asset: 'klein_model', filename: 'k.safetensors', blocking: false, verdict: 'too_small', reason: 'k.safetensors is only 10 B …' }],
  });
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

// --- Conscious "continue without ComfyUI" skip (Setup Volet 2) --------------

test('skipped ComfyUI (flag set, unreachable) renders a neutral "skipped" status', () => {
  const step = comfyStep({ skipped: true, reachable: false });
  assert.equal(step.status, 'skipped');
  assert.equal(step.skipped, true);
});

test('a running ComfyUI is never shown as skipped even if the flag lingers', () => {
  // The backend already annuls the skip once a dir is set; belt-and-suspenders on the
  // client: a reachable server always shows its real status, never "skipped".
  const step = comfyStep({ skipped: true, reachable: true, klein_missing: [] });
  assert.equal(step.skipped, false);
  assert.equal(step.status, 'ready');
});

test('no skip flag -> normal available status, skipped false', () => {
  const step = comfyStep({ reachable: false });
  assert.equal(step.skipped, false);
  assert.equal(step.status, 'available');
});

test('comfyuiDirVerdict maps each backend status to an actionable message', () => {
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy' }).tone, 'ok');
  assert.match(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy' }).message, /C:\/Comfy/);

  const nested = comfyuiDirVerdict({ status: 'nested', suggestion: 'C:/x/ComfyUI' });
  assert.equal(nested.tone, 'warn');
  assert.equal(nested.suggestion, 'C:/x/ComfyUI');   // drives the adopt button
  assert.match(nested.message, /launcher\/parent folder/);

  assert.match(comfyuiDirVerdict({ status: 'missing' }).message, /doesn't exist/);
  assert.match(comfyuiDirVerdict({ status: 'empty_dir' }).message, /empty/);
  assert.match(comfyuiDirVerdict({ status: 'not_comfyui' }).message, /isn't a ComfyUI install/);
  for (const s of ['missing', 'empty_dir', 'not_comfyui']) {
    assert.equal(comfyuiDirVerdict({ status: s }).tone, 'warn');
    assert.equal(comfyuiDirVerdict({ status: s }).suggestion, '');
  }
  // Blank / in-flight / unknown -> muted, nothing to render.
  assert.deepEqual(comfyuiDirVerdict({ status: 'empty' }), { tone: 'muted', suggestion: '', message: '' });
  assert.equal(comfyuiDirVerdict(null).message, '');
});

test('skip panel lists what turns off and what stays on', () => {
  assert.ok(COMFYUI_SKIP_LOST.length >= 4 && COMFYUI_SKIP_KEPT.length >= 4);
  const lost = COMFYUI_SKIP_LOST.join(' | ');
  assert.match(lost, /Klein/);
  assert.match(lost, /Test Studio/);
  const kept = COMFYUI_SKIP_KEPT.join(' | ');
  assert.match(kept, /Scraping/);
  assert.match(kept, /ai-toolkit/);
  assert.match(kept, /Hugging Face/);
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

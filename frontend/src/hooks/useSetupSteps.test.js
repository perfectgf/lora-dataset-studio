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

// --- installAllPlan (mirror of the backend orchestrator) --------------------
import { installAllPlan, INSTALL_ALL_ORDER } from './useSetupSteps.js';

// A fully-installed snapshot; each test flips just the pieces it needs MISSING.
const fullCaps = () => ({
  python: { ml_supported: true },
  face_scoring: true, masks: true, watermark_inpaint: true,
  ollama: { reachable: true, vision_model_ready: true, vision_model: 'qwen3-vl:8b' },
  comfyui: { dir_valid: true, klein_missing: [] },
});

test('installAllPlan is empty when everything installable is present', () => {
  assert.deepEqual(installAllPlan(fullCaps()), []);
});

test('installAllPlan folds null/empty caps to the always-runnable ML extras', () => {
  const mlOnly = ['face_scoring', 'masks', 'watermark_inpaint'];
  assert.deepEqual(installAllPlan(null), mlOnly);
  assert.deepEqual(installAllPlan({}), mlOnly);
});

test('installAllPlan skips face/masks on an unsupported Python but keeps watermark', () => {
  const caps = { ...fullCaps(), python: { ml_supported: false },
    face_scoring: false, masks: false, watermark_inpaint: false };
  assert.deepEqual(installAllPlan(caps), ['watermark_inpaint']);
});

test('installAllPlan queues the vision model only when Ollama is up and named', () => {
  const up = { ...fullCaps(),
    ollama: { reachable: true, vision_model_ready: false, vision_model: 'qwen3-vl:8b' } };
  assert.ok(installAllPlan(up).includes('ollama_model'));
  const noName = { ...fullCaps(),
    ollama: { reachable: true, vision_model_ready: false, vision_model: '' } };
  assert.ok(!installAllPlan(noName).includes('ollama_model'));
  const down = { ...fullCaps(),
    ollama: { reachable: false, vision_model_ready: false, vision_model: 'qwen3-vl:8b' } };
  assert.ok(!installAllPlan(down).includes('ollama_model'));
});

test('installAllPlan takes Klein weights only into a validated ComfyUI, in order', () => {
  const valid = { ...fullCaps(),
    comfyui: { dir_valid: true, klein_missing: ['klein_lora', 'klein_model', 'klein_vae'] } };
  assert.deepEqual(installAllPlan(valid), ['klein_model', 'klein_vae', 'klein_lora']);
  const invalid = { ...fullCaps(),
    comfyui: { dir_valid: false, klein_missing: ['klein_model'] } };
  assert.deepEqual(installAllPlan(invalid), []);
});

test('installAllPlan full order groups ML -> vision model -> Klein', () => {
  const caps = {
    python: { ml_supported: true },
    face_scoring: false, masks: false, watermark_inpaint: false,
    ollama: { reachable: true, vision_model_ready: false, vision_model: 'm' },
    comfyui: { dir_valid: true,
      klein_missing: ['klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora'] },
  };
  assert.deepEqual(installAllPlan(caps), INSTALL_ALL_ORDER);
});

// --- installCatalog (the full one-by-one install/reinstall menu) -------------
import { installCatalog } from './useSetupSteps.js';

const byAction = (cat) => Object.fromEntries(cat.map((c) => [c.action, c]));

test('installCatalog lists every app-installable component, present + available', () => {
  const cat = byAction(installCatalog(fullCaps()));
  // The eight components the app can install itself (never ComfyUI/Ollama/API keys).
  assert.deepEqual(
    installCatalog(fullCaps()).map((c) => c.action),
    ['face_scoring', 'masks', 'watermark_inpaint', 'ollama_model',
      'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora'],
  );
  // Everything installed in fullCaps -> every tile present, and available to REINSTALL.
  for (const c of Object.values(cat)) {
    assert.equal(c.present, true, `${c.action} present`);
    assert.equal(c.available, true, `${c.action} available to reinstall`);
  }
});

test('installCatalog stays fully available for reinstall when all is green', () => {
  // The menu must never collapse once installed — each item can always be repaired.
  const cat = installCatalog(fullCaps());
  assert.ok(cat.length === 8 && cat.every((c) => c.available));
});

test('installCatalog marks missing ML extras not-present but still available', () => {
  const cat = byAction(installCatalog({ ...fullCaps(),
    face_scoring: false, masks: false, watermark_inpaint: false }));
  for (const a of ['face_scoring', 'masks', 'watermark_inpaint']) {
    assert.equal(cat[a].present, false);
    assert.equal(cat[a].available, true);   // installable now (supported Python)
  }
});

test('installCatalog blocks fresh ML installs on an unsupported Python, with a hint', () => {
  const cat = byAction(installCatalog({ ...fullCaps(),
    python: { ml_supported: false, ml_range: '3.10–3.12' },
    face_scoring: false, masks: false }));
  // Can't install into the app's out-of-range Python -> unavailable + an actionable hint.
  assert.equal(cat.face_scoring.available, false);
  assert.match(cat.face_scoring.hint, /3\.10–3\.12/);
  // watermark auto-provisions its own venv, so it stays available regardless.
  assert.equal(cat.watermark_inpaint.available, true);
});

test('installCatalog still lets you REPAIR a present ML extra on an unsupported Python', () => {
  // A face-scoring already installed (into a dedicated env) can be reinstalled/repaired
  // even when the app's own Python is out of the wheel range.
  const cat = byAction(installCatalog({ ...fullCaps(),
    python: { ml_supported: false, ml_range: '3.10–3.12' }, face_scoring: true }));
  assert.equal(cat.face_scoring.present, true);
  assert.equal(cat.face_scoring.available, true);
});

test('installCatalog gates the vision model on a reachable, named Ollama', () => {
  const down = byAction(installCatalog({ ...fullCaps(),
    ollama: { reachable: false, vision_model_ready: false, vision_model: 'm' } }));
  assert.equal(down.ollama_model.available, false);
  assert.match(down.ollama_model.hint, /Start Ollama/);
  const noName = byAction(installCatalog({ ...fullCaps(),
    ollama: { reachable: true, vision_model_ready: false, vision_model: '' } }));
  assert.equal(noName.ollama_model.available, false);
  assert.match(noName.ollama_model.hint, /model name/);
});

test('installCatalog gates Klein weights on a validated ComfyUI', () => {
  const invalid = byAction(installCatalog({ ...fullCaps(),
    comfyui: { dir_valid: false, klein_missing: ['klein_model'] } }));
  for (const a of ['klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora']) {
    assert.equal(invalid[a].available, false);
    assert.match(invalid[a].hint, /valid ComfyUI folder/);
  }
  const valid = byAction(installCatalog({ ...fullCaps(),
    comfyui: { dir_valid: true, klein_missing: ['klein_vae'] } }));
  assert.equal(valid.klein_vae.present, false);       // still missing
  assert.equal(valid.klein_vae.available, true);      // installable into the valid tree
  assert.equal(valid.klein_model.present, true);      // not in klein_missing -> installed
});

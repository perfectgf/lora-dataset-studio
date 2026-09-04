import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deriveSetupSteps, deriveCapabilitySummary, kleinMissingLabels, KLEIN_ASSET_LABELS,
  comfyuiDirVerdict, COMFYUI_SKIP_LOST, COMFYUI_SKIP_KEPT,
  OLLAMA_SKIP_LOST, OLLAMA_SKIP_KEPT, ollamaSkipKept, ollamaGateReason,
  aitoolkitVerdict, AITOOLKIT_INSTALL_STEPS, SETUP_STEP_IDS,
} from './useSetupSteps.js';
// installAllPlan / installCatalog are imported further down, next to their own
// sections; kreaInstallPlan has no other importer here.
import { kreaInstallPlan } from './useSetupSteps.js';
import { localEngineUnavailableReason } from '../utils/localEngineReason.js';
import fs from 'node:fs';

const comfyStep = (comfyui) => deriveSetupSteps({ comfyui }).find((s) => s.id === 'comfyui');
// The same step, driven by a FULL capabilities payload — engines included. That is
// the shape the app actually receives, and the one where Setup reads the backend's
// verdict instead of re-deriving its own.
const comfyStepFull = (caps) => deriveSetupSteps(caps).find((s) => s.id === 'comfyui');
const qualityStep = (caps) => deriveSetupSteps(caps).find((s) => s.id === 'quality');

test('optional SigLIP2 readiness is visible without gating existing quality tools', () => {
  const required = {
    face_scoring: true, masks: true, watermark_inpaint: true, bank_scoring: true,
  };
  const without = qualityStep({ ...required, bank_siglip2: false });
  assert.equal(without.status, 'ready');
  assert.equal(without.bankSiglip2, false);
  assert.match(without.unlocks.join(' | '), /SigLIP2/);
  const withEngine = qualityStep({ ...required, bank_siglip2: true });
  assert.equal(withEngine.status, 'ready');
  assert.equal(withEngine.bankSiglip2, true);
});

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

test('a widget value this ComfyUI does not offer keeps the step from going green', () => {
  // Reported by IndependentProcess0 (Reddit). Every weight is on disk and every
  // node class exists, so both existing checks pass — and the first generation
  // still dies, because the graph pins a scheduler this ComfyUI never had. Since
  // nothing is substituted (a scheduler changes the render), Setup is where the
  // user has to find out, not the ComfyUI console after a failed batch.
  const step = comfyStep({
    reachable: true,
    klein_missing: [],
    klein_unsupported_enums: [{
      node_id: '77', class_type: 'KSampler', input: 'scheduler', value: 'beta57',
      pack: 'RES4LYF', url: 'https://github.com/ClownsharkBatwing/RES4LYF',
    }],
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial');
  assert.equal(step.unsupportedEnums.length, 1);
});

test('an install that offers every pinned value is unaffected', () => {
  // No regression for people whose ComfyUI already has the value: same render,
  // same green step. The server sends an empty list both when it verified and
  // found nothing AND when it could not verify at all (fail-open).
  const ready = comfyStep({ reachable: true, klein_missing: [], klein_unsupported_enums: [] });
  assert.equal(ready.hasKlein, true);
  assert.equal(ready.status, 'ready');
  assert.deepEqual(ready.unsupportedEnums, []);
  // An older backend that doesn't publish the field at all must not gate either.
  const legacy = comfyStep({ reachable: true, klein_missing: [] });
  assert.equal(legacy.hasKlein, true);
  assert.deepEqual(legacy.unsupportedEnums, []);
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

/* --- "Ready" must mean the same thing on every screen ------------------------

   Reported by zigzag4794 (Discord): Setup showed ✓ Installed for every required
   Klein model, ComfyUI tested OK, the 9.5 GB file was in models/unet/klein/ — and
   the Generate page kept Klein greyed out with "⚠ Klein model missing — download
   it in the Setup step". Both screens were sincere. `/api/capabilities` had the
   answer all along: both model files under `klein_invalid`, verdict
   `truncated_or_garbage`. The integrity validator existed and worked; the Setup
   screen simply never asked it, and judged each file on presence alone.

   These tests pin the contract, not the wording: ONE verdict (the backend's),
   consumed by both, and a sentence that names the real gap. */

const CORRUPT_KLEIN = [
  { asset: 'klein_model', filename: 'flux-2-klein-9b-kv-fp8.safetensors',
    verdict: 'truncated_or_garbage', blocking: true,
    reason: 'flux-2-klein-9b-kv-fp8.safetensors is shorter than its header declares' },
];

const zigzagCaps = () => ({
  engines: { klein: false, krea: false },   // what the BACKEND concluded
  comfyui: {
    reachable: true, dir_valid: true,
    klein_missing: ['klein_lora'],          // recommended only — nothing required is absent
    klein_unsupported_enums: [],
    klein_invalid: CORRUPT_KLEIN,
  },
});

test('a corrupted weight keeps Setup from claiming Klein is ready', () => {
  const step = comfyStepFull(zigzagCaps());
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'partial');
  assert.equal(step.kleinBroken.length, 1);
  assert.equal(step.kleinFilesReady, false);
});

test('Setup and the generation panel give the SAME reason, word for word', () => {
  // Not "both say something is wrong" — literally the same sentence, because both
  // read utils/localEngineReason.js. Two wordings for one gap is how the user ends
  // up believing they are two problems.
  const caps = zigzagCaps();
  assert.equal(comfyStepFull(caps).kleinReason,
    localEngineUnavailableReason('klein', caps));
});

test('Setup never says "missing" about a file that is on the disk', () => {
  const reason = comfyStepFull(zigzagCaps()).kleinReason;
  assert.doesNotMatch(reason, /missing/i);
  assert.match(reason, /flux-2-klein-9b-kv-fp8\.safetensors/);
  assert.match(reason, /Delete it and download it again/);
});

test('the install menu stops badging an unreadable file "✓ Installed"', () => {
  // THE screen zigzag was reading. `present: false` is not cosmetic — it is what
  // puts the asset back into the install plans so the button has work to do.
  const rows = installCatalog(zigzagCaps());
  const unet = rows.find((r) => r.action === 'klein_model');
  assert.equal(unet.present, false);
  assert.equal(unet.state, 'broken');
  assert.match(unet.stateLabel, /unreadable/);
  assert.match(unet.brokenReason, /flux-2-klein-9b-kv-fp8\.safetensors/);
  // A file that is genuinely fine is untouched.
  assert.equal(rows.find((r) => r.action === 'klein_vae').present, true);
});

test('a corrupted weight is re-queued by the one-click install, not skipped', () => {
  // Without this the plan is empty, the card says "everything is in place", and
  // the engine stays dark with no way forward from inside the app.
  assert.ok(installAllPlan(zigzagCaps()).includes('klein_model'));
  const kreaBroken = {
    engines: { krea: false },
    comfyui: { reachable: true, dir_valid: true, krea_missing: [],
      krea_nodes_installed: true, krea_nodes_missing: [],
      krea_invalid: [{ asset: 'krea_model', filename: 'k.safetensors',
        verdict: 'html_or_text', blocking: true }] },
  };
  assert.deepEqual(kreaInstallPlan(kreaBroken), ['krea_model']);
});

test('an unreachable ComfyUI produces "not checked", never a ✓', () => {
  // The unsupported-value and node probes fail OPEN by design — an unreachable
  // ComfyUI reports no gap rather than inventing one. So "no warnings" here is the
  // absence of an answer, not a clean bill of health, and the step must not read
  // as ready on the strength of it.
  const step = comfyStepFull({
    engines: { klein: false },
    comfyui: { reachable: false, dir_valid: true, klein_missing: [],
      klein_unsupported_enums: [], klein_invalid: [] },
  });
  assert.equal(step.hasKlein, false);
  assert.equal(step.status, 'available');
  assert.equal(step.kleinVerified, false);
  // …and the files themselves ARE fine, which is a different sentence.
  assert.equal(step.kleinFilesReady, true);
});

test('Setup does not second-guess the backend when it says the engine is up', () => {
  const step = comfyStepFull({
    engines: { klein: true },
    comfyui: { reachable: true, dir_valid: true, klein_missing: ['klein_lora'] },
  });
  assert.equal(step.hasKlein, true);
  assert.equal(step.status, 'ready');
  assert.equal(step.kleinReason, null);
  assert.equal(step.kleinVerified, true);
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
  assert.deepEqual(comfyuiDirVerdict({ status: 'empty' }),
    { tone: 'muted', suggestion: '', note: '', message: '' });
  assert.equal(comfyuiDirVerdict(null).message, '');
});

/* The wizard used to certify half the contract: it tested the URL and accepted the
   folder, but never checked that the app can actually PUT a file in ComfyUI's input
   folder — which is how every local engine hands over its source image. With ComfyUI
   in a second container that folder isn't shared, setup went green and the first
   generation died on a detail-free 500 (reported on Discord by nofaceman). */
test('a valid ComfyUI folder the app cannot write into warns without blocking', () => {
  const v = comfyuiDirVerdict({
    status: 'valid', resolved: 'C:/Comfy',
    input_check: { path: 'C:/Comfy/input', ok: false,
      problem: "ComfyUI's input folder is not writable from LoRA Dataset Studio: "
        + 'C:/Comfy/input. If ComfyUI runs in another container, in WSL or on another '
        + 'machine, this folder must be a shared volume visible to LoRA Dataset Studio '
        + 'at that exact path.' },
  });
  assert.equal(v.tone, 'ok');                 // still a valid install: NOT a blocker
  assert.match(v.message, /ComfyUI found/);
  assert.match(v.note, /not writable/);       // ...and the second half is said
  assert.match(v.note, /shared volume/);
});

test('a working install, or a backend that says nothing, adds no note', () => {
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy',
    input_check: { path: 'C:/Comfy/input', ok: true, problem: '' } }).note, '');
  // ok=null (nothing probed) and a missing field must never read as a failure
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy',
    input_check: { ok: null, problem: '' } }).note, '');
  assert.equal(comfyuiDirVerdict({ status: 'valid', resolved: 'C:/Comfy' }).note, '');
});

test('the wizard renders the input-folder note', () => {
  const jsx = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  assert.match(jsx, /v\.note/);
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
  // Three video probes, not two: the `video` install action ships PyAV AND a
  // bundled ffmpeg, so a "fully installed" snapshot has to assert both.
  video_decode: true, video_detect: true, video_encode: true,
  // The safe-zone pass's OCR half — its own action, its own probe.
  video_text: true,
  ollama: { reachable: true, vision_model_ready: true, vision_model: 'qwen3-vl:8b' },
  // reachable matters for the Krea node pack: an unreachable ComfyUI's node probe
  // fails open, so "nothing missing" from a stopped ComfyUI must not read as
  // "the pack is installed".
  comfyui: { dir_valid: true, reachable: true, klein_missing: [], krea_missing: [] },
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

test('installAllPlan never pulls the Ollama model implicitly', () => {
  for (const ollama of [
    { reachable: true, vision_model_ready: false, vision_model: 'qwen3-vl:8b' },
    { reachable: true, vision_model_ready: false, vision_model: '' },
    { reachable: false, vision_model_ready: false, vision_model: 'qwen3-vl:8b' },
  ]) {
    assert.ok(!installAllPlan({ ...fullCaps(), ollama }).includes('ollama_model'));
  }
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
  // Every component the app can install itself (never ComfyUI/Ollama/API keys).
  // The Krea 2 Edit rows land here too — the engine's ONE-CLICK install is its own
  // card, this menu is the per-piece repair path each of them also deserves.
  assert.deepEqual(
    installCatalog(fullCaps()).map((c) => c.action),
    ['face_scoring', 'masks', 'watermark_inpaint', 'video', 'shot_detect',
      'video_text', 'ollama_model',
      'klein_model', 'klein_text_encoder', 'klein_vae', 'klein_lora',
      'klein_enhancement_lora',
      'krea_nodes', 'krea_model', 'krea_text_encoder', 'krea_vae',
      'krea_identity_lora',
      // 📷 Camera angles — four rows, not five: its VAE is the krea_vae row
      // above (one file, one button).
      'camera_model', 'camera_lora', 'camera_speed_lora', 'camera_text_encoder',
      // 🎬 Video Test Studio — five weights and NO pack row. The lane's three
      // optional node packs are linked from its card, never installed by the
      // app: it downloads model files and does not add code to a ComfyUI.
      'h3_base', 'h3_text_encoder', 'h3_video_vae', 'h3_audio_vae',
      'h3_turbo_lora', 'h3_parasyte_lora', 'h3_dareties_lora',
      'lanpaint_nodes'],
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
  assert.ok(cat.length === 29 && cat.every((c) => c.available))   // 27 + the two arena accelerations;
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

test('the Video lane\'s three doors are rows of their own: ready, waiting, or missing with a door', () => {
  const row = (caps, label) => deriveCapabilitySummary(caps).find((s) => s.label === label);
  const DLSS = '✨ DLSS 5 neural rendering';
  const SMOOTH = '↗ Smooth (frame interpolation)';
  const LIVE = '🔴 Live lane (beta)';
  // Everything there.
  const on = { comfyui: { dir_valid: true, reachable: true, video_studio_ready: true,
    video_studio_options: { vfi: { available: true } } }, dlss5nr: { ready: true }, video_encode: true };
  for (const l of [DLSS, SMOOTH, LIVE]) assert.equal(row(on, l).ok, true, l);
  // ComfyUI down with the weights on disk: Smooth and Live wait (their
  // verdict needs the process), DLSS does not — it has a worker of its own.
  const off = { comfyui: { dir_valid: true, reachable: false, video_studio_missing: [] },
    dlss5nr: { ready: false }, video_encode: true };
  assert.equal(row(off, SMOOTH).pending, true);
  assert.match(row(off, SMOOTH).note, /launch ComfyUI/);
  assert.equal(row(off, LIVE).pending, true);
  assert.equal(row(off, DLSS).pending, undefined);
  assert.equal(row(off, DLSS).ok, false);
  // ComfyUI up, packs missing: Smooth is plainly missing, its door the video
  // install card (which lists the packs) — never "waiting".
  const noPacks = { comfyui: { dir_valid: true, reachable: true, video_studio_ready: true,
    video_studio_options: { vfi: { available: false, nodes: ['RIFE VFI'] } } }, video_encode: true };
  assert.equal(row(noPacks, SMOOTH).ok, false);
  assert.equal(row(noPacks, SMOOTH).pending, undefined);
  assert.equal(row(noPacks, SMOOTH).topic, 'setup-video-studio');
  // /object_info unreadable while ComfyUI answers (available: null) is not a
  // verdict either way: the row does not go green on it.
  const unread = { comfyui: { dir_valid: true, reachable: true, video_studio_ready: true,
    video_studio_options: { vfi: { available: null } } } };
  assert.equal(row(unread, SMOOTH).ok, false);
  // Weights there, ffmpeg not: Live names the gap and its door is the video
  // extra on the quality step, not the weights it already has.
  const noFfmpeg = { comfyui: { dir_valid: true, reachable: true, video_studio_ready: true }, video_encode: false };
  assert.equal(row(noFfmpeg, LIVE).ok, false);
  assert.match(row(noFfmpeg, LIVE).note, /ffmpeg/);
  assert.equal(row(noFfmpeg, LIVE).topic, 'setup-quality');
  // Weights missing: the door is the video install.
  const noWeights = { comfyui: { dir_valid: true, reachable: true, video_studio_ready: false,
    video_studio_missing: ['h3_unet'] }, video_encode: true };
  assert.equal(row(noWeights, LIVE).ok, false);
  assert.equal(row(noWeights, LIVE).pending, undefined);
  assert.equal(row(noWeights, LIVE).topic, 'setup-video-studio');
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


test('capability summary: configured-but-not-running ComfyUI reads as OK with a launch note', () => {
  const row = (caps, label) => deriveCapabilitySummary(caps).find((s) => s.label === label);
  // dir valid, API down -> Klein/Studio are pending with the launch note, not a bare miss
  const off = { comfyui: { dir_valid: true, reachable: false }, engines: {} };
  assert.equal(row(off, 'Klein (local)').pending, true);
  assert.match(row(off, 'Klein (local)').note, /launch ComfyUI/);
  assert.equal(row(off, '🖼️ Test Studio (images)').pending, true);
  // no ComfyUI configured at all -> honest plain miss, no note
  const none = { comfyui: { dir_valid: false, reachable: false }, engines: {} };
  assert.equal(row(none, 'Klein (local)').pending, undefined);
  // running -> plain ok, no note
  const on = { comfyui: { dir_valid: true, reachable: true }, engines: { klein: true }, studio_visible: true };
  assert.equal(row(on, 'Klein (local)').ok, true);
  assert.equal(row(on, 'Klein (local)').note, undefined);
});


// ── ai-toolkit setup copy ─────────────────────────────────────────────────────
// Reported on Reddit by Psyko_2000: an ai-toolkit installed by a community
// easy-install script ships a `python_embeded` folder and no venv, and the wizard
// answered "set up its Python venv per the README" — a cause the app never
// verified, and a remedy that install can never follow. He concluded the app
// REQUIRED a venv and asked in public, instead of filling the `aitoolkit.python`
// setting that already existed. These tests lock the CONTENT, since the content
// is what was false: showing the block proves nothing.
const trainStep = (aitoolkit) => deriveSetupSteps({ aitoolkit }).find((s) => s.id === 'training');
const DIR = '/opt/ai-toolkit';

test('a folder with no interpreter reports the FINDING, never "a venv is missing"', () => {
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true }), DIR);
  assert.equal(v.kind, 'no_interpreter');
  assert.match(v.headline, /no Python interpreter was found/i);
  assert.match(v.headline, new RegExp(DIR));
  // The headline must not diagnose a missing venv — that is the claim that was wrong.
  assert.doesNotMatch(v.headline, /venv/i);
});

test('both routes are offered, and the existing-interpreter one is not a footnote', () => {
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true }), DIR);
  // Route A: make one.
  assert.match(v.body, /create a venv/i);
  // Route B: use the one you already have — named, in the same sentence, with the
  // real-world shapes spelled out (Psyko_2000's is the portable/embedded one).
  assert.match(v.body, /already run ai-toolkit with/i);
  for (const shape of [/conda/i, /\buv\b/i, /system Python/i, /python_embeded/])
    assert.match(v.body, shape, `the body must name ${shape}`);
  // And a click that reaches the setting, not just prose about it.
  assert.equal(v.settingsSection, 'local-tools');
  assert.match(v.action, /ai-toolkit Python interpreter/);
  // The section id is a real Settings route, not a hopeful string.
  const registry = fs.readFileSync(new URL('../components/settings/registry.js', import.meta.url), 'utf8');
  const known = new Set([...registry.matchAll(/id: '([a-z0-9-]+)'/g)].map((m) => m[1]));
  assert.ok(known.size >= 8, 'settings registry did not parse');
  assert.ok(known.has(v.settingsSection));
});

test('an interpreter found in the folder becomes a one-click offer', () => {
  const found = ['/opt/ai-toolkit/python_embeded/python.exe'];
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true, python_candidates: found }), DIR);
  assert.deepEqual(v.candidates, found);
  // ...and the prose still stands on its own when nothing was found.
  const bare = aitoolkitVerdict(trainStep({ valid: false, dir_valid: true }), DIR);
  assert.deepEqual(bare.candidates, []);
  assert.ok(bare.body.length > 80);
});

test('a folder that is not an ai-toolkit checkout gets its own, different answer', () => {
  const v = aitoolkitVerdict(trainStep({ valid: false, dir_valid: false }), DIR);
  assert.equal(v.kind, 'not_a_checkout');
  assert.match(v.headline, /run\.py/);
  // Not the interpreter story: that would send the user to the wrong setting.
  assert.doesNotMatch(v.body, /interpreter/i);
});

test('the first-install steps do not define a working ai-toolkit as "has a venv"', () => {
  const all = AITOOLKIT_INSTALL_STEPS.map((s) => s.text).join(' ');
  assert.match(all, /venv/);              // still the documented happy path
  assert.match(all, /conda/i);            // but not the only one
  assert.match(all, /portable/i);
  assert.match(all, /ai-toolkit Python interpreter/);
  assert.ok(AITOOLKIT_INSTALL_STEPS.some((s) => s.command), 'the clone command survives');
});

test('SetupPage renders the verdict instead of hardcoding the old sentence', () => {
  const page = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  assert.doesNotMatch(page, /set up its Python venv per the README/);
  assert.doesNotMatch(page, /set up its venv per its README/);
  assert.match(page, /aitoolkitVerdict\(step, dir\)/);
  assert.match(page, /<SettingsLink section=\{verdict\.settingsSection\}/);
});

// --- The video extras are countable, listable and repairable ---------------------
// The wizard certified "12 of 12 capabilities ready" on a machine whose video
// lane could not cut a single file: the video capabilities were absent from the
// summary (so the denominator lied — the exact defect the Krea comment in
// deriveCapabilitySummary documents), absent from installCatalog (so the
// "Install or repair individually" screen had no row to click), and the wizard
// skipped the whole install screen because everything it DID count was green.
// Found live by the first real user, the day after the wave landed.
test('the capability summary counts the video pieces', () => {
  const rows = deriveCapabilitySummary({ video_decode: false, video_detect: false });
  const labels = rows.map((r) => r.label);
  const decode = rows.find((r) => /video/i.test(r.label) && /read/i.test(r.label));
  const detect = rows.find((r) => /shot detection/i.test(r.label));
  assert.ok(decode, `no video-decode row in the summary: ${labels.join(', ')}`);
  assert.ok(detect, `no shot-detection row in the summary: ${labels.join(', ')}`);
  assert.equal(decode.ok, false);
  assert.equal(detect.ok, false);
  const on = deriveCapabilitySummary({ video_decode: true, video_detect: true });
  assert.equal(on.find((r) => /shot detection/i.test(r.label)).ok, true);
});

// --- Four more installable capabilities were missing from "What's unlocked" -----
// setup_installer.INSTALL_ACTIONS already knew bank_scoring, bank_siglip2,
// watermark_detect and scrape_extras, and each already had a working Setup card
// (mlInstallCards.js) or, for scraping, a working install button on the Concept
// Sources panel — but none of the four had a row on the wizard's final screen.
// Same defect as the video pieces above, just for four different engines: a
// machine missing all four still certified "14 of 14 capabilities ready".
test('the capability summary counts bank scoring, SigLIP2, the watermark detector and scraping extras', () => {
  const off = deriveCapabilitySummary({
    bank_scoring: false, bank_siglip2: false, watermark_detect: false, scrape_deps: false,
  });
  const labels = off.map((r) => r.label);
  const bank = off.find((r) => /^Bank scoring/.test(r.label));
  const siglip = off.find((r) => /SigLIP2/.test(r.label));
  const wmDetect = off.find((r) => /Watermark detector/.test(r.label));
  const scrape = off.find((r) => /Scraping extras/.test(r.label));
  assert.ok(bank, `no bank-scoring row in the summary: ${labels.join(', ')}`);
  assert.ok(siglip, `no SigLIP2 row in the summary: ${labels.join(', ')}`);
  assert.ok(wmDetect, `no watermark-detector row in the summary: ${labels.join(', ')}`);
  assert.ok(scrape, `no scraping-extras row in the summary: ${labels.join(', ')}`);
  // Each reads its OWN capability key, not a neighbour's — the bug this guards
  // against is a copy-pasted row that always reports another engine's state.
  assert.equal(bank.ok, false);
  assert.equal(siglip.ok, false);
  assert.equal(wmDetect.ok, false);
  assert.equal(scrape.ok, false);
  assert.equal(bank.topic, 'setup-quality');
  assert.equal(siglip.topic, 'setup-quality');
  assert.equal(wmDetect.topic, 'setup-quality');
  assert.equal(scrape.topic, 'setup-quality');

  const on = deriveCapabilitySummary({
    bank_scoring: true, bank_siglip2: true, watermark_detect: true, scrape_deps: true,
  });
  assert.equal(on.find((r) => /^Bank scoring/.test(r.label)).ok, true);
  assert.equal(on.find((r) => /SigLIP2/.test(r.label)).ok, true);
  assert.equal(on.find((r) => /Watermark detector/.test(r.label)).ok, true);
  assert.equal(on.find((r) => /Scraping extras/.test(r.label)).ok, true);
});

// The third video piece. probe_video() reports decode / detect / encode apart
// on purpose ("a single boolean would be a lie here"), and ffmpeg fails on its
// own for a reason the backend documents: imageio-ffmpeg answers with a path
// whether or not its binary download finished, so `av` can import on a machine
// with no encoder at all. That machine scanned, detected and triaged fine — and
// the summary certified it complete while it could not cut or export one clip.
test('the capability summary counts clip encoding apart from decoding', () => {
  const rows = deriveCapabilitySummary({ video_decode: true, video_detect: true, video_encode: false });
  const labels = rows.map((r) => r.label);
  const encode = rows.find((r) => /video/i.test(r.label) && /encod/i.test(r.label));
  assert.ok(encode, `no clip-encoding row in the summary: ${labels.join(', ')}`);
  // It reads its OWN key — a row copy-pasted from "reading files" would be green here.
  assert.equal(encode.ok, false);
  assert.equal(rows.find((r) => /read/i.test(r.label) && /video/i.test(r.label)).ok, true);
  assert.equal(encode.topic, 'setup-quality');
  const on = deriveCapabilitySummary({ video_decode: true, video_detect: true, video_encode: true });
  assert.equal(on.find((r) => /video/i.test(r.label) && /encod/i.test(r.label)).ok, true);
});

// The `video` install action installs BOTH halves, so its catalog row cannot be
// "present" on decoding alone: that badged ✓ Installed on a machine with no
// encoder AND dropped the row from the install plans, hiding the one button
// that fixes it.
test('the install catalog treats the video action as missing when the encoder is', () => {
  const rows = installCatalog({ video_decode: true, video_detect: true, video_encode: false });
  const video = rows.find((r) => r.action === 'video');
  assert.ok(video, 'no "video" row in installCatalog');
  assert.equal(video.present, false);
  const whole = installCatalog({ video_decode: true, video_detect: true, video_encode: true });
  assert.equal(whole.find((r) => r.action === 'video').present, true);
});

// --- Every summary row must LEAD somewhere ---------------------------------
// "Krea 2 Edit (local)" rendered a ✗ and did nothing when clicked: it had no
// CAPABILITY_STEP_ID entry, so the row taught the user something was missing and
// offered no way to reach it — half the requirement. This test is the general
// form of that bug: any row added to deriveCapabilitySummary without a mapping,
// or mapped at a screen the wizard does not have, fails here.
const capabilityStepMap = () => {
  const page = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  const block = page.match(/const CAPABILITY_STEP_ID = \{([\s\S]*?)\n\}/);
  assert.ok(block, 'CAPABILITY_STEP_ID not found in SetupPage.jsx');
  const map = {};
  for (const m of block[1].matchAll(/'([^']+)':\s*'([^']+)',/g)) map[m[1]] = m[2];
  return { page, map };
};

test('every "What\'s unlocked" row maps to a wizard screen that exists', () => {
  const { page, map } = capabilityStepMap();
  // welcome is index 0 and screenOf() falls back to it — a row may never target it.
  const screensLine = page.match(/const SCREENS = \[([^\]]*)\]/);
  assert.ok(screensLine, 'SCREENS not found in SetupPage.jsx');
  const extraScreens = [...screensLine[1].matchAll(/'([^']+)'/g)]
    .map((m) => m[1]).filter((s) => s !== 'welcome');
  const valid = new Set([...SETUP_STEP_IDS, ...extraScreens]);
  const rows = deriveCapabilitySummary({});
  assert.ok(rows.length > 10, 'the summary is suspiciously short — has it stopped deriving?');
  for (const r of rows) {
    assert.ok(map[r.label],
      `"${r.label}" has no CAPABILITY_STEP_ID entry — its row renders inert`);
    assert.ok(valid.has(map[r.label]),
      `"${r.label}" points at "${map[r.label]}", which is not a wizard screen`);
  }
});

test('Krea 2 Edit points at the install screen, and screenOf can resolve it', () => {
  const { page, map } = capabilityStepMap();
  // Its ONE-CLICK installer is KreaInstallCard, mounted only inside
  // InstallEverything — i.e. on the install/repair screen. The comfyui step
  // carries Klein's weights, never Krea's, so mapping it there would land the
  // user on a screen with nothing to press.
  assert.equal(map['Krea 2 Edit (local)'], 'install');
  const installScreen = fs.readFileSync(
    new URL('../components/setup/InstallEverything.jsx', import.meta.url), 'utf8');
  assert.match(installScreen, /<KreaInstallCard/,
    'the install screen no longer mounts KreaInstallCard — the mapping now leads nowhere');
  assert.match(page, /<InstallEverything/,
    'SetupPage no longer renders InstallEverything on the install screen');
  // 'install' is not a tool step, so screenOf must fall back to SCREENS. Without
  // that lookup it resolved to indexOf(-1)+1 = 0 and dumped the user on welcome.
  const fn = page.match(/const screenOf = \(id\) => \{[\s\S]*?\n  \}/);
  assert.ok(fn, 'screenOf is no longer a block — re-check it still resolves non-step screens');
  assert.match(fn[0], /SCREENS\.indexOf\(id\)/,
    'screenOf ignores SCREENS: a row pointing at the install menu lands on welcome');
});

test('each new capability row maps to the quality wizard step in SetupPage', () => {
  const page = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  for (const label of [
    'Bank scoring (aesthetic · NSFW · style)',
    'SigLIP2 Bank semantics (optional)',
    'Watermark detector (optional)',
    'Scraping extras (optional)',
  ]) {
    const esc = label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp(`'${esc}':\\s*'quality'`);
    assert.match(page, re, `"${label}" is not wired to the quality step in CAPABILITY_STEP_ID`);
  }
});

// The scraping-extras install action now has a Setup card of its own — it used
// to be installable only from the Datasets ▸ Concept Sources panel, which the
// Setup wizard never pointed at (and could not: capabilityDestination() only
// resolves /settings and /setup routes). Without this card the new row's
// "manage in Setup wizard" promise would be a dead end.
test('the quality step offers a Setup card for scraping extras', () => {
  const cards = fs.readFileSync(new URL('../components/setup/mlInstallCards.js', import.meta.url), 'utf8');
  assert.match(cards, /action:\s*'scrape_extras'/);
  assert.match(cards, /cap:\s*'scrape_deps'/);
});

test('the install catalog offers the video extras for install and repair', () => {
  const rows = installCatalog({ video_decode: false, video_detect: true });
  const video = rows.find((r) => r.action === 'video');
  const shot = rows.find((r) => r.action === 'shot_detect');
  assert.ok(video, 'no "video" row in installCatalog');
  assert.ok(shot, 'no "shot_detect" row in installCatalog');
  assert.equal(video.present, false);
  assert.equal(shot.present, true);
  assert.ok(video.available && shot.available);
  assert.notEqual(video.label, 'video', 'the row shows a raw action id instead of a label');
  assert.notEqual(shot.label, 'shot_detect', 'the row shows a raw action id instead of a label');
});

// ── ChatGPT: two lanes, one engine ───────────────────────────────────────────
// Setup used to expose only the paid API key, so a Plus/Pro subscriber met
// "OpenAI API key — ○ Not set" and concluded the engine costs money. These lock
// the two facts that fixes it: either lane makes the engine ready, and the key
// lane's own state is read from the SECRETS payload (engines.chatgpt is the
// backend's OR of the two and cannot answer "is a key also set").
import { chatgptLanes, chatgptLaneSummary, CHATGPT_API_KEY_SECRET } from './useSetupSteps.js';

test('a connected subscription makes ChatGPT ready with no API key', () => {
  const l = chatgptLanes(
    { engines: { chatgpt: true }, chatgpt_subscription: { connected: true, email: 'a@b.c' } },
    {});
  assert.equal(l.ready, true);
  assert.equal(l.connected, true);
  assert.equal(l.keySet, false, 'a subscription must not be reported as an API key');
  assert.equal(l.both, false);
  assert.match(chatgptLaneSummary(l), /subscription/i);
  assert.doesNotMatch(chatgptLaneSummary(l), /Set up either one/);
});

test('an API key alone still makes ChatGPT ready', () => {
  const l = chatgptLanes({ engines: { chatgpt: true }, chatgpt_subscription: { connected: false } },
    { [CHATGPT_API_KEY_SECRET]: true });
  assert.equal(l.ready, true);
  assert.equal(l.keySet, true);
  assert.equal(l.connected, false);
  assert.match(chatgptLaneSummary(l), /API key/i);
});

test('neither lane leaves ChatGPT not-ready, and the copy names BOTH ways in', () => {
  const l = chatgptLanes({ engines: { chatgpt: false }, chatgpt_subscription: { connected: false } },
    {});
  assert.equal(l.ready, false);
  const s = chatgptLaneSummary(l);
  assert.match(s, /API key/i);
  assert.match(s, /subscription/i);
});

test('both lanes set up is reported as both, not as a conflict', () => {
  const l = chatgptLanes({ engines: { chatgpt: true }, chatgpt_subscription: { connected: true } },
    { [CHATGPT_API_KEY_SECRET]: true });
  assert.equal(l.both, true);
  assert.match(chatgptLaneSummary(l), /Both lanes/);
});

// A payload from before engines.chatgpt existed must not hide a lane the user can
// SEE is in place — a fresh install polls caps before its first save.
test('a legacy payload without engines still reports a connected subscription ready', () => {
  assert.equal(chatgptLanes({ chatgpt_subscription: { connected: true } }, {}).ready, true);
  assert.equal(chatgptLanes({}, { [CHATGPT_API_KEY_SECRET]: true }).ready, true);
  assert.equal(chatgptLanes(null, null).ready, false);
});

// The wizard must OFFER the subscription, not merely tolerate it: the whole
// defect was a screen that only ever showed the key field.
test('the Setup image step mounts the subscription connect flow', () => {
  const page = fs.readFileSync(new URL('../pages/SetupPage.jsx', import.meta.url), 'utf8');
  assert.match(page, /<ChatgptSubscriptionConnect/);
  assert.doesNotMatch(page, /help: 'Powers ChatGPT \(gpt-image-2\)\.'/,
    'the key field still claims to be what powers the engine');
});

// A wizard that greets a first-time user with a red "✗ Network error" under an
// engine he has not touched reads as a broken app. Errors belong to the action
// that produced them; a state fetch failing on its own must stay off the screen.
import { connectFeedback } from '../components/common/chatgptConnectFeedback.js';

test('the subscription row shows an error only when a user action produced it', () => {
  assert.equal(connectFeedback(null), null);
  assert.equal(connectFeedback({ message: 'Network error' }), null,
    'an error with no action behind it reaches the screen');
  assert.equal(connectFeedback({ action: 'connect' }), null);
  assert.equal(connectFeedback({ action: 'connect', message: 'Network error' }), 'Network error');
});

// One device-code implementation, mounted by both screens. Two copies of an OAuth
// polling loop is how "connected" starts meaning different things per page.
test('Settings and Setup share one subscription component', () => {
  const engines = fs.readFileSync(
    new URL('../components/settings/EnginesSection.jsx', import.meta.url), 'utf8');
  assert.match(engines, /<ChatgptSubscriptionConnect/);
  assert.doesNotMatch(engines, /chatgpt-oauth\/start/,
    'EnginesSection re-implements the device-code login instead of sharing it');
});

// --- Ollama: an optional tool the wizard stopped treating as a prerequisite ----
//
// The step used to refuse Next outright when Ollama was absent, and a NATIVE install
// had no per-step way out: the "No Ollama" card only renders for Docker deployments
// (ollamaStep leaves deploymentMode at 'local'). These lock the two lifts and the
// self-annulling skip, so neither can quietly come back.

const ollamaStepOf = (caps) => deriveSetupSteps(caps).find((s) => s.id === 'ollama');

test('a chosen "continue without Ollama" reads as a neutral skip, not a failure', () => {
  const s = ollamaStepOf({ ollama: { reachable: false, skipped: true } });
  assert.equal(s.skipped, true);
  assert.equal(s.status, 'skipped');
  assert.equal(ollamaGateReason(s), null, 'a skipped step must not hold the wizard');
});

test('the skip never survives a reachable Ollama — the real state wins', () => {
  // The backend derives `skipped` as (flag AND not reachable), so a running Ollama
  // arrives with skipped:false and its own gap on show. Nothing to undo by hand.
  const s = ollamaStepOf({ ollama: { reachable: true, vision_model_ready: false, skipped: false } });
  assert.equal(s.skipped, false);
  assert.notEqual(s.status, 'skipped');
});

test('a ready JoyCaption lifts the Ollama gate', () => {
  const caps = { ollama: { reachable: false, installed: false } };
  const blocked = ollamaStepOf(caps);
  assert.equal(blocked.joycaptionReady, false);
  assert.match(ollamaGateReason(blocked), /isn't installed/);

  const withJoy = ollamaStepOf({ ...caps, captioners: { joycaption: true } });
  assert.equal(withJoy.joycaptionReady, true);
  assert.equal(ollamaGateReason(withJoy), null,
    'JoyCaption writes the same prompt (prose for Z-Image, booru for SDXL) — it is not an SDXL-only fallback');
});

test('the gate still holds on states the user can act on right here', () => {
  // Lifting the gate for JoyCaption must not lift it for a Docker deployment that
  // has not been chosen yet, or a container mid-start: those are answerable on the page.
  const unchosen = { status: 'available', unconfigured: true };
  assert.match(ollamaGateReason(unchosen), /Choose No Ollama/);
  const starting = { status: 'initializing', managedInitializing: true };
  assert.match(ollamaGateReason(starting), /still starting/);
  // And a Docker deployment explicitly set to "None" was never a block.
  assert.equal(ollamaGateReason({ status: 'skipped', disabled: true }), null);
});

test('what continuing without Ollama costs is sourced, and captioning is not on that list', () => {
  const lost = OLLAMA_SKIP_LOST.join(' | ');
  const kept = OLLAMA_SKIP_KEPT.join(' | ');
  // The Ollama-only passes — each one a real gate in the app.
  assert.match(lost, /framing/i);
  assert.match(lost, /head-crop/i);
  assert.match(lost, /Describe/);
  assert.match(lost, /Short captions/i);
  // Captioning itself survives, and the panel says by what.
  assert.match(kept, /JoyCaption/);
  assert.ok(!OLLAMA_SKIP_LOST.some((t) => /^Captioning\b/i.test(t)),
    'captioning is not lost with Ollama — JoyCaption covers it');
});

test('nothing the wizard SAYS claims JoyCaption is SDXL-only', () => {
  // The premise that justified the hard block, asserted where it counts: the
  // sentences a user can actually read. Every branch of the gate, plus both lists.
  const everySentence = [
    ollamaGateReason({ status: 'available', unconfigured: true }),
    ollamaGateReason({ status: 'initializing', managedInitializing: true }),
    ollamaGateReason({ status: 'available', reachable: false, deploymentMode: 'host' }),
    ollamaGateReason({ status: 'available', reachable: false, installed: false }),
    ollamaGateReason({ status: 'available', reachable: false, installed: true }),
    ollamaGateReason({ status: 'partial', reachable: true, visionModelReady: false }),
    ollamaGateReason({ status: 'partial', reachable: true, visionModelReady: true }),
    ...OLLAMA_SKIP_LOST, ...OLLAMA_SKIP_KEPT,
  ].filter(Boolean).join(' | ');
  assert.ok(everySentence.length > 0);
  assert.doesNotMatch(everySentence, /only covers SDXL/i);
  assert.doesNotMatch(everySentence, /JoyCaption only/i);
});

test('the KEPT column never ticks a captioner this machine does not have', () => {
  const withJoy = ollamaSkipKept(true);
  assert.deepEqual(withJoy, OLLAMA_SKIP_KEPT);
  assert.match(withJoy.join(' | '), /JoyCaption/);

  const without = ollamaSkipKept(false);
  assert.doesNotMatch(without.join(' | '), /JoyCaption/,
    'promising JoyCaption captioning on an install that has none is the half-truth this avoids');
  // Everything true of ANY install stays — the column must not go empty.
  assert.equal(without.length, OLLAMA_SKIP_KEPT.length - 1);
  assert.match(without.join(' | '), /LoRA training/);
});

test('a Docker deployment is never read as a native skip, whatever the stored flag says', () => {
  // The `!dockerManaged` half of `skipped` is what keeps a stored flag from swallowing
  // a Docker diagnosis — 'host' unreachable names a precise remedy (open 11434 to
  // Docker) that a neutral "you chose to skip" would hide. Every other test here calls
  // deriveSetupSteps with ONE argument, which leaves deploymentMode at 'local' and
  // makes that half constant, so this is the only place it is actually exercised.
  const s = deriveSetupSteps(
    { ollama: { reachable: false, skipped: true } },
    { ollama: { mode: 'host', state: 'unreachable', ready: false } },
  ).find((x) => x.id === 'ollama');
  assert.equal(s.dockerManaged, true);
  assert.equal(s.skipped, false);
  assert.match(ollamaGateReason(s), /Host Ollama is selected/);
});

test('an unchosen Docker deployment is still a question, even with JoyCaption ready', () => {
  // The JoyCaption lift answers "can this install caption?". It must not answer
  // "which Ollama deployment do you want?" — nothing starts until a card is picked,
  // and "No Ollama" is one of the cards.
  const s = deriveSetupSteps(
    { ollama: { reachable: false }, captioners: { joycaption: true } },
    { ollama: { mode: 'unconfigured', ready: false } },
  ).find((x) => x.id === 'ollama');
  assert.equal(s.joycaptionReady, true);
  assert.equal(s.unconfigured, true);
  assert.match(ollamaGateReason(s), /Choose No Ollama/);
});

test('the installer stops offering an Ollama pull to someone running LM Studio', () => {
  // The trap is that it looks available: a machine that switched to LM Studio very
  // often still has Ollama answering, so `reachable` is true and the row would
  // offer several GB of a model this install will never call — on the screen where
  // people click everything to finish Setup.
  const caps = (provider) => ({
    local_llm: { provider },
    ollama: { reachable: true, vision_model_ready: false, vision_model: 'qwen3-vl:8b' },
  });
  const row = (provider) =>
    installCatalog(caps(provider)).find((c) => c.action === 'ollama_model');

  assert.equal(row('ollama').available, true);
  const lms = row('lmstudio');
  assert.equal(lms.available, false);
  assert.match(lms.hint, /LM Studio app/,
    'a row turned off without a reason is a dead end, which is what this menu exists to close');
});

test('an install that predates the provider setting still gets the Ollama pull', () => {
  const row = installCatalog({
    ollama: { reachable: true, vision_model_ready: false, vision_model: 'qwen3-vl:8b' },
  }).find((c) => c.action === 'ollama_model');
  assert.equal(row.available, true);
});

/* 📷 Camera angles — the one-click install, front-end side.

   The gap this pins shut is the one the Krea rows already document: an engine
   whose weights install through the 409 but appear NOWHERE on the screen where
   the user decides they are done. Every surface below existed for Klein, Krea
   and SeedVR2 and not for this lane until now — the plan, the catalog rows,
   the labels, and the counted capability. */
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CAMERA_INSTALL_ORDER, INSTALL_ALL_ACTION_LABELS, cameraInstallPlan,
  deriveCapabilitySummary, installAllPlan, installCatalog,
} from '../hooks/useSetupSteps.js';

const caps = (comfyui) => ({ comfyui });

test('the camera button queues only the missing weights, in the backend order', () => {
  assert.deepEqual(
    cameraInstallPlan(caps({ dir_valid: true,
      camera_missing: ['camera_lora', 'krea_vae'] })),
    ['camera_lora', 'krea_vae'],
  );
  // Someone who placed everything by hand sees an empty plan, not a 20 GB
  // re-download.
  assert.deepEqual(
    cameraInstallPlan(caps({ dir_valid: true, camera_missing: [] })), []);
  // No validated ComfyUI folder → no plan, never a guessed path.
  assert.deepEqual(
    cameraInstallPlan(caps({ dir_valid: false, camera_missing: ['camera_model'] })), []);
});

test('the VAE rides the Krea action — one file, one button', () => {
  // camera_missing reports the Qwen VAE under `krea_vae` (the lane shares the
  // file), and the plan passes that through rather than inventing a camera_vae.
  assert.ok(CAMERA_INSTALL_ORDER.includes('krea_vae'));
  assert.ok(!CAMERA_INSTALL_ORDER.includes('camera_vae'));
});

test('every queued action has a human label', () => {
  for (const a of CAMERA_INSTALL_ORDER) {
    assert.ok(INSTALL_ALL_ACTION_LABELS[a], `no label for ${a}`);
  }
});

test('"Install everything" never silently pulls the camera engine', () => {
  // ~21.6 GB for a verb nobody pressed would be hostile on a metered link —
  // same rule as Krea and SeedVR2.
  const plan = installAllPlan({ comfyui: {
    dir_valid: true,
    klein_missing: [], camera_missing: ['camera_model', 'camera_lora'],
  } });
  assert.ok(!plan.some((a) => a.startsWith('camera_')), plan.join(','));
});

test('the install menu lists every camera weight with its live state', () => {
  const rows = installCatalog(caps({
    dir_valid: true, camera_missing: ['camera_lora'], klein_missing: [], krea_missing: [],
  }));
  const byAction = Object.fromEntries(rows.map((r) => [r.action, r]));
  for (const a of ['camera_model', 'camera_lora', 'camera_speed_lora', 'camera_text_encoder']) {
    assert.ok(byAction[a], `no install row for ${a}`);
    assert.ok(byAction[a].available, `${a} not installable with a valid dir`);
  }
  assert.equal(byAction.camera_lora.present, false, 'a missing weight must read ✗');
  assert.equal(byAction.camera_model.present, true);
  // And exactly ONE row owns the shared VAE.
  assert.equal(rows.filter((r) => r.action === 'krea_vae').length, 1);
});

test('camera angles is a counted capability, never dropped from the total', () => {
  // The certification bug, third verse: "N of N ready" on a machine missing a
  // whole lane is worse than a red row.
  const summary = deriveCapabilitySummary({
    comfyui: { dir_valid: true, camera_ready: false, camera_missing: ['camera_model'] },
  });
  const row = summary.find((r) => r.label.includes('Camera angles'));
  assert.ok(row, 'camera angles missing from the readiness list');
  assert.equal(row.ok, false);
  const ready = deriveCapabilitySummary({
    comfyui: { dir_valid: true, camera_ready: true, camera_missing: [] },
  }).find((r) => r.label.includes('Camera angles'));
  assert.equal(ready.ok, true);
});

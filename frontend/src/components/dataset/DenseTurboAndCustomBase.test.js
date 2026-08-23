// Picking Krea 2 Turbo (or a local checkpoint) must not throw the user out of
// Full model, and the full-model recipe must offer somewhere to pick them.
//
// The reported symptom was "I still can't see where to put the turbo option",
// and it had TWO causes, so a test on either one alone would have gone green
// over a broken screen:
//   1. `isFullTransformerEligible` refused anything but official Raw, and an
//      effect in the panel then SAVED LoRA behind the user's back;
//   2. the family / variant / base controls are rendered inside the LoRA-only
//      branch, so a dense recipe had no control to change at all.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  denseTurboWarning,
  fullTransformerBaseLabel,
  isFullTransformerEligible,
} from '../../utils/trainingMode.js';

// Slice 1 moved the dense recipe/picker and the cloud dialog to their own
// files; this flow spans all three, so the contract reads them as one text.
const panel = readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8')
  + readFileSync(new URL('./FullTransformerRecipe.jsx', import.meta.url), 'utf8')
  + readFileSync(new URL('./CloudLaunchDialog.jsx', import.meta.url), 'utf8');

const picker = panel.slice(
  panel.indexOf('DENSE_BASE_PICKER_START'),
  panel.indexOf('DENSE_BASE_PICKER_END'),
);

test('the automatic fallback to LoRA no longer fires for a Turbo or custom-base Krea recipe', () => {
  // Both fallbacks are gated on the REAL predicate, so evaluating it is
  // evaluating the guard — not a paraphrase of it.
  assert.match(panel, /if \(!baseInfo \|\| !fullMode \|\| fullTransformerEligible\) \{/);
  assert.match(panel, /&& !isFullTransformerEligible\(nextSelection\)\) \{/);

  const turbo = { trainType: 'krea', variant: 'turbo', baseModel: '', customBase: false };
  const custom = {
    trainType: 'krea', variant: 'base',
    baseModel: 'C:/models/my-krea.safetensors', customBase: true,
  };
  // fullTransformerEligible true => the effect returns before saving LoRA.
  assert.equal(isFullTransformerEligible(turbo), true);
  assert.equal(isFullTransformerEligible(custom), true);
  // Leaving the family is still a real recipe transition and still falls back.
  assert.equal(isFullTransformerEligible({ ...turbo, trainType: 'zimage' }), false);
});

test('the full-model recipe renders its own base and variant controls', () => {
  // The MARKUP of those controls is asserted by mounting them for real, in
  // tests/dense-base-picker-render.test.mjs — a source-text match cannot tell
  // a rendered control from one that throws. What belongs here is the wiring:
  // the dense arm, and only the dense arm, reaches that component.
  assert.ok(picker.length > 0, 'the dense base picker block must exist');
  assert.match(picker, /<DenseBasePicker/);
  assert.match(picker, /baseSummary=\{denseBaseSummary\}/);
  assert.match(picker, /busy=\{trainingModeBusy\}/);
  const denseArm = panel.slice(
    panel.indexOf('FULL_TRANSFORMER_ADVANCED_BRANCH_START'),
    panel.indexOf('LORA_ADVANCED_CONTROLS_START'));
  assert.ok(denseArm.includes('DENSE_BASE_PICKER_START'));
});

test('a picked checkpoint IS the base, so its label ignores the variant', () => {
  // Backend truth: `_krea_name_or_path` returns the custom path whatever the
  // variant says. The picker greys the switch out for the same reason, which
  // the render test asserts on the produced markup.
  assert.equal(
    fullTransformerBaseLabel({ variant: 'turbo', baseModel: 'C:/m/a.safetensors' }),
    'custom: a.safetensors');
  assert.equal(fullTransformerBaseLabel({ variant: 'turbo' }), 'official Krea 2 Turbo');
});

test('every full-model summary is computed from the selection, never a Raw literal', () => {
  assert.doesNotMatch(panel, /Official Krea 2 Raw · full transformer/);
  assert.doesNotMatch(panel, /Full model · official Krea 2 Raw · cloud/);
  assert.match(panel, /Full model · \{denseBaseSummary\} · cloud/);
  assert.match(panel, /\{denseBaseSummary\} · cloud\s*<\/span>/);
  assert.equal(fullTransformerBaseLabel({ variant: 'turbo' }), 'official Krea 2 Turbo');
});

test('the untested-Turbo notice is shown before the spend, twice, and never blocks', () => {
  // Once in the panel, above the launch buttons…
  assert.match(panel, /id="dense-turbo-warning"/);
  assert.match(panel, /\{fullMode && denseTurboNotice && \(/);
  // …and once inside the GPU-rental dialog, the last screen before the money.
  assert.match(panel, /const turboNotice = fullMode \? denseTurboWarning\(\{ baseModel: base, variant \}\) : null;/);
  // It is a status, not an alert, and nothing about it gates a button.
  assert.doesNotMatch(panel, /disabled=\{[^}]*turboNotice/);
  assert.doesNotMatch(panel, /disabled=\{[^}]*denseTurboNotice/);
  const notice = denseTurboWarning({ variant: 'turbo' });
  assert.match(notice.body, /have not measured/);
});

test('the custom-base push section runs for full-model runs too', () => {
  // Without this the lifted refusal would have had no transport: the private
  // repo the pod downloads from is created by that section.
  assert.match(panel, /const isCustomBase = !!String\(base \|\| ''\)\.trim\(\);/);
  assert.doesNotMatch(panel, /const isCustomBase = !fullMode && /);
});

// Cloud custom-base push flow (source contract, same style as
// TrainingContinueGuards.test.js): the launch dialog must gate "Rent & train"
// on the private-repo readiness, offer the one-time push, and the panel must
// no longer flat-block the cloud button for custom weights on the three
// supported families.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const panel = readFileSync(new URL('./TrainingPanel.jsx', import.meta.url), 'utf8');

test('cloud dialog embeds the custom-base push gate and blocks launch until ready', () => {
  assert.match(panel, /function CustomBasePushSection\(/);
  assert.match(panel, /train\/cloud\/custom-base\?/);           // readiness poll
  assert.match(panel, /train\/cloud\/custom-base\/push/);       // one-time push POST
  assert.match(panel, /disabled=\{!selected \|\| launching \|\| !customBaseReady\}/);
});

test('push copy states PRIVATE repo + one-time + reused, in English', () => {
  assert.match(panel, /PRIVATE<\/b> repo/);
  assert.match(panel, /future cloud runs reuse it/);
  assert.match(panel, /never made public/);
});

test('custom weights no longer flat-block cloud for supported families', () => {
  // the old blanket refusal string must be gone from the disabled-reason chain…
  assert.ok(!panel.includes(
    "? 'Custom weights are local-only — cloud training uses the official Hugging Face bases'"));
  // …while VAE/TE overrides (SDXL-only, genuinely local-only) still block,
  // and an unconverted Z-Image custom base still asks for conversion first.
  assert.match(panel, /Custom VAE\/text-encoder overrides are local-only/);
  assert.match(panel, /Convert the custom base first — the cloud lane pushes the converted copy/);
});

test('arch sniff refusal stays confirmable from the push flow', () => {
  assert.match(panel, /CUSTOM_WEIGHTS_UNVERIFIED: /);
  assert.match(panel, /workspace\.training\.dialogs\.pushForce/);
});

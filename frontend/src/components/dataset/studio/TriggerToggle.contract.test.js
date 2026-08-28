// Case « Trigger word » — the checkbox that sends the test prompt as written.
// Source-reading contract: every launching surface must OFFER the box and
// CARRY the choice to its POST, under the one wire name the backend reads
// (`inject_trigger`, absent = the historical injected default), and the two
// panels must share ONE localStorage preference so the same user gets the
// same behaviour whichever screen launches.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const promptField = readFileSync(new URL('./PromptField.jsx', import.meta.url), 'utf8');
const panel = readFileSync(new URL('./RunSetupPanel.jsx', import.meta.url), 'utf8');
const runSetup = readFileSync(new URL('./StudioRunSetup.jsx', import.meta.url), 'utf8');
const comparison = readFileSync(new URL('./ComparisonStudio.jsx', import.meta.url), 'utf8');
const lightbox = readFileSync(new URL('./ResultLightbox.jsx', import.meta.url), 'utf8');

test('both prompt surfaces render the Trigger word checkbox', () => {
  for (const src of [promptField, runSetup]) {
    assert.match(src, /Trigger word/);
    assert.match(src, /onInjectTrigger/);
    assert.match(src, /checked=\{injectTrigger\}/);
  }
});

test('RunSetupPanel sends inject_trigger only when unticked (default body unchanged)', () => {
  assert.match(panel, /if \(!injectTrigger\) settings\.inject_trigger = false;/);
  assert.doesNotMatch(panel, /inject_trigger = true/);
});

test('ComparisonStudio sends inject_trigger only when unticked', () => {
  assert.match(comparison, /if \(!injectTrigger\) body\.inject_trigger = false;/);
});

test('the two panels share one persisted preference key', () => {
  const key = /studioInjectTrigger/;
  assert.match(panel, key);
  assert.match(comparison, key);
  // Default-true read: absent key must mean "inject", exactly like before.
  for (const src of [panel, comparison]) {
    assert.match(src, /localStorage\.getItem\('studioInjectTrigger'\) !== '0'/);
  }
});

test('the lightbox meta says when a cell was generated without the trigger', () => {
  assert.match(lightbox, /inject_trigger === false \? ' · no trigger'/);
});

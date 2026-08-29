// Case « Trigger word » — the checkbox that sends the test prompt as written.
// Source-reading contract: every launching surface must OFFER the box and
// CARRY the choice to its POST, under the one wire name the backend reads
// (`inject_trigger`, absent = the historical injected default). Persistence
// lives in ONE pure module (triggerPref.js) shared by the launching surfaces —
// so the same user gets the same behaviour whichever screen launches, and
// RunSetupPanel itself keeps touching no storage (the prompt-batch contract
// pins that).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const promptField = readFileSync(new URL('./PromptField.jsx', import.meta.url), 'utf8');
const panel = readFileSync(new URL('./RunSetupPanel.jsx', import.meta.url), 'utf8');
const runSetup = readFileSync(new URL('./StudioRunSetup.jsx', import.meta.url), 'utf8');
const comparison = readFileSync(new URL('./ComparisonStudio.jsx', import.meta.url), 'utf8');
const lightbox = readFileSync(new URL('./ResultLightbox.jsx', import.meta.url), 'utf8');
const pref = readFileSync(new URL('./triggerPref.js', import.meta.url), 'utf8');
const canvasPanel = readFileSync(
  new URL('../../canvas/CanvasGenerationPanel.jsx', import.meta.url), 'utf8');
const canvasBlend = readFileSync(new URL('../../canvas/CanvasBlendPanel.jsx', import.meta.url), 'utf8');
const stackPanel = readFileSync(new URL('./LoraStackPanel.jsx', import.meta.url), 'utf8');
const stackComposition = readFileSync(new URL('./StackCompositionPanel.jsx', import.meta.url), 'utf8');
const facts = readFileSync(new URL('../../../utils/generatedImageFacts.js', import.meta.url), 'utf8');

test('both prompt surfaces render the Trigger word checkbox', () => {
  for (const src of [promptField, runSetup]) {
    assert.match(src, /Trigger word/);
    assert.match(src, /onInjectTrigger/);
    assert.match(src, /checked=\{injectTrigger\}/);
  }
});

test('RunSetupPanel derives the launch body — it NEVER writes into genSettings', () => {
  // launchSettings returns the state object BY IDENTITY when no batch prompt is
  // ticked (pinned by promptBatch.test.js). Writing inject_trigger into it made
  // the key sticky: one unticked launch and a re-ticked box kept sending false.
  assert.match(panel, /const settings = injectTrigger \? base : \{ \.\.\.base, inject_trigger: false \};/);
  assert.doesNotMatch(panel, /settings\.inject_trigger\s*=/);
  assert.doesNotMatch(panel, /base\.inject_trigger\s*=/);
});

test('ComparisonStudio sends inject_trigger only when unticked', () => {
  assert.match(comparison, /if \(!injectTrigger\) body\.inject_trigger = false;/);
});

test('one shared preference module, default-true, written only when unticked', () => {
  for (const src of [panel, comparison, canvasPanel]) {
    assert.match(src, /from '(\.\.\/dataset\/studio\/)?(\.\/)?triggerPref'/);
    assert.match(src, /useState\(readInjectTrigger\)/);
    assert.match(src, /writeInjectTrigger\(v\)/);
  }
  assert.match(pref, /const KEY = 'studioInjectTrigger';/);
  assert.match(pref, /getItem\(KEY\) !== '0'/);
  assert.match(pref, /removeItem\(KEY\)/);
});

test('the canvas lifts ONE state to both the blend panel and the run panel', () => {
  // The blend panel announces what will be prefixed to the prompt; the run
  // panel holds the box. They must read the same state or the blend promises
  // an injection the unticked box cancels.
  assert.match(canvasPanel, /<CanvasBlendPanel[\s\S]*?injectTrigger=\{injectTrigger\}/);
  assert.match(canvasPanel, /<RunSetupPanel[\s\S]*?injectTrigger=\{injectTrigger\}/);
  assert.match(canvasPanel, /onInjectTrigger=\{toggleInjectTrigger\}/);
  // And RunSetupPanel accepts the parent's control without dropping its own
  // fallback (the dataset Studio keeps mounting it uncontrolled).
  assert.match(panel, /injectTriggerProp \?\? ownInjectTrigger/);
});

test('the stack surfaces stop promising an injection the box cancels', () => {
  assert.match(canvasBlend, /injectTrigger = true/);
  assert.match(canvasBlend, /unticked — no trigger word is added/);
  assert.match(stackPanel, /injectTrigger = true/);
  assert.match(stackPanel, /unticked — no trigger words are injected/);
  // Compare page passes its live state to the launch-side stack panel…
  assert.match(comparison, /<LoraStackPanel[\s\S]*?injectTrigger=\{injectTrigger\}/);
  // …and the RUN-side composition panel reads the run's own cells, not the box.
  assert.match(comparison, /injectTrigger=\{!cells\.some\(\(c\) => c\.inject_trigger === false\)\}/);
  assert.match(stackComposition, /NOT injected/);
});

test('the lightbox meta and the gallery facts both say when a cell ran without the trigger', () => {
  assert.match(lightbox, /inject_trigger === false \? ' · no trigger'/);
  assert.match(facts, /inject_trigger === false/);
  assert.match(facts, /'Trigger word', 'not injected'/);
});

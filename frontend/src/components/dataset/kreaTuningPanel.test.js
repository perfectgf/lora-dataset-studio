/* Contract for the "🧬 Krea 2 Edit tuning" panel of Generate variations.

   node --test parses .js, not .jsx, so the behaviour lives in utils/kreaDials.js
   (unit-tested next to it) and this file pins the WIRING in the component source
   — the same source-text approach settingDefaults.test.js uses.

   What must not silently come back:
   - a dial disappearing from this panel again. All FOUR calibration keys are
     editable here AND in Settings › Image engines: they used to be split three
     ways (grounding editable in Settings only, steps in Settings only, the other
     two here only), so "where do I change this?" had a different answer per
     dial. Duplicating a control cannot fork the value — every control writes the
     SAME global key through the same endpoint, so there is nothing to sync;
   - a slider writing on every change event instead of through the debounced
     saver — a drag would then be forty PUTs to /api/settings;
   - a default typed into the JSX. These sliders offer "Reset to default", and a
     copied literal makes that button restore a value that stopped being the
     default the day it moved server-side;
   - the design comment still claiming the panel is a read-out "not a second set
     of sliders", which is what it said while these sliders sat below it;
   - the two FILE PATH fields (base_model, identity_lora) migrating here. They
     are install-time values, not things you judge against an image. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const SRC = readFileSync(new URL('./VariationCatalog.jsx', import.meta.url), 'utf8');
// The registry split its topic data into ./topics/* (2026-08-24) — the
// contract is about the topics, so it reads all the sources as one text.
const helpRegistry = ['../../help/helpRegistry.js', '../../help/topics/settingsSections.js',
  '../../help/topics/workspaceSections.js', '../../help/topics/pages.js',
  '../../help/topics/videoLane.js', '../../help/topics/settingsFields.js',
  '../../help/topics/actions.js']
  .map((p) => readFileSync(new URL(p, import.meta.url), 'utf8')).join('\n');
const ENGINES = readFileSync(
  new URL('../settings/EnginesSection.jsx', import.meta.url), 'utf8');

test('all four dials are rendered here, with the bounds the server clamps to', () => {
  assert.match(SRC, /id="krea-ref-boost-dial"/);
  assert.match(SRC, /id="krea-identity-lora-strength-dial"/);
  assert.match(SRC, /id="krea-grounding-dial"/);
  assert.match(SRC, /id="krea-steps-dial"/);
  // Bounds come from the shared module, never retyped here.
  assert.match(SRC, /min=\{KREA_REF_BOOST_MIN\}[\s\S]{0,80}max=\{KREA_REF_BOOST_MAX\}/);
  assert.match(SRC,
    /min=\{KREA_IDENTITY_STRENGTH_MIN\}[\s\S]{0,110}max=\{KREA_IDENTITY_STRENGTH_MAX\}/);
  assert.match(SRC, /min=\{KREA_GROUNDING_MIN\}[\s\S]{0,80}max=\{KREA_GROUNDING_MAX\}/);
  assert.match(SRC, /min=\{KREA_STEPS_MIN\}[\s\S]{0,80}max=\{KREA_STEPS_MAX\}/);
});

test('Settings offers the same four, so neither screen is the poor relation', () => {
  assert.match(ENGINES, /id="krea-grounding"/);
  assert.match(ENGINES, /id="krea-steps"/);
  assert.match(ENGINES, /id="krea-ref-boost"/);
  assert.match(ENGINES, /id="krea-identity-lora-strength"/);
  // ...and every one of them can go back to the shipped value from there.
  for (const field of ['grounding_px', 'steps', 'ref_boost', 'identity_lora_strength']) {
    assert.match(ENGINES, new RegExp(`field="${field}"`),
      `no ResetToDefault for krea.${field} in Settings`);
  }
  // The bounds are IMPORTED, not a second copy of the server's clamps.
  assert.match(ENGINES, /from '\.\.\/\.\.\/utils\/kreaDials\.js'/);
});

test('the file-path fields stay in Settings alone', () => {
  // They are filled once at install; a path picker has no business on the screen
  // where you judge a rendered image.
  assert.doesNotMatch(SRC, /'krea',\s*'base_model'/);
  assert.doesNotMatch(SRC, /'krea',\s*'identity_lora'/);
  assert.match(ENGINES, /id="krea-base-model"/);
  assert.match(ENGINES, /id="krea-identity-lora"/);
});

test('every slider change goes through the debounced saver', () => {
  // All four onChange handlers call setKreaDial, which schedules; the panel must
  // own no putJson of its own for these keys.
  assert.match(SRC, /setKreaDial\('ref_boost',/);
  assert.match(SRC, /setKreaDial\('identity_lora_strength',/);
  assert.match(SRC, /setKreaDial\('grounding_px',/);
  assert.match(SRC, /setKreaDial\('steps',/);
  assert.match(SRC, /kreaDialSaver\.current\.schedule\(field, value\)/);
  assert.match(SRC, /createDialSaver\(/);
  // ...and it is flushed on unmount, so leaving mid-drag still saves.
  assert.match(SRC, /kreaDialSaver\.current\?\.flush\(\)/);
});

test('the defaults are read from the server payload, never typed here', () => {
  for (const field of ['grounding_px', 'steps', 'ref_boost', 'identity_lora_strength']) {
    assert.match(SRC, new RegExp(`defaultValueAt\\(configDefaults, 'krea', '${field}'\\)`));
  }
  assert.match(SRC, /setConfigDefaults\(d\.config_defaults \|\| \{\}\)/);
  // The literal shipped values must not appear as a useState seed or a `?? 0.25`.
  assert.doesNotMatch(SRC, /setKreaRefBoost\]\s*=\s*useState\(\s*0?\.\d/);
  assert.doesNotMatch(SRC, /setKreaIdentityStrength\]\s*=\s*useState\(\s*[\d.]/);
  assert.doesNotMatch(SRC, /setKreaGrounding\]\s*=\s*useState\(\s*[\d.]/);
  assert.doesNotMatch(SRC, /setKreaSteps\]\s*=\s*useState\(\s*[\d.]/);
});

test('the panel admits the sliders change EVERY future run', () => {
  // A control that quietly rewrites a global setting from a per-batch screen is
  // only acceptable while it says so. Keep the warning, whatever its wording —
  // but it must not say "two" now that there are four.
  const panel = SRC.slice(SRC.indexOf('🧬 Krea 2 Edit tuning'));
  assert.match(panel, /apply to <b>every<\/b> Krea\s*\n?\s*run/);
  assert.doesNotMatch(panel, /These two save straight/);
});

test('the design comment no longer contradicts the code', () => {
  assert.doesNotMatch(SRC, /not a second set of sliders/);
  // The old justification for keeping grounding a read-out ("duplicating that
  // explanation here would be the second truth") is gone with the read-out.
  assert.doesNotMatch(SRC, /grounding_px` stays a read-out/);
  assert.doesNotMatch(SRC, /Change it in\{' '\}/);
  // The rule that DID survive is still written down: no per-run copy.
  assert.match(SRC, /PER-RUN copy would be a second truth/);
});

test('the link to Settings survives — the paths and presets still live there', () => {
  assert.match(SRC, /focus="krea-engine"/);
});

test('each dial has a help topic, and the registry carries it', () => {
  // Built by concatenation on purpose: help-registry-contract.test.mjs scans
  // every .js under src/ for help-topic attributes, so writing one literally
  // HERE would make this test file look like an instrumented component to it.
  for (const key of ['ref_boost', 'identity_lora_strength', 'grounding_px', 'steps']) {
    const id = `krea.${key}`;
    assert.ok(SRC.includes(`topic=${JSON.stringify(id)}`), `${id} slider has no help topic`);
    assert.ok(helpRegistry.includes(`'${id}'`), `${id} is not in the help registry`);
  }
});

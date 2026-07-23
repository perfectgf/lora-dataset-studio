// Contract: the editable identity/Klein prompts live in ONE box that already
// holds the real built-in default (feature completion for @bbsorry / 雨田壹; the
// single-box rework asked for by the owner). node --test does not parse JSX, so
// these assert on source text — the same approach as DatasetLightbox.test.js.
// The behavioural guarantee (default text never persisted as a copy) is unit
// tested in ../common/promptOverride.test.js.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const engines = readFileSync(new URL('./EnginesSection.jsx', import.meta.url), 'utf8');
const settingsPage = readFileSync(new URL('../../pages/SettingsPage.jsx', import.meta.url), 'utf8');
const scraping = readFileSync(new URL('./ScrapingSection.jsx', import.meta.url), 'utf8');
const field = readFileSync(new URL('../common/PromptOverrideField.jsx', import.meta.url), 'utf8');
const modal = readFileSync(new URL('../dataset/IdentityPromptModal.jsx', import.meta.url), 'utf8');
const refPanel = readFileSync(new URL('../dataset/ReferencePanel.jsx', import.meta.url), 'utf8');

test('SettingsPage reads identity_prompt_defaults from the payload and threads it down', () => {
  assert.match(settingsPage, /setPromptDefaults\(data\.identity_prompt_defaults \|\| \{\}\)/);
  assert.match(settingsPage, /promptDefaults,/); // present in sectionProps
});

test('EnginesSection forwards promptDefaults to the identity prompts card', () => {
  assert.match(engines, /<IdentityPromptsCard[^>]*promptDefaults=\{props\.promptDefaults\}/);
});

test('the two-box layout is GONE — no read-only preview, no "load default" button', () => {
  // The old shape: an empty textarea + a "Built-in default (currently in use)"
  // block + "✎ Load default to edit". One box replaces all three.
  assert.doesNotMatch(engines, /DefaultPromptPreview/);
  assert.doesNotMatch(engines, /Load default to edit/);
  assert.doesNotMatch(engines, /Built-in default \(currently in use\)/);
});

test('every identity prompt is rendered by the shared single-box field', () => {
  // the three identity locks, from the SHARED metadata (no local copy)
  assert.match(engines, /import PromptOverrideField from '\.\.\/common\/PromptOverrideField'/);
  assert.match(engines, /IDENTITY_PROMPT_FIELDS\.map\(\(f\) => \(\s*<PromptOverrideField/);
  assert.match(engines, /defaultText=\{defaults\[f\.key\]\}/);
  // and the Klein-improve prompt (D), which keeps its on/off toggle
  assert.match(engines, /id="identity-prompt-klein-improve"/);
  assert.match(engines, /defaultText=\{defaults\.klein_improve\}/);
  assert.match(engines, /disabled=\{!improveEnabled\}/);
});

test('the single box shows the default and normalises a copy of it back to ""', () => {
  // value = override, else the real default text (never an empty box)
  assert.match(field, /value=\{promptBoxText\(value, defaultText\)\}/);
  // EVERY keystroke goes through the normaliser, so the config never holds a copy
  assert.match(field, /onChange\(normalizePromptOverride\(e\.target\.value, defaultText\)\)/);
  // the state is spelled out, both ways
  assert.match(field, /Following the built-in default/);
  assert.match(field, /Custom override/);
  // Reset clears back to '' (= follow the default), it does not re-type it
  assert.match(field, /onClick=\{\(\) => onChange\(''\)\}/);
  assert.match(field, /Reset to default/);
});

test('the Extra refs row opens the identity-prompt modal', () => {
  assert.match(refPanel, /import IdentityPromptModal from '\.\/IdentityPromptModal'/);
  assert.match(refPanel, /Edit the identity instruction used with multiple references/);
  assert.match(refPanel, /\{promptModal && <IdentityPromptModal onClose=/);
});

test('the modal shares the field and edits BOTH multi-reference prompts', () => {
  assert.match(modal, /import PromptOverrideField from '\.\.\/common\/PromptOverrideField'/);
  assert.match(modal, /EXTRA_REF_PROMPT_KEYS/);
  assert.match(modal, /activeExtraRefPromptKey/);
  // saves a PARTIAL config so a workspace save never rewrites other settings
  assert.match(modal, /putJson\('\/api\/settings', \{ config: \{ identity_prompts: patch \} \}\)/);
  assert.match(modal, /used by your current engine/);
});

test('the two Klein cards cross-reference each other to remove the ambiguity', () => {
  // engines card -> points at the scraping rescue card
  assert.match(engines, /Klein rescue — small scraped images/);
  // scraping card renamed + points at the manual identity prompts card
  assert.match(scraping, /title="Klein rescue — small scraped images"/);
  assert.match(scraping, /Small-image rescue instruction/);
  assert.match(scraping, /Identity &amp; Klein prompts/);
});

test('klein.small_image_prompt stays a genuinely optional EMPTY field', () => {
  // It is NOT part of the single-box migration: its config default is '' with no
  // shipped text behind it (backend reads klein.small_image_prompt, '') — empty
  // means "no instruction at all", not "use a built-in one". Pre-filling it would
  // invent a rescue prompt on the user's behalf.
  assert.doesNotMatch(scraping, /PromptOverrideField/);
  assert.match(scraping, /placeholder="Empty — reference image only"/);
});

// Contract: the editable identity/Klein prompts SHOW their real built-in default
// (feature completion for @bbsorry / 雨田壹). node --test does not parse JSX, so
// these assert on source text — the same approach as DatasetLightbox.test.js.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const engines = readFileSync(new URL('./EnginesSection.jsx', import.meta.url), 'utf8');
const settingsPage = readFileSync(new URL('../../pages/SettingsPage.jsx', import.meta.url), 'utf8');
const scraping = readFileSync(new URL('./ScrapingSection.jsx', import.meta.url), 'utf8');

test('SettingsPage reads identity_prompt_defaults from the payload and threads it down', () => {
  assert.match(settingsPage, /setPromptDefaults\(data\.identity_prompt_defaults \|\| \{\}\)/);
  assert.match(settingsPage, /promptDefaults,/); // present in sectionProps
});

test('EnginesSection forwards promptDefaults to the identity prompts card', () => {
  assert.match(engines, /<IdentityPromptsCard[^>]*promptDefaults=\{props\.promptDefaults\}/);
});

test('each identity field shows the real default as placeholder and a Load-default button', () => {
  // the real default text becomes the placeholder (not a generic "leave blank")
  assert.match(engines, /placeholder=\{defaultText \|\| /);
  // a preview block renders the default text with a copy-into-field button
  assert.match(engines, /function DefaultPromptPreview/);
  assert.match(engines, /Load default to edit/);
  // "Load default to edit" copies the default INTO the field (real override) —
  // now a visible button in the status row right under the textarea, not buried
  // in the preview block below the fold.
  assert.match(engines, /onClick=\{\(\) => onChange\(defaultText\)\}/);
  // shown only while the field is blank (blank still == use default)
  assert.match(engines, /\{blank && <DefaultPromptPreview text=\{defaultText\}/);
});

test('the Klein-improve field (D) also exposes its default + load button', () => {
  assert.match(engines, /placeholder=\{defaults\.klein_improve \|\| /);
  assert.match(engines, /improveEnabled && improveBlank && \(\s*<DefaultPromptPreview text=\{defaults\.klein_improve\}/);
});

test('the two Klein cards cross-reference each other to remove the ambiguity', () => {
  // engines card -> points at the scraping rescue card
  assert.match(engines, /Klein rescue — small scraped images/);
  // scraping card renamed + points at the manual identity prompts card
  assert.match(scraping, /title="Klein rescue — small scraped images"/);
  assert.match(scraping, /Small-image rescue instruction/);
  assert.match(scraping, /Identity &amp; Klein prompts/);
});

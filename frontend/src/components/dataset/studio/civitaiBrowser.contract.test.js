import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const modal = readFileSync(new URL('./CivitaiBrowserModal.jsx', import.meta.url), 'utf8');
const button = readFileSync(new URL('./CivitaiBrowserButton.jsx', import.meta.url), 'utf8');
const promptField = readFileSync(new URL('./PromptField.jsx', import.meta.url), 'utf8');
const comparisonSetup = readFileSync(new URL('./StudioRunSetup.jsx', import.meta.url), 'utf8');
const topics = readFileSync(new URL('../../../help/topics/pages.js', import.meta.url), 'utf8');

test('every prompt surface mounts the SAME Civitai browser button', () => {
  // Generation-surface parity is the standing rule: the dataset Test Studio and
  // the canvas share PromptField, the multi-LoRA comparison has its own prompt
  // block — a button added to one and not the other is exactly the kind of
  // silent divergence users report as a bug.
  assert.match(promptField, /<CivitaiBrowserButton prompt=\{value\} onPrompt=\{onChange\}\s*\n\s*picks=\{civitaiPicks\} onTogglePick=\{onToggleCivitaiPick\} \/>/);
  assert.match(comparisonSetup, /<CivitaiBrowserButton prompt=\{prompt\} onPrompt=\{onPrompt\} \/>/);
  assert.match(button, /CivitaiBrowserModal/);
});

test('📝 the browser feeds the prompt batch where a batch exists, and only there', () => {
  // A tick on a card is one more pass of the next run — the same batch the
  // saved-prompts cards feed, merged without doubles (promptBatch.mergeBatches)
  // — and the browser stays open, because the gesture is "collect several".
  assert.match(modal, /data-testid="civitai-batch-toggle"/);
  assert.match(modal, /role="checkbox" aria-checked=\{inBatch\}/);
  assert.match(modal, /onTogglePick\(c\.prompt\)/);
  assert.match(modal, /data-testid="civitai-batch-footer"/);
  // Without a handler nothing appears: the comparison surface has no batch at
  // all, so it shows no tick boxes rather than boxes that feed nothing.
  assert.match(modal, /const batchable = typeof onTogglePick === 'function'/);
  assert.match(button, /picks=\{picks\} onTogglePick=\{onTogglePick\}/);
  // The panel that owns the batch merges the two sources into ONE `prompts`.
  const panel = readFileSync(new URL('./RunSetupPanel.jsx', import.meta.url), 'utf8');
  assert.match(panel, /mergeBatches\(pickedPrompts, civitaiPicks\)/);
  assert.match(panel, /onToggleCivitaiPick=\{toggleCivitaiPick\}/);
  // …and says so under the field, where it is read even with an empty history.
  assert.match(promptField, /data-testid="civitai-batch-count"/);
});

test('a typed prompt is never silently replaced', () => {
  // Same overwrite rule as 🔎 Describe and 🎲 Caption: picking a Civitai prompt
  // over a non-empty field asks first.
  assert.match(button, /prompt && prompt\.trim\(\)\s*&& !window\.confirm/);
});

test('the browse call carries the filters and the exact continuation', () => {
  assert.match(modal, /\/api\/studio\/civitai\/images/);
  // Server-side accumulation answers with (next_cursor, next_skip) naming the
  // first listing item NOT yet consumed — both go back verbatim, or “Load
  // more” re-serves or skips items.
  assert.match(modal, /params\.set\('cursor', cont\.current\.cursor\)/);
  assert.match(modal, /params\.set\('skip', String\(cont\.current\.skip\)\)/);
  assert.match(modal, /require_prompt', '0'/);
  // A re-walked page may overlap what is shown: append-only dedup by id.
  assert.match(modal, /filter\(\(c\) => !seen\.has\(c\.id\)\)/);
  assert.match(modal, /exhausted/);
});

test('filter choices persist under stable localStorage keys', () => {
  // These keys hold choices made in people's browsers — renaming one resets it.
  for (const key of ['civitaiBrowse_period', 'civitaiBrowse_sort',
    'civitaiBrowse_level', 'civitaiBrowse_withPrompt']) {
    assert.match(modal, new RegExp(key));
  }
  // Mild default: the content-level ceiling starts at Safe, never at X.
  assert.match(modal, /'civitaiBrowse_level', 'none'/);
});

test('the no-key story points at the credential’s real home, not a dead end', () => {
  // Prompts are behind Civitai auth (measured: getGenerationData 401s without
  // a key; the v1 listing serves meta:null even authenticated). The banner must
  // route to the ONE place the app stores that credential.
  assert.match(modal, /hasKey === false/);
  assert.match(modal, /to="\/settings\/scraping"/);
  assert.match(modal, /keyRejected/);
});

test('external links leave the app without a referrer', () => {
  assert.match(modal, /target="_blank" rel="noreferrer"/);
});

test('the browser has a help topic on the studio anchor', () => {
  // The modal wears the badge; the registry must declare the topic or the
  // badge points at nothing.
  assert.match(modal, /HelpBadge topic="studio-civitai-browser"/);
  assert.match(topics, /'studio-civitai-browser'/);
});

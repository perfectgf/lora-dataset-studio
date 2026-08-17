import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const control = readFileSync(new URL('./DatasetCaptionControl.jsx', import.meta.url), 'utf8');
const source = readFileSync(new URL('./captionSource.js', import.meta.url), 'utf8');
const promptField = readFileSync(new URL('./PromptField.jsx', import.meta.url), 'utf8');
const comparisonSetup = readFileSync(new URL('./StudioRunSetup.jsx', import.meta.url), 'utf8');

test('the caption control offers BOTH libraries and persists the lock', () => {
  // Banks joined datasets as a source: the 🏷️ Caption pass captions a bank long
  // before anything is promoted, so the biggest pile of real captions on the
  // machine used to be the one this shortcut could not reach.
  assert.match(control, /apiFetch\('\/api\/dataset\/list'\)/);
  assert.match(control, /apiFetch\('\/api\/banks'\)/);
  // …and asked for WITHOUT ?rescan=1, which would re-walk every bank's folder.
  assert.doesNotMatch(control, /\/api\/banks\?rescan/);
  assert.match(control, /localStorage\.setItem\(STORAGE_KEY, JSON\.stringify\(dataset\)\)/);
  assert.match(control, /setLockedDataset\(choice\)/);
  assert.match(control, /drawCaption\(choice\)/);
});

test('the stored key never moved, and a kind-less choice still means dataset', () => {
  /* THE REGRESSION THIS EXISTS FOR. `studioCaptionDataset_v1` holds a choice made
     in people's browsers, from before banks were a source — `{id, name}`, no
     kind. Renaming the key would reset every one of them; reading the old shape
     as anything but a dataset would silently repoint a locked choice at whatever
     bank happens to share the number. The default in normaliseSource IS that
     alias path, and captionSource.test.js proves it behaves. */
  assert.match(control, /const STORAGE_KEY = 'studioCaptionDataset_v1';/);
  assert.match(source, /raw\?\.kind === BANK \? BANK : DATASET/);
});

test('the request body comes from the shared builder, so the dataset lane cannot drift', () => {
  // The literal `{ dataset_id: target.id }` moved into captionSource.js when a
  // second source appeared. The PROPERTY it guarded — a dataset asks with
  // dataset_id and nothing else — is asserted on the VALUE in
  // captionSource.test.js ("byte-identical to the one that shipped before").
  assert.match(control, /captionSourceBody\(target\)/);
  assert.match(control, /postJson\('\/api\/studio\/random-caption', body\)/);
  assert.match(source, /\{ bank_id: s\.id \} : \{ dataset_id: s\.id \}/);
});

test('the caption control keeps recovery actionable and accessible', () => {
  assert.match(control, /role="dialog" aria-modal="true"/);
  assert.match(control, /useFocusTrap\(dialogRef, open\)/);
  // The wording says SOURCE now — "dataset" would name half of what it offers.
  assert.match(control, /Choose or change the caption source/);
  assert.match(control, /Caption source locked/);
  assert.match(control, /err\?\.status === 422/);
  assert.match(control, /Choose a source/);
});

test('both Studio prompt surfaces use the shared control and protect typed prompts', () => {
  for (const src of [promptField, comparisonSetup]) {
    assert.match(src, /import DatasetCaptionControl from '\.\/DatasetCaptionControl'/);
    assert.match(src, /<DatasetCaptionControl onCaption=\{applyCaption\} \/>/);
    // The confirmation no longer promises a DATASET caption: the draw may come
    // from a bank, and a prompt is overwritten on the strength of that sentence.
    assert.match(src, /Replace the current (test )?prompt with a random caption drawn from your locked source\?/);
  }
});

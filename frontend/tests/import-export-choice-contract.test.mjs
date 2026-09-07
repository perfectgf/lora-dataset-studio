/* Two refusals the app made on the user's behalf without saying why
   (reported by Qeeyana on Reddit):
     - every imported image was resampled to 1024 px, with no setting anywhere;
     - a Style dataset could not export its ZIP until every image was captioned,
       with no way past it — which blocked the legitimate "let me caption these
       somewhere else" trip.

   node --test parses .js, not .jsx, so the component halves are asserted on
   source text; the sentence itself is a real module and is imported. */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { importInputLimitLine, importPolicyLine } from '../src/components/dataset/importPolicy.js';
import { helpTopics } from '../src/help/helpRegistry.js';

const here = dirname(fileURLToPath(import.meta.url));
const src = (p) => readFileSync(join(here, '..', 'src', p), 'utf8');

const dropzone = src('components/dataset/ImportDropzone.jsx');
const workspace = src('components/dataset/DatasetWorkspace.jsx');
const settings = src('components/settings/CaptioningSection.jsx');
const datasetHook = readFileSync(join(here, '..', 'src', 'hooks', 'useDataset.js'), 'utf8');

// --- the sentence at the point of import ------------------------------------

test('the import line states the resolution actually configured', () => {
  assert.equal(importPolicyLine({ max_side: 1024, encoding: 'standard' }),
    'stored as WebP q92, resized to 1024 px on the long side, ratio kept (input limit: 64 Mi-pixels and 16384 px per side)');
  assert.equal(importPolicyLine({ max_side: 2048, encoding: 'lossless' }),
    'stored as WebP lossless, resized to 2048 px on the long side, ratio kept (input limit: 64 Mi-pixels and 16384 px per side)');
});

test('every policy names the input safety limit before a conversion can happen', () => {
  const line = importPolicyLine({ max_side: 0, encoding: 'high', ceiling: 8192 });
  assert.match(line, /original size/);
  assert.match(line, /64 Mi-pixels/);
  assert.match(line, /16384 px/);
  assert.match(line, /WebP q100/);
  assert.equal(importInputLimitLine({ input_max_pixels: 16 * 1024 * 1024, input_max_side: 8192 }),
    '16 Mi-pixels and 8192 px per side');
});

test('a missing policy degrades to preserving originals, never to a stale WebP claim', () => {
  assert.equal(importPolicyLine(undefined),
    'stored byte-for-byte in the original file and format (input limit: 64 Mi-pixels and 16384 px per side)');
});

test('the dropzone reads the live policy, filters exactly the supported formats, and explains a rejection', () => {
  assert.ok(!dropzone.includes('(normalized to 1024, kept)'),
    'the hardcoded "normalized to 1024" claim must be gone from the UI text');
  assert.ok(dropzone.includes('importPolicyLine(importPolicy)'),
    'the line must be built from the live policy');
  assert.match(dropzone, /IMPORT_IMAGE_FORMATS} files stay byte-for-byte unchanged/,
    'the preservation promise must name the supported original formats');
  assert.ok(dropzone.includes('accept={IMPORT_IMAGE_ACCEPT}'),
    'the picker must use the exact static-image accept list');
  assert.ok(!dropzone.includes('accept="image/*"'),
    'the picker must not advertise unsupported image types');
  assert.match(dropzone, /Files larger than \{inputLimit\} are rejected — resize before importing, or raise the budget\./,
    'a size rejection must say how to remedy it — including the budget it can raise');
  assert.ok(dropzone.includes('focus="image-input-max-pixels"'),
    'the rejection hint must reach the budget that caused it');
  // and it points at the setting that changes it
  assert.ok(dropzone.includes('focus="dataset-import-encoding"'),
    'the hint must link to the setting it describes');
});

test('manual import makes server-side file refusals actionable instead of silently succeeding', () => {
  const importFiles = datasetHook.slice(datasetHook.indexOf('const importFiles ='),
    datasetHook.indexOf('// Concept only'));
  // `failed` is summed across the batches a drop is sent in (one request per
  // 20 files / ~64 MiB), so the refusal count is the drop's, not one batch's.
  assert.match(importFiles, /if \((?:d\.)?failed\) toast\.warning\(/);
  assert.match(importFiles, /failed \+= d\.failed \|\| 0/);
  assert.match(importFiles, /JPEG, PNG, WebP or BMP/);
  // No literal limit here any more: the budget is a setting, so the toast points
  // at it instead of carrying a copy that goes stale the moment it is changed.
  assert.ok(!/\d+ Mi-pixels/.test(importFiles),
    'the toast must not hardcode a budget the user can change');
  assert.match(importFiles, /Image size budget/);
  assert.match(importFiles, /resize a larger file, or raise the budget/);
});

// --- the setting exists, is resettable, is documented -----------------------

test('both import knobs are rendered and registered for help', () => {
  for (const id of ['dataset-import-max-side', 'dataset-import-encoding']) {
    assert.ok(settings.includes(`id="${id}"`), `${id} must be rendered`);
    assert.ok(helpTopics.some((t) => t.app?.focus === id),
      `${id} needs a help topic`);
  }
  assert.ok(settings.includes('section="dataset_import" field="max_side"'));
  assert.ok(settings.includes('section="dataset_import" field="encoding"'));
});

test('the resolution choice warns that changing it mid-way is not retroactive', () => {
  assert.ok(settings.includes('from now on'),
    'the setting must say it is not retroactive');
  assert.match(settings, /every source, including\s+WebP modes and Auto head-crop, must be no larger than/,
    'the same admission limit must be stated for preserve, crop, and WebP modes');
  assert.match(settings, /change the budget in <span className="font-medium">Image size budget<\/span> below,\s+or resize the file before importing/,
    'the settings explanation must make the refusal actionable');
});

// --- the export refusal is a refusal, not a wall ----------------------------

test('a Style export without captions can be confirmed past', () => {
  const guard = workspace.slice(workspace.indexOf('const exportZipGuarded'),
    workspace.indexOf('const importFolderPrompt'));
  assert.ok(guard.includes('isStyle && keptUncaptioned'), 'the guard must still exist');
  assert.ok(!/toast\.error/.test(guard),
    'the Style branch must refuse with a confirm, not a dead-end error toast');
  const style = guard.slice(guard.indexOf('isStyle && keptUncaptioned'));
  assert.match(style, /window\.confirm/);
  // cancelling still walks the user to the captions — the guard-rail stays useful
  assert.match(style, /jumpTo\(\{ targetId: 'gf-captions' \}\)/);
  // and it says WHY, and where the images can come back in
  assert.match(style, /Import dataset/);
});

test('the caption-elsewhere round trip is stated where the ZIP buttons are', () => {
  const at = workspace.indexOf('id="ds-caption-elsewhere"');
  assert.ok(at > 0, 'the round trip must be named next to the import/export buttons');
  const hint = workspace.slice(at, at + 900);
  assert.match(hint, /another tool/);
  assert.match(hint, /never overwritten/);
});

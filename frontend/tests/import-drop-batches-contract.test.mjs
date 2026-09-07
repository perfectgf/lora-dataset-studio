/** A drop is split into the requests the server takes — and every dropzone
 * hands the hook the server's figures to do it with.
 *
 * Source contract, like the others in this folder: node --test parses no JSX,
 * so it reads the files as text. What it pins is the wiring the reported 413
 * depended on (_nofaceman, Discord, 2026-09-07): the hook plans batches, the
 * dropzone passes the capability policy, and both workspace call sites pass it
 * through — the concept one used to drop the options on the floor.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const read = (rel) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8');
const hook = read('../src/hooks/useDataset.js');
const dropzone = read('../src/components/dataset/ImportDropzone.jsx');
const workspace = read('../src/components/dataset/DatasetWorkspace.jsx');

test('the hook splits a drop with planImportBatches and sends the batches one after another', () => {
  assert.match(hook, /import \{[^}]*planImportBatches[^}]*\} from '\.\.\/components\/dataset\/importBatches\.js'/);
  const body = hook.slice(hook.indexOf('const importFiles = useCallback'));
  assert.match(body, /planImportBatches\(files, policy\)/);
  assert.match(body, /for \(const batch of batches\)/);
  assert.match(body, /importBatchProgress\(sent, batch\.length, total\)/);
  assert.match(body, /oversizedFilesMessage\(oversized, limits\)/);
});

test('the dropzone passes the server policy along, and says the batch size', () => {
  assert.match(dropzone, /onImport\(files, \{ crop: [^}]*, policy: importPolicy \}\)/);
  assert.match(dropzone, /A big drop is sent in batches of \{maxFiles\} files\./);
});

test('both workspace dropzones forward the policy to the hook', () => {
  const sites = workspace.match(/<ImportDropzone[^>]*onImport=\{\(f, o\) => ds\.importFiles\(f, [^)]*\)\}/g) || [];
  assert.equal(sites.length, 2, 'concept and character dropzones both pass the options through');
  assert.ok(sites.every((s) => /policy/.test(s) || /ds\.importFiles\(f, o\)/.test(s)));
});

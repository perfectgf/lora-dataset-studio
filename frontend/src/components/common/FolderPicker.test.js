import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';

const picker = fs.readFileSync(new URL('./FolderPicker.jsx', import.meta.url), 'utf8');
const bank = fs.readFileSync(new URL('../../pages/BankPage.jsx', import.meta.url), 'utf8');
const dsWorkspace = fs.readFileSync(
  new URL('../dataset/DatasetWorkspace.jsx', import.meta.url), 'utf8');

test('the field tries the native server dialog first, in-app browser as fallback', () => {
  // Browse hits the native-dialog endpoint...
  assert.match(picker, /postJson\('\/api\/system\/pick-folder'/);
  // ...and only opens the in-app browser when the server has no native dialog.
  assert.match(picker, /if \(r\.available\)/);
  assert.match(picker, /setBrowsing\(true\)/);
  // The browser lists folders through the read-only listing endpoint.
  assert.match(picker, /\/api\/system\/list-folders/);
});

test('a cancelled native dialog leaves the value untouched', () => {
  // available + no path === cancelled: nothing is written.
  assert.match(picker, /if \(r\.path\) onChange\(r\.path\)/);
});

test('pickNativeFolder never throws on the expected no-desktop case', () => {
  // A network/endpoint failure degrades to available:false so the caller falls
  // back rather than surfacing an error.
  assert.match(picker, /catch\s*\{\s*return \{ available: false/);
});

test('the path field stays editable (pasting a path still works)', () => {
  assert.match(picker, /onChange=\{\(e\) => onChange\(e\.target\.value\)\}/);
});

test('the in-app browser lists folders only and never writes', () => {
  // "Use this folder" is disabled at the drive roots (must descend into a dir).
  assert.match(picker, /disabled=\{atRoot \|\| loading\}/);
  // Only the GET listing + the native POST are called — no mutating folder call.
  assert.doesNotMatch(picker, /\/api\/system\/(create|delete|write)/);
});

test('the Image bank uses the shared field for its folder input', () => {
  assert.match(bank, /import FolderPickerField from '\.\.\/components\/common\/FolderPicker'/);
  assert.match(bank, /<FolderPickerField[^>]*value=\{folder\}/s);
  // The bare <input id="bank-folder"> is gone (replaced by the field).
  assert.doesNotMatch(bank, /<input id="bank-folder"/);
});

test('dataset folder-import is native-first with the browser fallback', () => {
  assert.match(dsWorkspace,
    /import \{ pickNativeFolder, FolderBrowserModal \} from '\.\.\/common\/FolderPicker'/);
  assert.match(dsWorkspace, /await pickNativeFolder\(\)/);
  assert.match(dsWorkspace, /setFolderBrowseOpen\(true\)/);
  assert.match(dsWorkspace, /<FolderBrowserModal/);
  // the old blocking window.prompt path is gone
  assert.doesNotMatch(dsWorkspace, /window\.prompt\(\s*\n?\s*'Path of the dataset folder/);
});

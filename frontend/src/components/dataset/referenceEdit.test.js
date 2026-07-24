import test from 'node:test';
import assert from 'node:assert/strict';
import {
  EDIT_ENGINES, defaultEditEngine, editBlockedReason, batchLiveNote, editPhase,
} from './referenceEdit.js';
import { STORAGE_ENGINES, STORAGE_PRIMARY } from './engineSelection.js';

function fakeStorage(seed = {}) {
  const data = { ...seed };
  return { getItem(k) { return k in data ? data[k] : null; }, setItem(k, v) { data[k] = String(v); } };
}

test('EDIT_ENGINES excludes Klein (edit is API-only this wave)', () => {
  assert.deepEqual(EDIT_ENGINES, ['chatgpt', 'nanobanana']);
});

test('defaultEditEngine mirrors the primary generation engine when it can edit', () => {
  assert.equal(defaultEditEngine(fakeStorage({ [STORAGE_ENGINES]: JSON.stringify(['nanobanana']) })),
    'nanobanana');
  assert.equal(defaultEditEngine(fakeStorage({ [STORAGE_PRIMARY]: 'chatgpt' })), 'chatgpt');
});

test('defaultEditEngine falls back to ChatGPT when the primary cannot edit', () => {
  // Klein is the primary but cannot edit -> a live default, not a dead selection.
  assert.equal(defaultEditEngine(fakeStorage({ [STORAGE_ENGINES]: JSON.stringify(['klein']) })),
    'chatgpt');
});

test('defaultEditEngine with no stored preference uses the historic default (Nano Banana)', () => {
  // readEngines falls back to DEFAULT_ENGINE (nanobanana), which CAN edit.
  assert.equal(defaultEditEngine(fakeStorage()), 'nanobanana');
});

test('editBlockedReason blocks an empty prompt and an un-editable engine', () => {
  assert.equal(editBlockedReason('add glasses', 'chatgpt'), null);
  assert.match(editBlockedReason('', 'chatgpt'), /describe/i);
  assert.match(editBlockedReason('   ', 'nanobanana'), /describe/i);
  assert.match(editBlockedReason('x', 'klein'), /ChatGPT or Nano Banana/i);
});

test('batchLiveNote informs only while a generate batch runs, never blocks', () => {
  assert.equal(batchLiveNote(null), null);
  assert.equal(batchLiveNote({ kind: 'caption' }), null);
  assert.match(batchLiveNote({ kind: 'generate' }), /future batches/i);
});

test('editPhase derives the modal phase from the server reference_edit object', () => {
  assert.equal(editPhase(null), 'idle');
  assert.equal(editPhase(undefined), 'idle');
  assert.equal(editPhase({ status: 'running' }), 'running');
  assert.equal(editPhase({ status: 'ready', candidate_filename: 'x.webp' }), 'ready');
  assert.equal(editPhase({ status: 'failed', error: 'boom' }), 'failed');
  assert.equal(editPhase({ status: 'weird' }), 'idle');   // unknown → idle (form)
});

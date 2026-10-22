// Reads the image Bank TREE, not one file: the Encre redesign split the
// workspace into a top bar, a filter rail, a passes panel and the grid, and a
// wiring assertion must survive a move (see bankTreeSource.js).
import { bankTreeSource } from './bankTreeSource.js';
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  datasetConflictBlock, deleteDestination, deletePreviewState,
  deleteRejectedStart, isRecoverable, overlapNotice, sharedFileCount,
  sharedFilesWarning,
} from './bankOverlap.js';

test('a permanent delete is never described as recoverable', () => {
  assert.equal(isRecoverable('trash'), true);
  assert.equal(isRecoverable('app_trash'), true);
  assert.equal(isRecoverable('delete'), false);
  assert.match(deleteDestination('delete'), /for good/);
});

test('each destination names a place the user can actually go look', () => {
  assert.match(deleteDestination('trash'), /Recycle Bin/);
  assert.match(deleteDestination('app_trash'), /Settings/);
});

test('an unknown mode is treated as permanent, never as safe', () => {
  assert.equal(isRecoverable(undefined), false);
  assert.match(deleteDestination(undefined), /for good/);
});

test('overlapNotice: a bank on its own says nothing', () => {
  assert.equal(overlapNotice([]), null);
  assert.equal(overlapNotice(null), null);
  assert.equal(overlapNotice([{ id: 1 }]), null);   // nameless row is not a notice
});

test('overlapNotice names the other bank and why it matters', () => {
  const n = overlapNotice([{ id: 2, name: 'Everything', relation: 'parent' }]);
  assert.match(n, /Everything/);
  assert.match(n, /Delete rejected/);
});

test('overlapNotice lists every overlapping bank', () => {
  const n = overlapNotice([{ id: 2, name: 'A' }, { id: 3, name: 'B' }]);
  assert.match(n, /“A”, “B”/);
});

test('sharedFilesWarning: nothing shared -> no warning', () => {
  assert.equal(sharedFilesWarning({ shared: [] }), null);
  assert.equal(sharedFilesWarning(null), null);
  assert.equal(sharedFilesWarning({ shared: [{ name: 'A', files: 0 }] }), null);
});

test('sharedFilesWarning says how many files the other bank loses', () => {
  const w = sharedFilesWarning({ shared: [{ id: 2, name: 'Everything', files: 1541 }] });
  assert.match(w, /1541/);
  assert.match(w, /Everything/);
  assert.match(w, /decisions/);      // the cost is the triage, not just the bytes
});

test('sharedFileCount totals across banks and survives junk', () => {
  assert.equal(sharedFileCount({ shared: [{ files: 3 }, { files: 4 }] }), 7);
  assert.equal(sharedFileCount({ shared: [{ files: 'x' }] }), 0);
  assert.equal(sharedFileCount(null), 0);
});

// ── The destructive button fails CLOSED when its evidence is missing ─────────

test('deletePreviewState: no preview yet -> not ready, and it says it is checking', () => {
  const s = deletePreviewState(null);
  assert.equal(s.ready, false);
  assert.equal(s.state, 'checking');
  assert.match(s.title, /Checking/);
});

test('deletePreviewState: a FAILED preview never arms the button', () => {
  const s = deletePreviewState({ failed: true });
  assert.equal(s.ready, false);
  assert.equal(s.state, 'failed');
  // and it says so out loud — the old code dropped the ⚠ banner in silence
  assert.match(s.title, /Could not check/);
  assert.match(s.text, /Nothing is deleted/);
});

test('deletePreviewState: a real preview arms, even with nothing shared', () => {
  assert.equal(deletePreviewState({ mode: 'trash', shared: [] }).ready, true);
  assert.equal(deletePreviewState({ mode: 'delete' }).state, 'ready');
});

test('the delete dialog wires the fail-closed check into the destructive button', () => {
  const src = fs.readFileSync(new URL('./DeleteRejectedDialog.jsx', import.meta.url), 'utf8');
  // armed alone is not enough: the preview has to have landed
  assert.match(src, /disabled=\{busy \|\| !armed \|\| !check\.ready \|\| block\.blocked\}/);
  assert.match(src, /if \(busy \|\| !armed \|\| !check\.ready \|\| block\.blocked\) return/);
  // and an unverified destination is never asserted as fact
  assert.match(src, /still being checked/);
});

test('datasetConflictBlock: silent by default, hard stop when the bank is on a dataset', () => {
  assert.equal(datasetConflictBlock(null).blocked, false);
  assert.equal(datasetConflictBlock(undefined).blocked, false);
  const b = datasetConflictBlock({ scope: 'dataset', dataset_id: 7, message: 'X — Import to bank copies.' });
  assert.equal(b.blocked, true);
  assert.match(b.title, /dataset/);
  assert.equal(b.text, 'X — Import to bank copies.');
});

test('datasetConflictBlock: with no server sentence it still names the way out', () => {
  // The block must never degrade into a bare "no" just because the payload is
  // older/leaner than this build.
  const b = datasetConflictBlock({ scope: 'dataset' });
  assert.equal(b.blocked, true);
  assert.match(b.text, /Import to bank/);
});

test('the delete dialog blocks on a dataset conflict rather than warning about it', () => {
  const src = fs.readFileSync(new URL('./DeleteRejectedDialog.jsx', import.meta.url), 'utf8');
  assert.match(src, /datasetConflictBlock\(preview\?\.dataset_conflict\)/);
});

test('the workspace disables 🗑 Delete rejected for dataset conflicts and protected sources', () => {
  // The banner promises "Delete rejected is disabled here". A button that still
  // opened the dialog, only to be refused inside it, would make that a lie.
  const src = bankTreeSource();
  const guard = src.match(/onClick=\{\(\) => setDeleteRejectedOpen\(true\)\}\s+disabled=\{([^}]+)\}/)?.[1];
  assert.ok(guard, 'the destructive button has a disabled guard');
  assert.match(guard, /!!payload\?\.dataset_conflict/);
  assert.match(guard, /!!payload\?\.source_write_block/);
  assert.match(guard, /savingReadOnly/);
  assert.match(src, /payload\?\.dataset_conflict && \(/);
});

test('deleteRejectedStart: a started run points at the bar instead of going quiet', () => {
  // The complaint this answers: the dialog closed and nothing said whether the
  // deletion was running, finished, or dead.
  const n = deleteRejectedStart({ ok: true, total: 4200 });
  assert.equal(n.type, 'info');
  assert.match(n.text, /4200 rejected file\(s\)/);
  assert.match(n.text, /top of the bank/);
  assert.match(n.text, /stop it/);
});

test('deleteRejectedStart: a run that already finished reports its numbers', () => {
  const n = deleteRejectedStart(
    { total: 3, mode: 'trash', deleted: 0, trashed: 2, already_absent: 1,
      rows_removed: 3, skipped: [] },
    { destination: 'the Recycle Bin', recoverable: true });
  assert.equal(n.type, 'success');
  assert.match(n.text, /3 rejected file\(s\) moved to the Recycle Bin/);
});

test('deleteRejectedStart: a permanent delete is never called recoverable', () => {
  const n = deleteRejectedStart(
    { mode: 'delete', deleted: 2, trashed: 0, already_absent: 0, rows_removed: 2,
      skipped: [{ relpath: 'a.jpg', reason: 'locked' }] },
    { destination: 'nowhere', recoverable: false });
  assert.match(n.text, /permanently deleted/);
  assert.match(n.text, /1 skipped/);
});

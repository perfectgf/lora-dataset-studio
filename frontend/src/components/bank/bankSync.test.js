import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bankListSyncToast, folderCheckNote, folderSyncNote, folderSyncToast,
} from './bankSync.js';

const clean = { added: 0, missing: 0, unavailable: false, error: null };

test('folderSyncToast: nothing added -> silent', () => {
  assert.equal(folderSyncToast(clean), null);
  assert.equal(folderSyncToast(null), null);
  assert.equal(folderSyncToast(undefined), null);
});

test('folderSyncToast: new files are announced (the counters must never move mutely)', () => {
  const t = folderSyncToast({ ...clean, added: 42 });
  assert.equal(t.type, 'success');
  assert.match(t.text, /42 new image\(s\)/);
});

test('folderSyncToast: a real failure still surfaces as an error', () => {
  const t = folderSyncToast({ ...clean, error: 'the folder could not be read' });
  assert.equal(t.type, 'error');
  assert.match(t.text, /could not be read/);
});

test('folderSyncToast: hitting the ceiling warns with BOTH numbers and a remedy', () => {
  // The ceiling no longer refuses the batch, so the toast has to say what DID
  // land as well as what did not — and what to do about it.
  const t = folderSyncToast({ ...clean, added: 120, not_added: 30, limit: 200000 });
  assert.equal(t.type, 'warning');
  assert.match(t.text, /120 new image\(s\) added/);
  assert.match(t.text, /30 were not/);
  assert.match(t.text, /200,000/);
  assert.match(t.text, /second bank/);
});

test('folderSyncToast: nothing fitted at all is still spoken, not silent', () => {
  const t = folderSyncToast({ ...clean, added: 0, not_added: 500, limit: 200000 });
  assert.equal(t.type, 'warning');
  assert.match(t.text, /500 were not/);
});

test('bankListSyncToast: aggregates one line for the whole list', () => {
  assert.equal(bankListSyncToast([{ folder_sync: clean }, {}]), null);
  const one = bankListSyncToast([{ folder_sync: { ...clean, added: 3 } }, { folder_sync: clean }]);
  assert.match(one.text, /^3 new image\(s\) found in the folder/);
  const many = bankListSyncToast([
    { folder_sync: { ...clean, added: 3 } },
    { folder_sync: { ...clean, added: 4 } },
  ]);
  assert.match(many.text, /7 new image\(s\) found across 2 banks/);
});

test('bankListSyncToast: a bank at its ceiling warns from the LIST too', () => {
  const t = bankListSyncToast([
    { folder_sync: { ...clean, added: 10 } },
    { folder_sync: { ...clean, added: 0, not_added: 7 } },
  ]);
  assert.equal(t.type, 'warning');
  assert.match(t.text, /10 new image\(s\) added/);
  assert.match(t.text, /7 were not/);
});

test('folderSyncNote: in sync -> no note', () => {
  assert.equal(folderSyncNote(clean), null);
  assert.equal(folderSyncNote(null), null);
});

test('folderSyncNote: missing files are reported as KEPT, never deleted', () => {
  const n = folderSyncNote({ ...clean, missing: 7 });
  assert.equal(n.tone, 'warn');
  assert.match(n.text, /7 image\(s\)/);
  assert.match(n.text, /kept/);
});

test('folderSyncNote: an unavailable folder says the bank is intact', () => {
  const n = folderSyncNote({ ...clean, unavailable: true, missing: 0 });
  assert.equal(n.tone, 'error');
  assert.match(n.text, /unavailable/);
  assert.match(n.text, /keeps every image and decision/);
});

test('folderSyncNote: both off-sync states offer to repoint the bank', () => {
  // A folder that moved is the usual cause of BOTH shapes, and repointing is
  // the fix — the note is where the user is already looking, so it carries the
  // affordance instead of leaving them to hunt for it.
  for (const sync of [{ ...clean, unavailable: true }, { ...clean, missing: 7 }]) {
    const n = folderSyncNote(sync);
    assert.equal(n.canRelocate, true);
    assert.match(n.text, /new location/);
  }
  assert.equal(folderSyncNote(clean), null);
});

test('folderSyncNote: unavailable wins over a stale missing count', () => {
  assert.match(folderSyncNote({ ...clean, unavailable: true, missing: 900 }).text,
    /unavailable/);
});

test('folderSyncNote: missing files also offer to FORGET them, count attached', () => {
  // The warning's second honest cause — the files were really deleted (a
  // downloader cleaning up its own intermediates, a by-hand tidy) — gets its
  // own remedy, and the button needs the number to say what it would drop.
  const n = folderSyncNote({ ...clean, missing: 8175 });
  assert.equal(n.canForget, true);
  assert.equal(n.missing, 8175);
  assert.match(n.text, /really gone/);
});

test('folderSyncNote: an unavailable folder never offers to forget', () => {
  // There the walk itself failed: EVERY file reads as missing, and a forget
  // trusting that verdict would erase the whole triage. The server refuses the
  // call too — this pins that the UI does not even dangle the button.
  const n = folderSyncNote({ ...clean, unavailable: true, missing: 900 });
  assert.ok(!n.canForget);
});

// --- folderCheckNote: the price of not walking on every page load ------------
// The bank list stopped re-inventorying every source folder before rendering
// (690-1 190 ms on a real 86 493-image library, on a page people pass through).
// The counts can therefore lag, and the ONLY thing that makes that acceptable
// is the page saying so — a silently stale list would be worse than a slow one.
test('folderCheckNote: an unwalked list says its counts can lag, and how to fix it', () => {
  const n = folderCheckNote([{ folder_sync: { ...clean, walked: false, age: null } }]);
  assert.equal(n.stale, true);
  assert.match(n.text, /Rescan folders/);
});

test('folderCheckNote: after a rescan it says how fresh it is, not that it is stale', () => {
  const n = folderCheckNote([
    { folder_sync: { ...clean, walked: true, age: 4 } },
    { folder_sync: { ...clean, walked: true, age: 130 } },
  ]);
  assert.equal(n.stale, false);
  // The OLDEST walk is the honest one to quote for a list of several banks.
  assert.match(n.text, /2 min ago/);
});

test('folderCheckNote: one unchecked bank among fresh ones is counted, not hidden', () => {
  const n = folderCheckNote([
    { folder_sync: { ...clean, walked: true, age: 3 } },
    { folder_sync: { ...clean, walked: false, age: null } },
  ]);
  assert.equal(n.stale, true);
  assert.match(n.text, /1 of these banks/);
});

test('folderCheckNote: nothing to be honest about -> no line', () => {
  assert.equal(folderCheckNote([]), null);
  assert.equal(folderCheckNote(null), null);
  assert.equal(folderCheckNote([{}]), null);
});

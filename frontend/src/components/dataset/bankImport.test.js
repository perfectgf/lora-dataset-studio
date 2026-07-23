import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
  bankImportOption, bankImportOptions, promotableUrl, promoteUrl, promoteAllBody,
  banksUrl, promotableCounts, isBankJobLive, bankActivity, promoteOutcome,
} from './bankImport.js';

// A bank that was scanned but never triaged: everything is still undecided.
const untriaged = { id: 3, name: 'Dump', total: 900, keep: 0, reject: 0 };
// A bank whose kept set already landed on this dataset.
const exhausted = { id: 4, name: 'Shoot A', total: 120, keep: 40, reject: 80 };

test('0 promotable because nothing was KEPT is not the same as 0 because already imported', () => {
  const a = bankImportOption(untriaged, 0);
  const b = bankImportOption(exhausted, 0);
  assert.equal(a.ready, false);
  assert.equal(b.ready, false);
  assert.notEqual(a.reason, b.reason);
  assert.equal(a.reason, 'untriaged');
  assert.equal(b.reason, 'already-imported');
  // and the wording must SAY which is which — a bare "nothing to promote" is the bug
  assert.match(a.hint, /undecided/i);
  assert.match(a.hint, /triage/i);
  assert.match(b.hint, /already/i);
  assert.match(b.hint, /40/);
  assert.doesNotMatch(a.hint, /already/i);
});

test('an all-rejected bank says so instead of asking for more triage', () => {
  const o = bankImportOption({ id: 9, name: 'Rejects', total: 50, keep: 0, reject: 50 }, 0);
  assert.equal(o.reason, 'all-rejected');
  assert.match(o.hint, /rejected/i);
  assert.doesNotMatch(o.hint, /undecided/i);
});

test('an empty bank is its own case, never "triage it"', () => {
  const o = bankImportOption({ id: 1, name: 'New', total: 0, keep: 0, reject: 0 }, 0);
  assert.equal(o.reason, 'empty');
  assert.equal(o.ready, false);
  assert.doesNotMatch(o.hint, /triage/i);
});

test('a promotable bank is importable and announces the honest count', () => {
  const o = bankImportOption({ id: 7, name: 'Shoot B', total: 300, keep: 210, reject: 90 }, 12);
  assert.equal(o.ready, true);
  assert.equal(o.reason, null);
  assert.equal(o.count, 12);
  assert.match(o.hint, /12 kept images to import/);
  // singular is not "1 kept images"
  assert.match(bankImportOption(untriaged, 1).hint, /1 kept image to import/);
});

test('the count is unknown until it loads — never rendered as 0', () => {
  const o = bankImportOption(exhausted, undefined);
  assert.equal(o.reason, 'loading');
  assert.equal(o.count, null);
  assert.equal(o.ready, false);
  assert.equal(bankImportOption(exhausted, null).reason, 'loading');
});

test('bankImportOptions keeps the server order and pairs each bank with ITS count', () => {
  const rows = bankImportOptions([untriaged, exhausted], { 4: 5 });
  assert.deepEqual(rows.map((r) => r.id), [3, 4]);
  assert.equal(rows[0].reason, 'loading');   // no count yet for bank 3
  assert.equal(rows[1].ready, true);
  assert.equal(rows[1].count, 5);
  assert.deepEqual(bankImportOptions(null), []);
});

test('the promotable route is per-target: it carries THIS dataset id', () => {
  assert.equal(promotableUrl(3, 42), '/api/bank/3/promotable?dataset_id=42');
  // string ids from a <select> must not leak into the URL as-is
  assert.equal(promotableUrl('3', '42'), '/api/bank/3/promotable?dataset_id=42');
  assert.equal(promoteUrl(3), '/api/bank/3/promote');
  assert.deepEqual(promoteAllBody('42'), { dataset_id: 42, image_ids: [] });
});

test('a live bank job is one that exists and has not finished', () => {
  assert.equal(isBankJobLive(null), false);
  assert.equal(isBankJobLive({ kind: 'promote', finished: false }), true);
  assert.equal(isBankJobLive({ kind: 'promote', finished: true }), false);
  const banks = [{ id: 3, activity: { kind: 'promote', finished: false } }, { id: 4 }];
  assert.equal(bankActivity(banks, 3).kind, 'promote');
  assert.equal(bankActivity(banks, '3').kind, 'promote', 'ids compare numerically');
  assert.equal(bankActivity(banks, 4), null);
  assert.equal(bankActivity([], 3), null);
});

test('the end-of-job message repeats the server, it never invents a number', () => {
  assert.equal(promoteOutcome(null), null);
  assert.deepEqual(promoteOutcome({ error: 'boom' }), { kind: 'error', text: 'Import failed — boom' });
  assert.deepEqual(promoteOutcome({ cancelled: true, detail: 'Stopped — 4 copied' }),
    { kind: 'info', text: 'Stopped — 4 copied' });
  assert.deepEqual(promoteOutcome({ detail: '17 images copied' }),
    { kind: 'success', text: '17 images copied' });
  assert.equal(promoteOutcome({}).text, 'Import finished.');
});

test('the bank list carries the target so the counts come back with it', () => {
  assert.equal(banksUrl(42), '/api/banks?dataset_id=42');
  assert.equal(banksUrl('42'), '/api/banks?dataset_id=42', 'ids are coerced');
});

test('counts are read off the list, and a bank without one stays unknown', () => {
  const banks = [
    { id: 3, promotable: 12 },
    { id: 4, promotable: 0 },
    { id: 5 },                  // server omitted it — we do NOT invent a 0
  ];
  assert.deepEqual(promotableCounts(banks), { 3: 12, 4: 0 });
  assert.deepEqual(promotableCounts([]), {});
  assert.deepEqual(promotableCounts(undefined), {});
  // …and the unknown one renders as "Counting…", not as an empty bank.
  const [, , unknown] = bankImportOptions(banks, promotableCounts(banks));
  assert.equal(unknown.reason, 'loading');
  assert.equal(unknown.ready, false);
});

// ---- wiring guards on the panel (JSX can't be imported by node --test) ------

const panel = readFileSync(new URL('./BankImportPanel.jsx', import.meta.url), 'utf8');

test('the panel goes through the shared route helpers, not hand-built URLs', () => {
  assert.match(panel, /banksUrl\(/);
  assert.match(panel, /promoteUrl\(/);
  assert.doesNotMatch(panel, /`\/api\/bank\/\$\{[^}]+\}\/promote/);
});

test('the chooser reads its counts off the bank list, one request for all banks', () => {
  // Opening the panel on a library of N banks must not cost N /promotable calls.
  assert.doesNotMatch(panel, /promotableUrl/);
  assert.doesNotMatch(panel, /apiFetch\('\/api\/banks'\)/);
  assert.match(panel, /promotableCounts\(/);
});

test('the promote POST is caught — postJson throws on 400/409', () => {
  // a 409 (a job already runs on that bank) must surface as a toast, not a dead click
  assert.match(panel, /catch\s*\(/);
  assert.match(panel, /toast\.error/);
});

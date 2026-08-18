import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BANK_SCRAPE_BATCH,
  bankScrapeBatches,
  bankScrapeDestination,
  runBankScrapeImport,
  summarizeBankScrapeImport,
} from './bankScrapeImport.js';

const items = (n) => Array.from({ length: n }, (_, i) => ({ url: `http://x/${i}.jpg` }));

test('a big selection is cut into server-sized batches', () => {
  assert.equal(bankScrapeBatches(items(130)).length, 3);
  assert.equal(bankScrapeBatches(items(130))[2].length, 130 - 2 * BANK_SCRAPE_BATCH);
  assert.deepEqual(bankScrapeBatches(null), []);
});

test('a new-bank destination needs a name, an existing one a real id', () => {
  assert.deepEqual(bankScrapeDestination({ mode: 'new', name: ' Pile ' }), { name: 'Pile' });
  assert.equal(bankScrapeDestination({ mode: 'new', name: '  ' }), null);
  assert.deepEqual(bankScrapeDestination({ mode: 'existing', bankId: '7' }), { bank_id: 7 });
  assert.equal(bankScrapeDestination({ mode: 'existing', bankId: '' }), null);
});

test('a multi-batch scrape into a NEW bank creates exactly one bank', async () => {
  const sent = [];
  const post = async (_url, body) => {
    sent.push(body);
    return { ok: true, bank_id: 12, created: sent.length === 1, saved: body.items.length,
      already_there: 0, added: body.items.length, skipped: {} };
  };
  const res = await runBankScrapeImport({
    items: items(130), destination: { name: 'Pile' }, post });
  assert.equal(res.ok, true);
  assert.equal(res.bankId, 12);
  assert.equal(res.saved, 130);
  assert.equal(sent.length, 3);
  assert.deepEqual(sent[0].name, 'Pile');
  // batches 2 and 3 RESUME — no second bank, no name resent
  assert.equal(sent[1].name, undefined);
  assert.equal(sent[1].bank_id, 12);
  assert.equal(sent[2].bank_id, 12);
});

test('resuming into an existing bank never sends a name', async () => {
  const sent = [];
  const post = async (_u, body) => {
    sent.push(body);
    return { ok: true, bank_id: 3, created: false, saved: 1, added: 1, skipped: {} };
  };
  await runBankScrapeImport({ items: items(1), destination: { bank_id: 3 }, post });
  assert.deepEqual(sent, [{ items: sent[0].items, bank_id: 3 }]);
});

test('a failing batch stops and reports what already landed', async () => {
  let n = 0;
  const post = async (_u, body) => {
    n += 1;
    if (n === 2) return { ok: false, error: 'boom' };
    return { ok: true, bank_id: 5, created: true, saved: body.items.length,
      added: body.items.length, skipped: {} };
  };
  const res = await runBankScrapeImport({
    items: items(130), destination: { name: 'Pile' }, post });
  assert.equal(res.ok, false);
  assert.equal(res.error, 'boom');
  assert.equal(res.saved, BANK_SCRAPE_BATCH);
  assert.equal(res.bankId, 5);
});

test('an empty selection or a missing destination is refused before any request', async () => {
  let called = false;
  const post = async () => { called = true; return { ok: true }; };
  assert.equal((await runBankScrapeImport({ items: [], destination: { name: 'x' }, post })).ok, false);
  assert.equal((await runBankScrapeImport({ items: items(1), destination: null, post })).ok, false);
  assert.equal(called, false);
});

test('the summary never calls a byte-identical re-download a duplicate', () => {
  const msg = summarizeBankScrapeImport({ saved: 4, alreadyThere: 2, added: 4,
    skipped: { errors: 1 } });
  assert.match(msg, /4 image\(s\) downloaded/);
  assert.match(msg, /2 already in the folder/);
  assert.match(msg, /1 could not be downloaded/);
  assert.doesNotMatch(msg, /duplicate/i);
});


test('a server refusal (thrown post) surfaces instead of escaping as a rejection', async () => {
  // The REAL post (fetchClient.postJson) THROWS on any non-2xx — a resolved
  // {ok:false} never happens for a 400/409. This pins the catch that turned
  // every server refusal from an unhandled rejection (no toast, totals lost,
  // onDone skipped) into the same envelope a soft failure uses.
  let calls = 0;
  const post = async (_url, body) => {
    calls += 1;
    if (calls === 1) {
      return { ok: true, bank_id: 9, created: true, saved: body.items.length,
        already_there: 0, added: body.items.length, skipped: {} };
    }
    const err = new Error('HTTP 409');
    err.status = 409;
    err.body = { error: 'a pass owns this bank right now' };
    throw err;
  };
  const res = await runBankScrapeImport({
    items: items(BANK_SCRAPE_BATCH + 3),
    destination: { name: 'Pile' },
    post,
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, 'a pass owns this bank right now');
  assert.equal(res.status, 409);
  // The first batch's work is not erased by the second batch's refusal.
  assert.equal(res.saved, BANK_SCRAPE_BATCH);
  assert.equal(res.bankId, 9);
});


test('a thrown post with no JSON body still yields a readable error', async () => {
  const post = async () => {
    const err = new Error('Network error');
    throw err;
  };
  const res = await runBankScrapeImport({
    items: items(2), destination: { name: 'Pile' }, post,
  });
  assert.equal(res.ok, false);
  assert.equal(res.error, 'Network error');
});


test('a sync_error rides the totals to the caller', async () => {
  const post = async (_url, body) => ({
    ok: true, bank_id: 3, saved: body.items.length, already_there: 0,
    added: 0, skipped: {}, sync_error: 'the source folder is not reachable right now',
  });
  const res = await runBankScrapeImport({
    items: items(2), destination: { bank_id: 3 }, post,
  });
  assert.equal(res.ok, true);
  assert.equal(res.syncError, 'the source folder is not reachable right now');
});

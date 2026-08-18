import test from 'node:test';
import assert from 'node:assert/strict';
import {
  VIDEO_BANK_SCRAPE_BATCH,
  VIDEO_BANK_SCRAPE_ENDPOINT,
  findVideoBank,
  runVideoBankScrapeImport,
  scrapableVideoBanks,
  summarizeVideoBankScrapeImport,
  videoBankScrapeDestination,
  videoBankScrapeFolderNotice,
  videoBankScrapeNextStep,
} from './videoBankScrapeImport.js';

const bank = (id, scrapable) => ({ id, name: `b${id}`, scrapable, counts: { sources: 3 } });
const items = (n) => Array.from({ length: n }, (_, i) => ({ url: `https://x/${i}.mp4` }));

test('a new bank needs a name, an existing one needs an id', () => {
  assert.deepEqual(videoBankScrapeDestination({ mode: 'new', name: 'Clips' }),
    { name: 'Clips' });
  assert.equal(videoBankScrapeDestination({ mode: 'new', name: '   ' }), null);
  assert.deepEqual(
    videoBankScrapeDestination({ mode: 'existing', bankId: '4', banks: [bank(4, true)] }),
    { bank_id: 4 });
  assert.equal(
    videoBankScrapeDestination({ mode: 'existing', bankId: '', banks: [bank(4, true)] }),
    null);
  assert.equal(
    videoBankScrapeDestination({ mode: 'existing', bankId: '0', banks: [bank(4, true)] }),
    null);
});

test('a bank over the user’s own folder IS a destination', () => {
  // This is the change. `app_folder: false` means the user pointed the bank at a
  // folder of their own — which used to disqualify it, and now only decides
  // whether there is something to say before the click. Picking it is consent.
  const own = { id: 7, name: 'Own footage', scrapable: true, app_folder: false,
    source_path: '/media/rushes' };
  assert.deepEqual(
    videoBankScrapeDestination({ mode: 'existing', bankId: '7', banks: [own] }),
    { bank_id: 7 });
});

test('a bank sitting on a dataset folder is still not a destination', () => {
  // The one refusal consent cannot lift. Refusing it HERE means the user is not
  // offered a choice the server would answer with a 400 after they clicked.
  assert.equal(
    videoBankScrapeDestination({ mode: 'existing', bankId: '7', banks: [bank(7, false)] }),
    null);
  assert.equal(
    videoBankScrapeDestination({ mode: 'existing', bankId: '9', banks: [bank(7, true)] }),
    null);
  // With no bank list to check against (the page has not loaded them yet) the
  // id is taken at face value — the server is still the authority.
  assert.deepEqual(videoBankScrapeDestination({ mode: 'existing', bankId: '7' }),
    { bank_id: 7 });
});

test('every bank is offered except one whose folder belongs to a dataset', () => {
  assert.deepEqual(scrapableVideoBanks([bank(1, true), bank(2, false), bank(3, true)])
    .map((b) => b.id), [1, 3]);
  assert.deepEqual(scrapableVideoBanks(null), []);
  assert.deepEqual(scrapableVideoBanks([null, undefined]), []);
});

test('the folder notice fires only for a folder of the user’s own, and names it', () => {
  // A bank the app created has a folder nobody else looks at — saying "files
  // will go in it" there is noise. A folder you assembled yourself is the case
  // worth one sentence, with the path, because that is what someone would go and
  // check.
  const own = { id: 7, app_folder: false, source_path: 'D:\\footage\\rushes 2026' };
  const notice = videoBankScrapeFolderNotice(own);
  assert.match(notice, /Downloads will be added to this bank/);
  assert.ok(notice.includes('D:\\footage\\rushes 2026'));
  assert.equal(videoBankScrapeFolderNotice({ id: 8, app_folder: true,
    source_path: '/data/video_bank_sources/x' }), '');
  assert.equal(videoBankScrapeFolderNotice(null), '');
  // A row that predates the field (or a payload that lost it) still warns, and
  // simply cannot name the folder — silence would be the wrong way to fail here.
  const noPath = videoBankScrapeFolderNotice({ id: 9, app_folder: false });
  assert.match(noPath, /Downloads will be added to this bank/);
  assert.ok(!noPath.includes('undefined'));
});

test('the selected bank is found from the string a <select> holds', () => {
  const banks = [{ id: 3, name: 'a' }, { id: 12, name: 'b' }];
  assert.equal(findVideoBank(banks, '12').name, 'b');
  assert.equal(findVideoBank(banks, 12).name, 'b');
  assert.equal(findVideoBank(banks, ''), null);
  assert.equal(findVideoBank(banks, '0'), null);
  assert.equal(findVideoBank(banks, '99'), null);
  assert.equal(findVideoBank(null, '3'), null);
});

test('the selection is cut at the server’s own per-request cap', async () => {
  const seen = [];
  await runVideoBankScrapeImport({
    items: items(14), destination: { name: 'Clips' },
    post: async (url, body) => {
      seen.push({ url, count: body.items.length, name: body.name, id: body.bank_id });
      return { ok: true, bank_id: 12, created: !body.bank_id, saved: body.items.length,
        added: body.items.length, already_there: 0, skipped: {} };
    },
  });
  assert.equal(VIDEO_BANK_SCRAPE_BATCH, 6);
  assert.deepEqual(seen.map((s) => s.count), [6, 6, 2]);
  assert.ok(seen.every((s) => s.url === VIDEO_BANK_SCRAPE_ENDPOINT));
});

test('only the FIRST batch may create the bank — the rest resume into it', async () => {
  // The failure this guards: a 30-clip scrape silently becoming five banks of six.
  const seen = [];
  const res = await runVideoBankScrapeImport({
    items: items(13), destination: { name: 'Clips' },
    post: async (_url, body) => {
      seen.push(body);
      return { ok: true, bank_id: 12, created: !body.bank_id, saved: body.items.length,
        added: body.items.length, already_there: 1, skipped: { errors: 1 } };
    },
  });
  assert.equal(seen[0].name, 'Clips');
  assert.equal(seen[0].bank_id, undefined);
  assert.deepEqual(seen.slice(1).map((b) => b.bank_id), [12, 12]);
  assert.ok(seen.slice(1).every((b) => b.name === undefined));
  assert.equal(res.ok, true);
  assert.equal(res.bankId, 12);
  assert.equal(res.created, true);
  assert.equal(res.saved, 13);
  assert.equal(res.alreadyThere, 3);
  assert.deepEqual(res.skipped, { errors: 3 });
});

test('a failing batch stops the run and reports what already landed', async () => {
  let n = 0;
  const res = await runVideoBankScrapeImport({
    items: items(12), destination: { name: 'Clips' },
    post: async () => {
      n += 1;
      if (n === 1) {
        return { ok: true, bank_id: 5, created: true, saved: 6, added: 6,
          already_there: 0, skipped: {} };
      }
      return { ok: false, error: 'a detection job is already running on this bank' };
    },
  });
  assert.equal(res.ok, false);
  assert.equal(res.bankId, 5);
  assert.match(res.error, /already running/);
  assert.equal(res.saved, 6);
});

test('the batch progress callback counts batches, not items', async () => {
  const calls = [];
  await runVideoBankScrapeImport({
    items: items(7), destination: { bank_id: 3 },
    post: async () => ({ ok: true, bank_id: 3, saved: 1, added: 1, skipped: {} }),
    onBatch: (info) => calls.push(info),
  });
  assert.deepEqual(calls, [
    { index: 0, count: 2, total: 7 },
    { index: 1, count: 2, total: 7 },
  ]);
});

test('the summary says videos, and never calls identical bytes a duplicate', () => {
  assert.equal(
    summarizeVideoBankScrapeImport({ saved: 4, added: 4, alreadyThere: 0, skipped: {} }),
    '4 video(s) downloaded into the bank');
  const full = summarizeVideoBankScrapeImport({
    saved: 4, added: 3, alreadyThere: 2, skipped: { errors: 1, not_video: 2 } });
  assert.match(full, /4 video\(s\) downloaded into the bank/);
  assert.match(full, /3 inventoried/);
  assert.match(full, /2 already in the folder/);
  assert.ok(!/duplicate/i.test(full));
  assert.equal(summarizeVideoBankScrapeImport(null),
    '0 video(s) downloaded into the bank');
});

test('a file refused BY THE INTAKE is not reported as a failed download', () => {
  // The server keeps them apart: `not_video` is a GIF or an audio-only file the
  // resolver was happy to keep and the bank cannot hold — it downloaded fine.
  // Saying "could not be downloaded" would send someone to look at their
  // connection for a problem that is about the file.
  const refused = summarizeVideoBankScrapeImport({
    saved: 1, added: 1, skipped: { not_video: 3 } });
  assert.match(refused, /3 were not a video this bank can hold/);
  assert.ok(!/could not be downloaded/.test(refused));

  const network = summarizeVideoBankScrapeImport({
    saved: 1, added: 1, skipped: { errors: 2, too_large: 1 } });
  assert.match(network, /3 could not be downloaded/);
  assert.ok(!/can hold/.test(network));

  // Both at once: two counts, two sentences, no double-counting.
  const both = summarizeVideoBankScrapeImport({
    saved: 0, added: 0, skipped: { not_video: 2, errors: 1 } });
  assert.match(both, /2 were not a video this bank can hold/);
  assert.match(both, /1 could not be downloaded/);
});

test('a successful import points at the next step, which is not obvious', () => {
  // A scraped video bank holds FILES and zero shots until the passes run.
  assert.match(videoBankScrapeNextStep({ added: 3 }), /Find shots/);
  assert.equal(videoBankScrapeNextStep({ added: 0 }), '');
  assert.equal(videoBankScrapeNextStep(null), '');
});


test('the summary refuses to read a failed inventory as the perfect run', () => {
  // `added && added !== saved` short-circuits at zero — which used to be
  // exactly the case that needed a sentence.
  assert.match(
    summarizeVideoBankScrapeImport({ saved: 6, added: 0, skipped: {} }),
    /none inventoried yet — press ↻ Rescan folder/);
  assert.match(
    summarizeVideoBankScrapeImport({
      saved: 6, added: 0, skipped: {},
      syncError: 'the source folder is not reachable right now' }),
    /none inventoried — the source folder is not reachable right now/);
  // A clean run keeps the clean sentence.
  assert.equal(
    summarizeVideoBankScrapeImport({ saved: 2, added: 2, skipped: {} }),
    '2 video(s) downloaded into the bank');
});


test('a missing scraper install has its own sentence, apart from network failures', () => {
  const line = summarizeVideoBankScrapeImport({
    saved: 1, added: 1, skipped: { no_curl: 3, errors: 2 } });
  assert.match(line, /3 need the scraper extras \(Setup → Install scraper extras\)/);
  assert.match(line, /2 could not be downloaded/);
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  downloadImagesAsFiles, filesDownloadSummary, galleryZipPlanUrl, galleryZipUrl,
  imageDownloadUrl, nameFromDisposition, planNotice, zipButtonState,
  ZIP_SCOPE_ALL, ZIP_SCOPE_SELECTION,
} from './galleryDownload.js';

/* ⬇ Taking the pictures away.

   The naming lives in the BACKEND (one implementation, tested in
   backend/tests/test_gallery_download.py — the browser only reads the name off
   Content-Disposition). What lives here, and is worth pinning, is the other
   half: WHICH images a click is about, and whether the screen says so before
   the archive arrives. A ZIP that is quietly short is the exact failure this
   project removes everywhere else, so the button's own label carries the cut. */

const RUN = { kind: 'run', recordId: 42 };
const CK = { recordId: 42, step: 2500 };

// ---- the addresses --------------------------------------------------------

test('the URLs follow the scope, and a selection rides as ids', () => {
  assert.equal(galleryZipUrl(RUN), '/api/train/run/42/images/zip');
  assert.equal(galleryZipUrl(CK), '/api/train/checkpoint/42/2500/images/zip');
  assert.equal(galleryZipPlanUrl(RUN), '/api/train/run/42/images/zip/plan');
  assert.equal(galleryZipUrl(RUN, [3, 1, 2]), '/api/train/run/42/images/zip?ids=3,1,2');
  assert.equal(imageDownloadUrl(7), '/api/train/image/7/download');
});

test('an empty selection is NOT the whole gallery', () => {
  // ids=[] must not silently degrade into "everything": the backend refuses an
  // empty selection, and that refusal is the honest answer to a click that
  // meant "these ones" when nothing was picked.
  assert.equal(galleryZipUrl(RUN, []), '/api/train/run/42/images/zip?ids=');
});

test('a target the panel has not got yet has no URL at all', () => {
  assert.equal(galleryZipUrl(null), null);
  assert.equal(galleryZipPlanUrl(undefined), null);
  assert.equal(imageDownloadUrl(null), null);
});

// ---- what the button says -------------------------------------------------

test('outside Select mode the button offers the WHOLE gallery, counted', () => {
  const s = zipButtonState({ picking: false, selectedCount: 0, totalCount: 37 });
  assert.equal(s.scope, ZIP_SCOPE_ALL);
  assert.match(s.label, /37/);
  assert.equal(s.disabled, false);
});

test('in Select mode the SAME button narrows to the picks', () => {
  // One button, two meanings, driven by the mode that is already on screen —
  // rather than two buttons that both say "download" and differ by a word.
  const s = zipButtonState({ picking: true, selectedCount: 4, totalCount: 37 });
  assert.equal(s.scope, ZIP_SCOPE_SELECTION);
  assert.match(s.label, /4/);
  assert.doesNotMatch(s.label, /37/);
});

test('Select mode with nothing picked disables rather than downloading all 37', () => {
  const s = zipButtonState({ picking: true, selectedCount: 0, totalCount: 37 });
  assert.equal(s.disabled, true);
  assert.equal(s.scope, ZIP_SCOPE_SELECTION);
});

test('an empty gallery has no download button at all', () => {
  assert.equal(zipButtonState({ picking: false, selectedCount: 0, totalCount: 0 }).shown,
    false);
});

test('THE CAP IS ON THE BUTTON, not discovered inside the archive', () => {
  const s = zipButtonState({ picking: false, selectedCount: 0, totalCount: 812, cap: 500 });
  assert.match(s.label, /500/);
  assert.match(s.title, /812/);
  assert.match(s.title, /500/);
  assert.equal(s.capped, true);
});

test('under the cap nothing is said about it', () => {
  const s = zipButtonState({ picking: false, selectedCount: 0, totalCount: 12, cap: 500 });
  assert.equal(s.capped, false);
  assert.doesNotMatch(s.title, /500/);
});

test('while it is working the button says so and cannot be fired twice', () => {
  const s = zipButtonState({ picking: false, selectedCount: 0, totalCount: 5, busy: true });
  assert.equal(s.disabled, true);
  assert.match(s.label, /…/);
});

// ---- the preflight's sentence ---------------------------------------------

test('a plan with missing files says so, and the download still happens', () => {
  const n = planNotice({ ok: true, total: 40, included: 37, missing: 3,
    truncated: false, note: '3 file(s) are no longer on disk and were left out.' });
  assert.equal(n.blocked, false);
  assert.equal(n.kind, 'warn');
  assert.match(n.text, /no longer on disk/);
});

test('a plan where nothing survives BLOCKS the download and says why', () => {
  const n = planNotice({ ok: false, total: 4, included: 0, missing: 4,
    note: 'None of these 4 images can be downloaded — 4 file(s) are no longer on disk.' });
  assert.equal(n.blocked, true);
  assert.equal(n.kind, 'error');
  assert.match(n.text, /no longer on disk/);
});

test('a clean plan is not worth a notice', () => {
  assert.equal(planNotice({ ok: true, total: 5, included: 5, missing: 0,
    truncated: false, note: 'Downloading 5 image(s).' }), null);
});

test('a truncated plan is worth a notice even with nothing missing', () => {
  const n = planNotice({ ok: true, total: 812, included: 500, missing: 0,
    truncated: true, note: 'Downloading the newest 500 of 812 (one archive holds at most 500).' });
  assert.equal(n.blocked, false);
  assert.match(n.text, /500 of 812/);
});

test('a preflight that could not be reached does not pretend it is fine', () => {
  const n = planNotice(null);
  assert.equal(n.blocked, true);
  assert.equal(n.kind, 'error');
});

// ---- the name that comes back ---------------------------------------------

/* Built in Python (one implementation — services/gallery_download.py) and only
   READ here, so what can go wrong on this side is the reading: the escaped
   header form, and a header a proxy stripped. Neither may end with the picture
   saved as "download". */

test('the plain Content-Disposition form is read', () => {
  assert.equal(
    nameFromDisposition('attachment; filename="Nova-Style_run42_step002500_1187.png"'),
    'Nova-Style_run42_step002500_1187.png');
});

test('the escaped form WINS over the ASCII fallback beside it', () => {
  // Flask sends both when the name needed escaping; matching the first
  // `filename` in the string would pick the lossy one.
  const h = 'attachment; filename="run42_step000500_9.png"; '
    + "filename*=UTF-8''run42_step000500_9%20copy.png";
  assert.equal(nameFromDisposition(h), 'run42_step000500_9 copy.png');
});

test('a missing or unparseable header falls back rather than throwing', () => {
  assert.equal(nameFromDisposition(null), 'image.png');
  assert.equal(nameFromDisposition(''), 'image.png');
  assert.equal(nameFromDisposition('attachment'), 'image.png');
  // A percent sequence that is not valid UTF-8 must not take the download down.
  assert.equal(nameFromDisposition("attachment; filename*=UTF-8''%E0%A4%A"),
    'image.png');
});

// --- ⬇ the selection as plain files -----------------------------------------

const okResponse = (name) => ({
  ok: true,
  headers: { get: () => `attachment; filename="${name}"` },
  blob: async () => ({ size: 1 }),
});

test('files download one at a time, in selection order, under server names', async () => {
  const fetched = [];
  const saved = [];
  const result = await downloadImagesAsFiles([11, 22, 33], {
    fetchImpl: async (url) => { fetched.push(url); return okResponse(`img${fetched.length}.png`); },
    saveBlob: async (_blob, name) => saved.push(name),
  });
  assert.deepEqual(fetched, ['/api/train/image/11/download',
    '/api/train/image/22/download', '/api/train/image/33/download']);
  assert.deepEqual(saved, ['img1.png', 'img2.png', 'img3.png']);
  assert.deepEqual(result, { saved: 3, skipped: 0, stopped: 0 });
});

test('a missing file SKIPS and the others still land', async () => {
  // The gallery legitimately lists rows whose file a resume or a trash sweep
  // took off the disk; one of them must not cost the user the other two.
  const saved = [];
  const result = await downloadImagesAsFiles([1, 2, 3], {
    fetchImpl: async (url) => (url.includes('/2/') ? { ok: false } : okResponse('x.png')),
    saveBlob: async () => saved.push(1),
  });
  assert.equal(saved.length, 2);
  assert.deepEqual(result, { saved: 2, skipped: 1, stopped: 0 });
});

test('a thrown fetch counts as skipped, never as a crash', async () => {
  const result = await downloadImagesAsFiles([1, 2], {
    fetchImpl: async (url) => { if (url.includes('/1/')) throw new Error('net'); return okResponse('x.png'); },
    saveBlob: async () => {},
  });
  assert.deepEqual(result, { saved: 1, skipped: 1, stopped: 0 });
});

test('leaving Select mode stops AFTER the file in flight, and says how many never started', async () => {
  let stop = false;
  const saved = [];
  const result = await downloadImagesAsFiles([1, 2, 3, 4], {
    fetchImpl: async () => okResponse('x.png'),
    saveBlob: async (_b, n) => { saved.push(n); if (saved.length === 2) stop = true; },
    shouldStop: () => stop,
  });
  assert.equal(result.saved, 2);
  assert.equal(result.stopped, 2);
});

test('the summary names every branch it can end on', () => {
  assert.equal(filesDownloadSummary({ saved: 3, skipped: 0, stopped: 0 }), '3 files saved');
  assert.equal(filesDownloadSummary({ saved: 1, skipped: 0, stopped: 0 }), '1 file saved');
  assert.equal(filesDownloadSummary({ saved: 2, skipped: 1, stopped: 0 }),
    '2 files saved, 1 skipped (file no longer on disk)');
  assert.equal(filesDownloadSummary({ saved: 2, skipped: 0, stopped: 2 }),
    '2 files saved, 2 not started');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import {
  IMPORT_FALLBACK_MAX_FILES,
  IMPORT_FALLBACK_MAX_REQUEST_BYTES,
  IMPORT_MULTIPART_HEADROOM_BYTES,
  formatMiB,
  importBatchLimits,
  importBatchProgress,
  oversizedFilesMessage,
  planImportBatches,
} from './importBatches.js';

const MiB = 1024 * 1024;
const file = (name, mib) => ({ name, size: Math.round(mib * MiB) });
const names = (batch) => batch.map((f) => f.name);

test('the limits come from the capability, with the shipped defaults as a rolling-update fallback', () => {
  assert.deepEqual(importBatchLimits(), {
    maxFiles: IMPORT_FALLBACK_MAX_FILES,
    maxBytes: IMPORT_FALLBACK_MAX_REQUEST_BYTES,
    budgetBytes: IMPORT_FALLBACK_MAX_REQUEST_BYTES - IMPORT_MULTIPART_HEADROOM_BYTES,
  });
  const l = importBatchLimits({ max_files_per_request: 5, max_request_bytes: 10 * MiB });
  assert.equal(l.maxFiles, 5);
  assert.equal(l.maxBytes, 10 * MiB);
  assert.equal(l.budgetBytes, 9 * MiB);
  // Garbage never becomes a limit of 0 or NaN.
  assert.equal(importBatchLimits({ max_files_per_request: 'x', max_request_bytes: -1 }).maxFiles, 20);
});

test('the reported case goes up in ONE request against a backend that raised its import ceiling', () => {
  // 8 × 12 = 96 MiB. The import route takes 512 MiB — a photo drop is a local
  // file copy, not a stranger's upload — so nothing is split by bytes here.
  const drop = Array.from({ length: 8 }, (_, i) => file(`body${i}.jpg`, 12));
  const { batches, oversized } = planImportBatches(drop, { max_request_bytes: 512 * MiB });
  assert.equal(oversized.length, 0);
  assert.deepEqual(batches.map((b) => b.length), [8]);
});

test('against an OLD backend that publishes no ceiling, the same drop is split rather than refused', () => {
  const drop = Array.from({ length: 8 }, (_, i) => file(`body${i}.jpg`, 12));
  const { batches, oversized } = planImportBatches(drop);
  assert.equal(oversized.length, 0);
  // 63 MiB budget: 5 × 12 = 60 fits, the 6th would make 72.
  assert.deepEqual(batches.map((b) => b.length), [5, 3]);
  assert.deepEqual(names(batches[0]), ['body0.jpg', 'body1.jpg', 'body2.jpg', 'body3.jpg', 'body4.jpg']);
});

test('the file count still closes a batch at the raised ceiling: the vision pass is the real bound', () => {
  // 30 small photos, 512 MiB of room: the split is 20 + 10, because with auto
  // head-crop each image costs a vision pass that holds ComfyUI for the batch.
  const drop = Array.from({ length: 30 }, (_, i) => file(`p${i}.jpg`, 2));
  const { batches } = planImportBatches(drop, { max_request_bytes: 512 * MiB });
  assert.deepEqual(batches.map((b) => b.length), [20, 10]);
});

test('a batch also closes on the file count, and order is preserved across batches', () => {
  const drop = Array.from({ length: 45 }, (_, i) => file(`p${i}.jpg`, 0.5));
  const { batches } = planImportBatches(drop);
  assert.deepEqual(batches.map((b) => b.length), [20, 20, 5]);
  assert.deepEqual(names(batches[2]), ['p40.jpg', 'p41.jpg', 'p42.jpg', 'p43.jpg', 'p44.jpg']);
});

test('a single file above the budget is named, never sent, and never sinks the others', () => {
  const drop = [file('ok1.jpg', 3), file('huge.png', 80), file('ok2.jpg', 3)];
  const { batches, oversized, limits } = planImportBatches(drop);
  assert.deepEqual(names(oversized), ['huge.png']);
  assert.deepEqual(batches.map(names), [['ok1.jpg', 'ok2.jpg']]);
  const msg = oversizedFilesMessage(oversized, limits);
  // "the app accepts", never "the server takes": LDS runs on the user's machine.
  assert.match(msg, /^1 file larger than the 64 MiB the app accepts at a time was not sent: huge\.png \(80 MiB\)\./);
  assert.match(msg, /Resize or re-encode/);
});

test('an empty drop plans nothing, and a plural oversized list names three and counts the rest', () => {
  assert.deepEqual(planImportBatches([]).batches, []);
  assert.deepEqual(planImportBatches(null).batches, []);
  const many = Array.from({ length: 5 }, (_, i) => file(`big${i}.png`, 70));
  const { oversized, limits } = planImportBatches(many);
  const msg = oversizedFilesMessage(oversized, limits);
  assert.match(msg, /^5 files larger/);
  assert.match(msg, /big0\.png \(70 MiB\), big1\.png \(70 MiB\), big2\.png \(70 MiB\) and 2 more\./);
  assert.equal(oversizedFilesMessage([], limits), '');
});

test('the progress line reads like the scrape import one', () => {
  assert.equal(importBatchProgress(0, 20, 57), 'Importing 1–20 of 57…');
  assert.equal(importBatchProgress(40, 20, 57), 'Importing 41–57 of 57…');
});

test('sizes are said in MiB with a decimal only when it matters', () => {
  assert.equal(formatMiB(80 * MiB), '80 MiB');
  assert.equal(formatMiB(3.26 * MiB), '3.3 MiB');
  assert.equal(formatMiB(0), '0 MiB');
});

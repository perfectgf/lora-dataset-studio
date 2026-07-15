import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildSmallImageRescuePairs,
  filterSmallImageRescueGrid,
  isSmallImageRescueRow,
  summarizeScrapeImport,
} from './smallImageRescue.js';

const source = { id: 10, filename: 'source.webp', status: 'pending', derivation_kind: 'small_image_source' };

test('scrape summary reports Klein rescues separately from skipped images', () => {
  const summary = summarizeScrapeImport({
    imported: 3,
    rescueQueued: 2,
    rescueFailed: 1,
    skipped: { duplicates: 4, low_res: 0 },
  });
  assert.match(summary.message, /2 small images queued for Klein review/);
  assert.match(summary.message, /1 Klein rescue failed/);
  assert.match(summary.message, /4 duplicates skipped/);
  assert.doesNotMatch(summary.message, /small images skipped/);
  assert.equal(summary.severity, 'warning');
});

test('pair builder exposes queued, ready and failed candidates', () => {
  const queued = { id: 11, parent_image_id: 10, filename: null, status: 'pending', derivation_kind: 'klein_small_image' };
  assert.equal(buildSmallImageRescuePairs([source, queued])[0].phase, 'queued');

  const ready = { ...queued, filename: 'klein.webp' };
  assert.equal(buildSmallImageRescuePairs([source, ready])[0].phase, 'ready');

  const failed = { ...queued, status: 'failed', fail_reason: 'ComfyUI unavailable' };
  assert.equal(buildSmallImageRescuePairs([source, failed])[0].phase, 'failed');
});

test('pair builder only resolves mutually exclusive choices', () => {
  const candidate = { id: 11, parent_image_id: 10, filename: 'klein.webp', status: 'reject', derivation_kind: 'klein_small_image' };
  const originalChoice = buildSmallImageRescuePairs([{ ...source, status: 'keep' }, candidate])[0];
  assert.deepEqual({ resolved: originalChoice.resolved, choice: originalChoice.choice },
    { resolved: true, choice: 'original' });

  const unsafeBothKept = buildSmallImageRescuePairs([
    { ...source, status: 'keep' }, { ...candidate, status: 'keep' },
  ])[0];
  assert.equal(unsafeBothKept.resolved, false);
});

test('generic grid hides unresolved pairs and only exposes a resolved winner', () => {
  const regular = { id: 1, filename: 'regular.webp', status: 'keep' };
  const candidate = { id: 11, parent_image_id: 10, filename: 'klein.webp', status: 'pending', derivation_kind: 'klein_small_image' };
  assert.deepEqual(filterSmallImageRescueGrid([regular, source, candidate]).map((image) => image.id), [1]);

  const resolved = filterSmallImageRescueGrid([
    regular,
    { ...source, status: 'reject' },
    { ...candidate, status: 'keep' },
  ]);
  assert.deepEqual(resolved.map((image) => image.id), [1, 11]);
});

test('rescue provenance rows are recognizable for generic-action guards', () => {
  assert.equal(isSmallImageRescueRow(source), true);
  assert.equal(isSmallImageRescueRow({ derivation_kind: 'klein_small_image' }), true);
  assert.equal(isSmallImageRescueRow({ source: 'generated' }), false);
});

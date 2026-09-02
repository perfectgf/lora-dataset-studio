import test from 'node:test';
import assert from 'node:assert/strict';

import {
  GALLERY_KINDS, GALLERY_PAGE_LIMIT, datasetFilterOptions, galleryEmptyMessage,
  galleryFeedUrl, galleryFiltered, galleryImproveLaunchMessage,
  gallerySummaryLine, liveQueueIds, mergeGalleryHead, mergeGalleryPage,
  queueDrained,
} from './appGallery.js';
import { galleryZipPlanUrl, galleryZipUrl } from './galleryDownload.js';

// --- the feed URL: what one click actually asks for --------------------------

test('the head of the feed carries only the limit', () => {
  assert.equal(galleryFeedUrl(), `/api/gallery/images?limit=${GALLERY_PAGE_LIMIT}`);
});

test('every filter and the cursor land in the query', () => {
  assert.equal(
    galleryFeedUrl({ datasetId: '7', kind: 'improved', liked: true },
      { beforeId: 123, limit: 30 }),
    '/api/gallery/images?limit=30&before_id=123&dataset_id=7&kind=improved&liked=1');
});

test('the All positions add no parameter — the backend must see no filter', () => {
  const url = galleryFeedUrl({ datasetId: '', kind: '', liked: false });
  assert.ok(!url.includes('dataset_id') && !url.includes('kind') && !url.includes('liked'));
});

test('the kind filter offers exactly the three backend positions', () => {
  assert.deepEqual(GALLERY_KINDS.map((k) => k.id), ['', 'renders', 'improved']);
});

// --- merging pages ------------------------------------------------------------

test('a new page appends after the feed and a duplicate id is dropped', () => {
  const page1 = [{ id: 9 }, { id: 8 }];
  const merged = mergeGalleryPage(page1, [{ id: 8 }, { id: 7 }]);
  assert.deepEqual(merged.map((i) => i.id), [9, 8, 7]);
});

test('a page with nothing new returns the SAME array — no wasted re-render', () => {
  const page1 = [{ id: 9 }];
  assert.equal(mergeGalleryPage(page1, [{ id: 9 }]), page1);
});

// --- what the screen says ------------------------------------------------------

test('the summary states the cut whenever the grid shows fewer than the scope', () => {
  assert.equal(gallerySummaryLine({ count: 3, shown: 3 }), '3 images, newest first.');
  assert.equal(gallerySummaryLine({ count: 120, shown: 60 }),
    'Showing the newest 60 of 120 — the rest load as you scroll.');
  assert.equal(gallerySummaryLine({ count: 0, shown: 0 }), '');
});

test('an empty grid names the RIGHT problem — filters or no renders at all', () => {
  assert.match(galleryEmptyMessage({}), /Nothing generated yet/);
  assert.match(galleryEmptyMessage({ kind: 'improved' }), /No image matches these filters/);
  assert.equal(galleryFiltered({}), false);
  assert.equal(galleryFiltered({ liked: true }), true);
  assert.equal(galleryFiltered({ datasetId: '3' }), true);
});

test('the dataset picker opens on All and carries every count', () => {
  const opts = datasetFilterOptions(
    [{ id: 3, name: 'Nova', count: 2 }, { id: 5, name: 'Vega', count: 1 }], 3);
  assert.deepEqual(opts, [
    { value: '', label: 'All datasets (3)' },
    { value: '3', label: 'Nova (2)' },
    { value: '5', label: 'Vega (1)' },
  ]);
});

test('the improve toast names THIS feed, not a checkpoint gallery', () => {
  const msg = galleryImproveLaunchMessage('Klein');
  assert.match(msg, /^Klein started/);
  assert.match(msg, /top of this gallery/);
  assert.match(msg, /original is left untouched/);
  assert.ok(!msg.includes('checkpoint'));
});

// --- the app-wide ZIP scope ----------------------------------------------------

test('the app scope builds the gallery ZIP URLs, selection attached', () => {
  assert.equal(galleryZipUrl({ kind: 'app' }, [4, 5]),
    '/api/gallery/images/zip?ids=4,5');
  assert.equal(galleryZipPlanUrl({ kind: 'app' }, [4, 5]),
    '/api/gallery/images/zip/plan?ids=4,5');
  // An empty selection STAYS an empty selection — the backend refuses it;
  // degrading it to "no parameter" would ask for an accidental everything.
  assert.equal(galleryZipUrl({ kind: 'app' }, []), '/api/gallery/images/zip?ids=');
});

test('the record-scoped ZIP URLs are unchanged by the new scope', () => {
  assert.equal(galleryZipUrl({ kind: 'run', recordId: 9 }),
    '/api/train/run/9/images/zip');
  assert.equal(galleryZipUrl({ recordId: 9, step: 500 }, [1]),
    '/api/train/checkpoint/9/500/images/zip?ids=1');
});

// --- the feed keeping itself current ----------------------------------------

test('a fresh head page puts what is NEW on top and keeps the rest in place', () => {
  const onScreen = [{ id: 9 }, { id: 8 }, { id: 7 }];
  const head = [{ id: 11 }, { id: 10 }, { id: 9 }, { id: 8 }];
  assert.deepEqual(mergeGalleryHead(onScreen, head).map((i) => i.id),
    [11, 10, 9, 8, 7],
    'the feed is newest-first, so an image made since the last read belongs '
    + 'above the ones already on screen — and the pages scrolled into stay put');
});

test('nothing new returns the SAME array — no re-render, no scroll jump', () => {
  const onScreen = [{ id: 3 }, { id: 2 }];
  assert.equal(mergeGalleryHead(onScreen, [{ id: 3 }, { id: 2 }]), onScreen);
  assert.equal(mergeGalleryHead(onScreen, []), onScreen);
  assert.equal(mergeGalleryHead(onScreen, null), onScreen);
});

test('a head read never REMOVES a row', () => {
  // Page 1 holds the newest 60; an older image the reader paged into is simply
  // not in it, and a background read is not the place to decide it is gone.
  const onScreen = [{ id: 5 }, { id: 4 }, { id: 1 }];
  assert.deepEqual(mergeGalleryHead(onScreen, [{ id: 6 }, { id: 5 }]).map((i) => i.id),
    [6, 5, 4, 1]);
});

test('the live job ids are read off the queue listing, junk and all', () => {
  assert.deepEqual([...liveQueueIds({ jobs: [{ job_id: 'a' }, { job_id: 'b' }] })],
    ['a', 'b']);
  for (const bad of [null, undefined, {}, { jobs: null }, { jobs: [null, {}, 5] }]) {
    assert.equal(liveQueueIds(bad).size, 0, `${JSON.stringify(bad)} must read as empty`);
  }
});

test('a job leaving the queue is the signal to re-read the feed', () => {
  const before = liveQueueIds({ jobs: [{ job_id: 'a' }, { job_id: 'b' }] });
  // One of two finished: refresh NOW, not when the whole batch is done — eight
  // images arrive one at a time and the feed would sit stale between them.
  assert.equal(queueDrained(before, liveQueueIds({ jobs: [{ job_id: 'b' }] })), true);
  assert.equal(queueDrained(before, liveQueueIds({ jobs: [] })), true);
});

test('a queue that only grew, or did not move, refreshes nothing', () => {
  const before = liveQueueIds({ jobs: [{ job_id: 'a' }] });
  assert.equal(queueDrained(before, liveQueueIds({ jobs: [{ job_id: 'a' }] })), false);
  assert.equal(queueDrained(before,
    liveQueueIds({ jobs: [{ job_id: 'a' }, { job_id: 'c' }] })), false,
  'queueing more work produced no image — a feed read there would be spent for nothing');
  assert.equal(queueDrained(new Set(), liveQueueIds({ jobs: [{ job_id: 'a' }] })), false,
    'the first reading of the page can never look like a completion');
});

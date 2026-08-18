import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  destinationAcceptsItem,
  destinationMediaKinds,
  destinationMediaLabel,
  formatScanItemDuration,
  scrapeItemLabel,
  scrapeItemMediaKind,
  scrapeTileThumbUrl,
  setAsideNotice,
  sourceModesForDestination,
  splitScanItemsForDestination,
} from './scrapeDestinationMedia.js';

const image = (url = 'https://cdn.test/a.jpg') => ({ url, type: 'image' });
const video = (url = 'https://cdn.test/a.mp4', extra = {}) =>
  ({ url, type: 'video', ...extra });

test('a video bank takes videos, the two image destinations take images', () => {
  assert.deepEqual(destinationMediaKinds('dataset'), ['image']);
  assert.deepEqual(destinationMediaKinds('bank'), ['image']);
  assert.deepEqual(destinationMediaKinds('video-bank'), ['video']);
  assert.ok(destinationAcceptsItem('video-bank', video()));
  assert.ok(!destinationAcceptsItem('video-bank', image()));
  assert.ok(destinationAcceptsItem('bank', image()));
  assert.ok(!destinationAcceptsItem('bank', video()));
});

test('an unknown destination degrades to the historical image behaviour', () => {
  // A typo must cost the old behaviour, never an empty grid with no explanation.
  assert.deepEqual(destinationMediaKinds('somewhere-else'), ['image']);
  assert.ok(destinationAcceptsItem(undefined, image()));
  assert.ok(!destinationAcceptsItem(undefined, video()));
});

test('an item with no type reads as an image, exactly like the old filter', () => {
  // The historical code was `.filter((it) => it.type === 'image')`, so a source
  // that omits `type` used to be dropped. It is now KEPT on the image lane —
  // deliberate: Pexels and the web search both send items with a type, and the
  // fallback that matters is "not explicitly a video".
  assert.equal(scrapeItemMediaKind({ url: 'x' }), 'image');
  assert.equal(scrapeItemMediaKind({ url: 'x', type: 'gif' }), 'image');
  assert.equal(scrapeItemMediaKind({ url: 'x', type: 'video' }), 'video');
  assert.equal(scrapeItemMediaKind(null), 'image');
});

test('a mixed scan is split, never silently truncated', () => {
  const items = [image('1'), video('2'), image('3'), video('4'), video('5')];
  const { accepted, setAside } = splitScanItemsForDestination('video-bank', items);
  assert.deepEqual(accepted.map((it) => it.url), ['2', '4', '5']);
  assert.deepEqual(setAside.map((it) => it.url), ['1', '3']);
  const forBank = splitScanItemsForDestination('bank', items);
  assert.equal(forBank.accepted.length, 2);
  assert.equal(forBank.setAside.length, 3);
});

test('splitting tolerates a missing or malformed item list', () => {
  assert.deepEqual(splitScanItemsForDestination('bank', null),
    { accepted: [], setAside: [] });
  assert.deepEqual(splitScanItemsForDestination('bank', undefined),
    { accepted: [], setAside: [] });
});

test('the set-aside line names both sides and gets its plurals right', () => {
  assert.equal(setAsideNotice('video-bank', 0), '');
  assert.equal(setAsideNotice('video-bank', 1),
    '1 image set aside — this destination takes videos.');
  assert.equal(setAsideNotice('video-bank', 18),
    '18 images set aside — this destination takes videos.');
  assert.equal(setAsideNotice('bank', 1),
    '1 video set aside — this destination takes images.');
  assert.equal(setAsideNotice('dataset', 4),
    '4 videos set aside — this destination takes images.');
});

test('destinationMediaLabel is what the buttons and toasts say', () => {
  assert.equal(destinationMediaLabel('video-bank'), 'videos');
  assert.equal(destinationMediaLabel('video-bank', 1), 'video');
  assert.equal(destinationMediaLabel('dataset'), 'images');
});

test('a video with no poster gets NO thumbnail url instead of one that 415s', () => {
  // The proxy only restreams raster types. Pointed at an .mp4 it answers 415,
  // the tile is filed as a dead link and the clip disappears from the picker —
  // and gallery-dl sources send no poster at all, so this is the common case.
  assert.equal(scrapeTileThumbUrl(video('https://cdn.test/clip.mp4')), null);
  assert.equal(
    scrapeTileThumbUrl(video('https://cdn.test/clip.mp4',
      { thumbnail: 'https://cdn.test/poster.jpg' })),
    `/api/scrape/thumb?url=${encodeURIComponent('https://cdn.test/poster.jpg')}`);
  // An image still falls back to its own url, exactly as before.
  assert.equal(scrapeTileThumbUrl(image('https://cdn.test/a.jpg')),
    `/api/scrape/thumb?url=${encodeURIComponent('https://cdn.test/a.jpg')}`);
  assert.equal(scrapeTileThumbUrl(null), null);
});

test('a thumbnail url with query params is encoded, not interpolated', () => {
  const raw = 'https://cdn.test/p.jpg?sig=a&b=c#frag';
  assert.equal(scrapeTileThumbUrl({ url: raw, type: 'image' }),
    `/api/scrape/thumb?url=${encodeURIComponent(raw)}`);
  assert.ok(!scrapeTileThumbUrl({ url: raw, type: 'image' }).includes('&b=c'));
});

test('the duration badge is m:ss, and empty when the source gave none', () => {
  assert.equal(formatScanItemDuration(0), '');
  assert.equal(formatScanItemDuration(undefined), '');
  assert.equal(formatScanItemDuration('nope'), '');
  assert.equal(formatScanItemDuration(-4), '');
  assert.equal(formatScanItemDuration(7), '0:07');
  assert.equal(formatScanItemDuration(63.4), '1:03');
  assert.equal(formatScanItemDuration(600), '10:00');
});

test('a tile always has a label, even when the source titled nothing', () => {
  assert.equal(scrapeItemLabel(video()), 'scraped video');
  assert.equal(scrapeItemLabel(image()), 'scraped image');
  assert.equal(scrapeItemLabel(video('u', { title: 'a cat' })), 'a cat');
  assert.equal(
    scrapeItemLabel({ url: 'u', type: 'image', platform: 'pexels', photographer: 'A' }),
    'Pexels photo by A');
});

// --- the panel actually uses it -------------------------------------------------
// No JSX harness in this repo, so the wiring is pinned on the source, the same
// convention as ConceptSourcesPanel.pagination.test.js.
const panel = readFileSync(
  new URL('./ConceptSourcesPanel.jsx', import.meta.url), 'utf8');

test('the panel no longer throws video items away at scan time', () => {
  assert.doesNotMatch(panel, /filter\(\(it\) => it\.type === 'image'\)/);
  assert.match(panel, /splitScanItemsForDestination\(destination, items\)/);
});

test('the grid renders what the destination accepts, and says what it set aside', () => {
  assert.match(panel, /const \{ accepted, setAside \} = splitScanItemsForDestination/);
  assert.match(panel, /setAsideNotice\(destination, setAside\.length\)/);
});

test('a tile only renders an <img> when there is a live thumbnail to render', () => {
  assert.match(panel, /const thumb = scrapeTileThumbUrl\(it\);/);
  // A dead VIDEO poster degrades to the placeholder instead of removing the
  // clip: the poster is a separate CDN asset from what the backend downloads,
  // so its 404 says nothing about the clip. `markBroken` stays the image
  // tiles' verdict, where the thumb falls back to the medium itself.
  assert.match(panel, /\{thumb && !posterBroken\.has\(it\.url\) \? \(/);
  assert.match(panel, /isVideo\s*\n?\s*\? setPosterBroken/);
});


test('a video destination only offers the URL tab — the other three are image-only by construction', () => {
  const MODES = [['reddit', 'Reddit'], ['pexels', 'Pexels'],
    ['websearch', 'Web images'], ['url', 'URL']];
  assert.deepEqual(sourceModesForDestination('video-bank', MODES),
    [['url', 'URL']]);
  // Image destinations keep every tab, and an unknown destination degrades to
  // the historical image behaviour like everywhere else in this module.
  assert.deepEqual(sourceModesForDestination('dataset', MODES), MODES);
  assert.deepEqual(sourceModesForDestination('bank', MODES), MODES);
  assert.deepEqual(sourceModesForDestination('typo', MODES), MODES);
  assert.deepEqual(sourceModesForDestination('video-bank', null), []);
});

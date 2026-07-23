import test from 'node:test';
import assert from 'node:assert/strict';
import { clearScraperScanState, isDatasetImportBlocked, isStopGenerationBlocked, loadScraperScanState, saveScraperScanState } from './scraperState.js';

function memoryStorage() {
  const values = new Map();
  return { getItem: (key) => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)), removeItem: (key) => values.delete(key) };
}

test('dataset image imports stay available during every generation engine', () => {
  assert.equal(isDatasetImportBlocked({ localBusy: false, activity: null }), false);
  for (const engine of ['klein', 'nanobanana', 'chatgpt', undefined])
    assert.equal(isDatasetImportBlocked({
      localBusy: false, activity: { kind: 'generate', engine },
    }), false);
});

test('dataset image imports still block local overlap and non-generation activity', () => {
  assert.equal(isDatasetImportBlocked({
    localBusy: true, activity: { kind: 'generate', engine: 'chatgpt' },
  }), true);
  for (const activity of [{ kind: 'caption' }, { kind: 'classify' }, { kind: 'watermark_clean' }])
    assert.equal(isDatasetImportBlocked({ localBusy: false, activity }), true);
});

// Regression guard: a generation batch publishes a 'generate' activity, which makes
// `busy` true for its whole duration. Disabling Stop on `busy` therefore killed the
// only way to end a batch — on EVERY engine, not just the API ones.
test('Stop generation stays clickable while a batch is running', () => {
  for (const engine of ['klein', 'nanobanana', 'chatgpt', undefined])
    assert.equal(isStopGenerationBlocked({
      busy: true, activity: { kind: 'generate', engine },
    }), false);
  assert.equal(isStopGenerationBlocked({ busy: true, activity: { kind: 'improve' } }), false);
});

test('Stop generation still blocks on other activity and on a stop in flight', () => {
  for (const activity of [{ kind: 'caption' }, { kind: 'recaption' },
    { kind: 'watermark_clean' }, { kind: 'analyze_faces' }, { kind: 'classify' }])
    assert.equal(isStopGenerationBlocked({ busy: true, activity }), true);
  assert.equal(isStopGenerationBlocked({
    busy: true, activity: { kind: 'generate' }, cancelling: true,
  }), true);
  assert.equal(isStopGenerationBlocked({ busy: false, activity: null }), false);
});

test('scan results and selection survive reload per dataset', () => {
  const storage = memoryStorage();
  const state = { sourceMode: 'pexels', url: 'https://example.test/gallery',
    kw: 'portrait', sub: 'photos', pexelsKeyword: 'cinematic portrait',
    pexelsLocale: 'en-US', pexelsOrientation: 'portrait',
    activeScanUrl: 'https://www.pexels.com/en-us/search/cinematic%20portrait/?orientation=portrait',
    activePlatform: 'pexels',
    items: [{ url: 'https://example.test/a.webp', title: 'A', type: 'image' }], page: 2,
    paginated: true, fullAlbums: true, rescueSmall: false,
    selected: new Set(['https://example.test/a.webp']) };
  saveScraperScanState(11, state, storage);
  const restored = loadScraperScanState(11, storage);
  assert.deepEqual(restored.items, state.items);
  assert.deepEqual([...restored.selected], [...state.selected]);
  assert.equal(restored.sourceMode, 'pexels');
  assert.equal(restored.pexelsKeyword, state.pexelsKeyword);
  assert.equal(restored.pexelsLocale, 'en-US');
  assert.equal(restored.pexelsOrientation, 'portrait');
  assert.equal(restored.activeScanUrl, state.activeScanUrl);
  assert.equal(restored.activePlatform, 'pexels');
  assert.equal(loadScraperScanState(12, storage).items.length, 0);
});

test('legacy scan cache restores its URL as the active pagination target', () => {
  const storage = memoryStorage();
  storage.setItem('lds:scraper-scan:v1:13', JSON.stringify({
    url: 'https://www.reddit.com/search/?q=portrait', kw: 'portrait',
    items: [{ url: 'https://example.test/legacy.webp', type: 'image' }], page: 1,
  }));
  const restored = loadScraperScanState(13, storage);
  assert.equal(restored.sourceMode, 'reddit');
  assert.equal(restored.activeScanUrl, 'https://www.reddit.com/search/?q=portrait');
  assert.equal(restored.activePlatform, 'reddit');
  assert.equal(restored.pexelsLocale, 'fr-FR');
  assert.equal(restored.pexelsOrientation, '');
});

test('Pexels source drafts persist before the first result exists', () => {
  const storage = memoryStorage();
  saveScraperScanState(14, {
    sourceMode: 'pexels', pexelsKeyword: 'studio portrait',
    pexelsLocale: 'en-US', pexelsOrientation: 'square',
    items: [], selected: new Set(),
  }, storage);
  const restored = loadScraperScanState(14, storage);
  assert.equal(restored.sourceMode, 'pexels');
  assert.equal(restored.pexelsKeyword, 'studio portrait');
  assert.equal(restored.pexelsLocale, 'en-US');
  assert.equal(restored.pexelsOrientation, 'square');
  assert.deepEqual(restored.items, []);
});

test('reset clears only the targeted dataset scan', () => {
  const storage = memoryStorage();
  const scan = { items: [{ url: 'https://example.test/a.webp', type: 'image' }],
    selected: new Set(['https://example.test/a.webp']) };
  saveScraperScanState(21, scan, storage); saveScraperScanState(22, scan, storage);
  clearScraperScanState(21, storage);
  assert.equal(loadScraperScanState(21, storage).items.length, 0);
  assert.equal(loadScraperScanState(22, storage).items.length, 1);
});

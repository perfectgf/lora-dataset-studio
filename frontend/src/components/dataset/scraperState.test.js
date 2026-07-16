import test from 'node:test';
import assert from 'node:assert/strict';
import { clearScraperScanState, isDatasetImportBlocked, loadScraperScanState, saveScraperScanState } from './scraperState.js';

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

test('scan results and selection survive reload per dataset', () => {
  const storage = memoryStorage();
  const state = { url: 'https://example.test/gallery', kw: 'portrait', sub: 'photos',
    items: [{ url: 'https://example.test/a.webp', title: 'A', type: 'image' }], page: 2,
    paginated: true, fullAlbums: true, rescueSmall: false,
    selected: new Set(['https://example.test/a.webp']) };
  saveScraperScanState(11, state, storage);
  const restored = loadScraperScanState(11, storage);
  assert.deepEqual(restored.items, state.items);
  assert.deepEqual([...restored.selected], [...state.selected]);
  assert.equal(loadScraperScanState(12, storage).items.length, 0);
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

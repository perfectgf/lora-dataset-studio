const STORAGE_PREFIX = 'lds:scraper-scan:v1:';

const emptyState = () => ({ url: '', kw: '', sub: '', items: [], page: 0,
  paginated: false, fullAlbums: false, rescueSmall: false, selected: new Set() });
const storageFor = (storage) => {
  if (storage !== undefined) return storage;
  try { return globalThis.localStorage || null; } catch { return null; }
};
const keyFor = (datasetId) => `${STORAGE_PREFIX}${datasetId}`;

export function isDatasetImportBlocked({ localBusy, activity }) {
  return !!localBusy || (!!activity && activity.kind !== 'generate');
}

export function loadScraperScanState(datasetId, storage) {
  const target = storageFor(storage);
  if (!datasetId || !target) return emptyState();
  try {
    const raw = JSON.parse(target.getItem(keyFor(datasetId)) || 'null');
    if (!raw || typeof raw !== 'object') return emptyState();
    const items = Array.isArray(raw.items)
      ? raw.items.filter((item) => item && typeof item.url === 'string') : [];
    const liveUrls = new Set(items.map((item) => item.url));
    const selected = new Set((Array.isArray(raw.selected) ? raw.selected : [])
      .filter((url) => typeof url === 'string' && liveUrls.has(url)));
    return { url: typeof raw.url === 'string' ? raw.url : '',
      kw: typeof raw.kw === 'string' ? raw.kw : '', sub: typeof raw.sub === 'string' ? raw.sub : '',
      items, page: Number.isInteger(raw.page) && raw.page >= 0 ? raw.page : 0,
      paginated: !!raw.paginated, fullAlbums: !!raw.fullAlbums,
      rescueSmall: !!raw.rescueSmall, selected };
  } catch { return emptyState(); }
}

export function saveScraperScanState(datasetId, state, storage) {
  const target = storageFor(storage);
  if (!datasetId || !target) return;
  const items = Array.isArray(state?.items) ? state.items : [];
  try {
    if (!items.length) { target.removeItem(keyFor(datasetId)); return; }
    target.setItem(keyFor(datasetId), JSON.stringify({
      url: state?.url || '', kw: state?.kw || '', sub: state?.sub || '', items,
      page: Number.isInteger(state?.page) ? state.page : 0,
      paginated: !!state?.paginated, fullAlbums: !!state?.fullAlbums,
      rescueSmall: !!state?.rescueSmall,
      selected: [...(state?.selected instanceof Set ? state.selected : state?.selected || [])],
    }));
  } catch { /* cache quota/private mode must never break the scraper */ }
}

export function clearScraperScanState(datasetId, storage) {
  const target = storageFor(storage);
  if (!datasetId || !target) return;
  try { target.removeItem(keyFor(datasetId)); } catch { /* visual reset still works */ }
}

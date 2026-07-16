import { normalizePexelsLocale, normalizePexelsOrientation } from './scraperSourceSearch.js';

const STORAGE_PREFIX = 'lds:scraper-scan:v1:';

const SOURCE_MODES = new Set(['reddit', 'pexels', 'url']);

const emptyState = () => ({ sourceMode: 'reddit', url: '', kw: '', sub: '',
  pexelsKeyword: '', pexelsLocale: 'fr-FR', pexelsOrientation: '',
  activeScanUrl: '', activePlatform: '', items: [], page: 0,
  paginated: false, fullAlbums: false, rescueSmall: false, selected: new Set() });
const storageFor = (storage) => {
  if (storage !== undefined) return storage;
  try { return globalThis.localStorage || null; } catch { return null; }
};
const keyFor = (datasetId) => `${STORAGE_PREFIX}${datasetId}`;

function inferPlatform(url, items) {
  const itemPlatform = items.find((item) => typeof item.platform === 'string')?.platform;
  if (itemPlatform) return itemPlatform;
  try {
    const host = new URL(url).hostname.toLowerCase();
    if (host === 'pexels.com' || host === 'www.pexels.com') return 'pexels';
    if (host === 'reddit.com' || host.endsWith('.reddit.com')) return 'reddit';
  } catch { /* legacy/empty URL */ }
  return '';
}

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
    const url = typeof raw.url === 'string' ? raw.url : '';
    const activeScanUrl = typeof raw.activeScanUrl === 'string'
      ? raw.activeScanUrl : (items.length ? url : '');
    const activePlatform = typeof raw.activePlatform === 'string'
      ? raw.activePlatform : inferPlatform(activeScanUrl, items);
    const legacyMode = (raw.kw || raw.sub)
      ? 'reddit'
      : (activePlatform === 'pexels' && /\/(?:search|chercher)\//i.test(activeScanUrl)
        ? 'pexels' : 'url');
    return { sourceMode: SOURCE_MODES.has(raw.sourceMode) ? raw.sourceMode : legacyMode,
      url,
      kw: typeof raw.kw === 'string' ? raw.kw : '', sub: typeof raw.sub === 'string' ? raw.sub : '',
      pexelsKeyword: typeof raw.pexelsKeyword === 'string' ? raw.pexelsKeyword : '',
      pexelsLocale: normalizePexelsLocale(raw.pexelsLocale),
      pexelsOrientation: normalizePexelsOrientation(raw.pexelsOrientation),
      activeScanUrl, activePlatform,
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
    const saved = {
      sourceMode: SOURCE_MODES.has(state?.sourceMode) ? state.sourceMode : 'reddit',
      url: state?.url || '', kw: state?.kw || '', sub: state?.sub || '',
      pexelsKeyword: state?.pexelsKeyword || '',
      pexelsLocale: normalizePexelsLocale(state?.pexelsLocale),
      pexelsOrientation: normalizePexelsOrientation(state?.pexelsOrientation),
      activeScanUrl: state?.activeScanUrl || '', activePlatform: state?.activePlatform || '',
      items,
      page: Number.isInteger(state?.page) ? state.page : 0,
      paginated: !!state?.paginated, fullAlbums: !!state?.fullAlbums,
      rescueSmall: !!state?.rescueSmall,
      selected: [...(state?.selected instanceof Set ? state.selected : state?.selected || [])],
    };
    const hasMeaningfulState = items.length || saved.sourceMode !== 'reddit' || saved.url
      || saved.kw || saved.sub || saved.pexelsKeyword || saved.activeScanUrl
      || saved.pexelsLocale !== 'fr-FR' || saved.pexelsOrientation
      || saved.fullAlbums || saved.rescueSmall;
    if (!hasMeaningfulState) { target.removeItem(keyFor(datasetId)); return; }
    target.setItem(keyFor(datasetId), JSON.stringify(saved));
  } catch { /* cache quota/private mode must never break the scraper */ }
}

export function clearScraperScanState(datasetId, storage) {
  const target = storageFor(storage);
  if (!datasetId || !target) return;
  try { target.removeItem(keyFor(datasetId)); } catch { /* visual reset still works */ }
}

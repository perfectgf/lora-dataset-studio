/**
 * ConceptSourcesPanel — build a concept dataset from scraped images.
 *
 * A concept LoRA is built from REAL images, so instead of the face tooling
 * (reference photo, variations, face analysis) you: paste a gallery URL → scan
 * (read-only /api/scrape/scan) → pick images → import them DIRECTLY into the
 * dataset (/dataset/<id>/scrape-import). Nothing touches a shared pool.
 *
 * Guidance surfaced in the UI: aim for 20-50 varied images, keep at most ~10 per
 * gallery (one gallery ≈ one shoot). Server-side filters (dedup, min side < 768px,
 * ratio > 3:1) run at import — low-resolution images can optionally be preserved
 * as reviewable Klein rescue pairs instead of being skipped.
 */
import { useState, useCallback, useEffect } from 'react';
import { useToast } from '../common/Toast';
import { postJson } from '../../hooks/useDataset';
import { useCapabilities } from '../../context/CapabilitiesContext';
import InstallRunner from '../setup/InstallRunner';
import { clearScraperScanState, loadScraperScanState, saveScraperScanState } from './scraperState';
import { HelpBadge } from '../../help/HelpMode';
import PexelsAttribution from './PexelsAttribution';
import {
  buildPexelsSearchUrl,
  isPexelsUrl,
  loadPexelsAuthorization,
  normalizePexelsKeyword,
  resolveScanTarget,
  savePexelsAuthorization,
} from './scraperSourceSearch';
import { useI18n } from '../../i18n/I18nContext';

const thumbFor = (it) =>
  `/api/scrape/thumb?url=${encodeURIComponent(it.thumbnail || it.url)}`;

const SOURCE_GROUPS = [
  { label: 'SFW', tone: 'emerald', sources: [
    ['Reddit', 'https://www.reddit.com/'], ['Instagram', 'https://www.instagram.com/'],
    ['X / Twitter', 'https://x.com/'], ['Civitai images', 'https://civitai.com/images'],
    ['Pexels', 'https://www.pexels.com/'],
  ] },
  { label: 'NSFW', tone: 'rose', sources: [
    ['PornPics', 'https://www.pornpics.com/'], ['Sex.com', 'https://www.sex.com/'],
    ['Picazor', 'https://picazor.com/'], ['Erome', 'https://www.erome.com/'],
    ['Fapello', 'https://fapello.com/'],
  ] },
];

const SOURCE_MODES = [
  ['reddit', 'Reddit'],
  ['pexels', 'Pexels'],
  ['url', 'URL'],
];

const PLATFORM_LABELS = {
  civitai: 'Civitai', instagram: 'Instagram', pexels: 'Pexels', pornpics: 'PornPics',
  reddit: 'Reddit', sexcom: 'Sex.com', x: 'X / Twitter', generic: 'URL source',
};

const platformLabel = (platform, fallback) => PLATFORM_LABELS[platform]
  || (platform ? platform.charAt(0).toUpperCase() + platform.slice(1) : fallback);

export default function ConceptSourcesPanel({ datasetId, onImport, busy }) {
  const toast = useToast();
  const { t } = useI18n();
  const { caps, refresh } = useCapabilities();
  const [restoredScan] = useState(() => loadScraperScanState(datasetId));
  const [sourceMode, setSourceMode] = useState(restoredScan.sourceMode);
  // `url` is only the editable URL-mode draft. Pagination uses activeScanUrl.
  const [url, setUrl] = useState(restoredScan.url);
  const [kw, setKw] = useState(restoredScan.kw);       // Reddit keyword search
  const [sub, setSub] = useState(restoredScan.sub);     // optional subreddit scope
  const [pexelsKeyword, setPexelsKeyword] = useState(restoredScan.pexelsKeyword);
  const [pexelsLocale, setPexelsLocale] = useState(restoredScan.pexelsLocale);
  const [pexelsOrientation, setPexelsOrientation] = useState(restoredScan.pexelsOrientation);
  const [pexelsAuthorized, setPexelsAuthorized] = useState(() => loadPexelsAuthorization());
  const [activeScanUrl, setActiveScanUrl] = useState(restoredScan.activeScanUrl);
  const [activePlatform, setActivePlatform] = useState(restoredScan.activePlatform);
  const [items, setItems] = useState(restoredScan.items);
  const [page, setPage] = useState(restoredScan.page);
  const [paginated, setPaginated] = useState(restoredScan.paginated);
  const [scanning, setScanning] = useState(false);
  // Gallery-listing scans (PornPics category/tag/search): OFF = one cover per
  // matched gallery (the keyword-relevant shot), ON = every photo of each gallery.
  const [fullAlbums, setFullAlbums] = useState(restoredScan.fullAlbums);
  // Generative rescue is deliberately opt-in for every import. The source and
  // Klein result both stay out of training until the side-by-side review.
  const [rescueSmall, setRescueSmall] = useState(restoredScan.rescueSmall);
  const [selected, setSelected] = useState(() => new Set(restoredScan.selected));
  const [importing, setImporting] = useState(false);
  // URLs whose thumbnail failed to load (dead/expired source links). Hidden from
  // the grid so you only ever see & pick live images — dead galleries are common.
  const [broken, setBroken] = useState(() => new Set());
  // Preview tile size (px). A category scrape returns whole galleries (many
  // off-concept frames) → larger previews speed up eyeballing. Persisted.
  const [tile, setTile] = useState(() => {
    const v = Number(localStorage.getItem('conceptTileSize'));
    return v >= 72 && v <= 320 ? v : 120;
  });
  const changeTile = (v) => {
    setTile(v);
    try { localStorage.setItem('conceptTileSize', String(v)); } catch { /* ignore */ }
  };

  useEffect(() => {
    saveScraperScanState(datasetId, { sourceMode, url, kw, sub, pexelsKeyword,
      pexelsLocale, pexelsOrientation, activeScanUrl, activePlatform,
      items, page, paginated, fullAlbums, rescueSmall, selected });
  }, [datasetId, sourceMode, url, kw, sub, pexelsKeyword, pexelsLocale,
    pexelsOrientation, activeScanUrl, activePlatform, items, page, paginated,
    fullAlbums, rescueSmall, selected]);

  // Page zero uses the submitted form target. Later pages are deliberately pinned
  // to the last successful target, even if another tab/draft has since been edited.
  const runScan = useCallback(async (nextPage, explicitUrl) => {
    const target = resolveScanTarget({ nextPage, explicitUrl,
      draftUrl: url, activeScanUrl });
    if (!target || scanning) return;
    if (isPexelsUrl(target) && !pexelsAuthorized) {
      setSourceMode('pexels');
      toast.error(t('workspace.scrape.pexels.authError'));
      return;
    }
    setScanning(true);
    try {
      const body = await postJson('/api/scrape/scan',
        { url: target, page: nextPage, include_albums: fullAlbums });
      if (!body || !body.scannable) { toast.error((body && body.error) || t('workspace.scrape.toast.scanFailed')); return; }
      // Images only (the dataset import rejects video/gif anyway).
      const imgs = (body.items || []).filter((it) => it.type === 'image');
      const responsePage = body.page;
      const isFreshScan = responsePage === 0;
      setItems((prev) => {
        if (isFreshScan) return imgs;
        const seenUrls = new Set(prev.map((it) => it.url));
        const additions = imgs.filter((it) => {
          if (!it.url || seenUrls.has(it.url)) return false;
          seenUrls.add(it.url);
          return true;
        });
        return [...prev, ...additions];
      });
      setPaginated(!!body.paginated);
      setPage(responsePage);
      if (isFreshScan) {
        setActiveScanUrl(target);
        setActivePlatform(typeof body.platform === 'string' ? body.platform : '');
        setSelected(new Set()); setBroken(new Set());
      }  // fresh scan resets; "Load more" keeps it
      if (imgs.length === 0 && isFreshScan) toast.info(t('workspace.scrape.toast.noImages'));
    } finally {
      setScanning(false);
    }
  }, [url, activeScanUrl, scanning, fullAlbums, pexelsAuthorized, toast, t]);

  // Reddit search, three modes depending on which field is filled:
  //   keyword only          → search all of Reddit for the term
  //   keyword + subreddit    → search that term WITHIN one community (cleaner)
  //   subreddit only         → browse that community's top posts (no term)
  // All build a reddit URL routed through the same scan pipeline; the backend
  // RedditSource enumerates via the authenticated OAuth API (anon browsing is walled).
  const runRedditSearch = useCallback(() => {
    if (scanning) return;
    const q = kw.trim();
    const s = sub.trim().replace(/^\/?(r\/)?/i, '').replace(/[^A-Za-z0-9_]/g, '');
    if (!q && !s) return;
    let built;
    if (q) {
      const p = new URLSearchParams({ q, sort: 'top', t: 'all', type: 'link' });
      built = s
        ? `https://www.reddit.com/r/${s}/search/?${p.toString()}&restrict_sr=1`
        : `https://www.reddit.com/search/?${p.toString()}`;
    } else {
      built = `https://www.reddit.com/r/${s}/top/?t=all`;   // subreddit only → browse top
    }
    runScan(0, built);
  }, [kw, sub, scanning, runScan]);

  const runPexelsSearch = useCallback(() => {
    if (scanning) return;
    if (!pexelsAuthorized) { toast.error(t('workspace.scrape.pexels.authError')); return; }
    const built = buildPexelsSearchUrl(
      pexelsKeyword, pexelsLocale, pexelsOrientation);
    if (!built) return;
    runScan(0, built);
  }, [pexelsKeyword, pexelsLocale, pexelsOrientation,
    pexelsAuthorized, scanning, runScan, toast, t]);

  const changePexelsAuthorization = (confirmed) => {
    setPexelsAuthorized(confirmed);
    savePexelsAuthorization(confirmed);
  };

  const toggle = (u) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(u)) next.delete(u); else next.add(u);
      return next;
    });
  };

  // Thumbnail failed → the source image is dead/expired. Hide it and un-select it.
  const markBroken = (u) => {
    setBroken((prev) => new Set(prev).add(u));
    setSelected((prev) => {
      if (!prev.has(u)) return prev;
      const next = new Set(prev); next.delete(u); return next;
    });
  };

  const handleImport = async () => {
    const chosen = items.filter((it) => selected.has(it.url))
      .map((it) => ({
        url: it.url,
        title: it.title || '',
        ...(it.platform === 'pexels' ? {
          platform: 'pexels',
          source_url: it.source_url,
          photographer: it.photographer,
          photographer_url: it.photographer_url,
        } : {}),
      }));
    if (chosen.length === 0 || importing) return;
    setImporting(true);
    try {
      const d = await onImport?.(chosen, { rescueSmall });
      if (d?.ok) setSelected(new Set());
    } finally {
      setImporting(false);
    }
  };

  const resetScan = () => {
    clearScraperScanState(datasetId);
    setSourceMode('reddit'); setUrl(''); setKw(''); setSub('');
    setPexelsKeyword(''); setPexelsLocale('fr-FR'); setPexelsOrientation('');
    setActiveScanUrl(''); setActivePlatform(''); setItems([]); setPage(0);
    setPaginated(false); setFullAlbums(false); setRescueSmall(false);
    setSelected(new Set()); setBroken(new Set());
  };

  return (
    <section className="bg-surface rounded-xl border border-border p-3 flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <h2 className="text-content font-semibold text-sm">🕷️ {t('workspace.scrape.title')}</h2>
        <span className="text-content-subtle text-[0.6875rem]"
          title={t('workspace.scrape.targetTitle')}>
          {t('workspace.scrape.target')}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2" aria-label={t('workspace.scrape.compatibleSites')}>
        {SOURCE_GROUPS.map((group) => (
          <div key={group.label}
            className={`rounded-lg border px-2.5 py-2 ${group.tone === 'emerald'
              ? 'border-emerald-400/30 bg-emerald-500/5'
              : 'border-rose-400/30 bg-rose-500/5'}`}>
            <span className={`mr-2 text-[0.6875rem] font-bold ${group.tone === 'emerald'
              ? 'text-emerald-300' : 'text-rose-300'}`}>{group.label}</span>
            <span className="inline-flex flex-wrap gap-x-2 gap-y-1">
              {group.sources.map(([name, href]) => (
                <a key={name} href={href} target="_blank" rel="noreferrer"
                  className="text-[0.6875rem] text-content-muted underline decoration-white/20 underline-offset-2 hover:text-content">
                  {name} ↗
                </a>
              ))}
            </span>
          </div>
        ))}
      </div>

      {/* Scrape extras (curl_cffi, gallery-dl, cloudscraper…) live in the
          optional requirements-scrape.txt. Pexels enumeration uses its official
          API, while thumbnail proxying and imports still need curl_cffi. */}
      {caps.scrape_deps === false && (
        <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 p-2 flex flex-col gap-1.5">
          <p className="text-amber-200 text-[0.6875rem]">
            {t('workspace.scrape.dependencies.missing')}
          </p>
          <InstallRunner action="scrape_extras" buttonLabel={`⬇ ${t('workspace.scrape.dependencies.install')}`}
            onDone={() => refresh(true)} />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <div role="group" aria-label={t('workspace.scrape.source')}
          className="inline-flex rounded-lg border border-border bg-surface-raised p-0.5">
          {SOURCE_MODES.map(([mode, label]) => (
            <button key={mode} type="button"
              aria-pressed={sourceMode === mode}
              onClick={() => setSourceMode(mode)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 ${
                sourceMode === mode
                  ? 'bg-indigo-500 text-white shadow-sm'
                  : 'text-content-muted hover:bg-white/5 hover:text-content'}`}>
              {label}
            </button>
          ))}
        </div>
        <button type="button" onClick={handleImport} disabled={busy || importing || selected.size === 0}
          className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {importing ? t('workspace.scrape.importing') : `⬇ ${t('workspace.scrape.import')} ${selected.size || ''}`}
        </button>
      </div>

      {sourceMode === 'reddit' && (
        <div className="rounded-lg border border-border bg-white/5 px-2 py-2 flex flex-col gap-1.5">
          <span className="text-content-subtle text-[0.6875rem] flex items-center gap-1">
            <span aria-hidden>🔎</span> {t('workspace.scrape.reddit.title')}
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={kw} onChange={(e) => setKw(e.target.value)}
              aria-label={t('workspace.scrape.reddit.keywordLabel')}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runRedditSearch(); } }}
              placeholder={t('workspace.scrape.reddit.keywordPlaceholder')}
              title={t('workspace.scrape.reddit.keywordTitle')}
              className="flex-[2] min-w-[9rem] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
            />
            <span className="text-content-subtle text-sm shrink-0">{t('workspace.scrape.reddit.inCommunity')} r/</span>
            <input
              value={sub} onChange={(e) => setSub(e.target.value)}
              aria-label={t('workspace.scrape.reddit.communityLabel')}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runRedditSearch(); } }}
              placeholder={t('workspace.scrape.reddit.communityPlaceholder')}
              title={t('workspace.scrape.reddit.communityTitle')}
              className="flex-[1] min-w-[7rem] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
            />
            <button type="button" onClick={runRedditSearch}
              disabled={scanning || (!kw.trim() && !sub.trim())}
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40 shrink-0">
              {scanning ? t('workspace.scrape.searching') : t('workspace.scrape.reddit.search')}
            </button>
          </div>
          <p className="text-content-muted text-[0.6875rem] leading-relaxed">
            {t('workspace.scrape.reddit.help')}
          </p>
        </div>
      )}

      {sourceMode === 'pexels' && (
        <div className="rounded-lg border border-border bg-white/5 px-2.5 py-2 flex flex-col gap-2">
          <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 p-2 text-[0.6875rem] leading-relaxed text-amber-100">
            <p>
              <b>{t('workspace.scrape.pexels.authorizationTitle')}</b>{' '}
              {t('workspace.scrape.pexels.authorizationBody')}{' '}
              <a href="https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions"
                target="_blank" rel="noreferrer"
                className="font-semibold underline underline-offset-2 hover:text-white">
                {t('workspace.scrape.pexels.terms')} ↗
              </a>
            </p>
            <label className="mt-1.5 flex cursor-pointer items-start gap-2 font-semibold text-amber-50">
              <input type="checkbox" checked={pexelsAuthorized}
                onChange={(e) => changePexelsAuthorization(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 rounded border-amber-300 accent-amber-500" />
              <span>{t('workspace.scrape.pexels.confirm')}</span>
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input value={pexelsKeyword} onChange={(e) => setPexelsKeyword(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runPexelsSearch(); } }}
              placeholder={t('workspace.scrape.pexels.keywordPlaceholder')}
              aria-label={t('workspace.scrape.pexels.keywordLabel')}
              className="min-w-[12rem] flex-[2] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none" />
            <select value={pexelsLocale} onChange={(e) => setPexelsLocale(e.target.value)}
              aria-label={t('workspace.scrape.pexels.language')}
              className="px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm focus:border-indigo-500 outline-none">
              <option value="fr-FR">Français (fr-FR)</option>
              <option value="en-US">English (en-US)</option>
            </select>
            <select value={pexelsOrientation} onChange={(e) => setPexelsOrientation(e.target.value)}
              aria-label={t('workspace.scrape.pexels.orientation')}
              className="px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm focus:border-indigo-500 outline-none">
              <option value="">{t('workspace.scrape.pexels.anyOrientation')}</option>
              <option value="portrait">{t('workspace.scrape.pexels.portrait')}</option>
              <option value="landscape">{t('workspace.scrape.pexels.landscape')}</option>
              <option value="square">{t('workspace.scrape.pexels.square')}</option>
            </select>
            <button type="button" onClick={runPexelsSearch}
              disabled={scanning || !pexelsAuthorized || !normalizePexelsKeyword(pexelsKeyword)}
              title={pexelsAuthorized
                ? t('workspace.scrape.pexels.searchTitle')
                : t('workspace.scrape.pexels.confirmFirst')}
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40 shrink-0">
              {scanning ? t('workspace.scrape.searching') : t('workspace.scrape.pexels.search')}
            </button>
          </div>
        </div>
      )}

      {sourceMode === 'url' && (
        <div className="rounded-lg border border-border bg-white/5 px-2 py-2 flex flex-col gap-1.5">
          <form className="flex flex-wrap gap-2"
            onSubmit={(e) => { e.preventDefault(); runScan(0); }}>
            <input type="url" value={url} onChange={(e) => setUrl(e.target.value)}
              aria-label={t('workspace.scrape.url.label')}
              placeholder={t('workspace.scrape.url.placeholder')}
              className="flex-1 min-w-[14rem] px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none" />
            <button type="submit" disabled={scanning || !url.trim()}
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40">
              {scanning ? t('workspace.scrape.url.scanning') : t('workspace.scrape.url.scan')}
            </button>
            <HelpBadge topic="action-scrape-scan" className="self-center" />
          </form>
          <p className="text-content-muted text-[0.6875rem] leading-relaxed">
            {t('workspace.scrape.url.help')}
          </p>
          {/pornpics\.com/i.test(url) && !/\/galleries\//i.test(url) && (
            <label className="flex items-center gap-2 text-[0.6875rem] text-content-muted cursor-pointer"
              title={t('workspace.scrape.url.fullAlbumsTitle')}>
              <input type="checkbox" checked={fullAlbums}
                onChange={(e) => setFullAlbums(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-border-strong accent-indigo-500" />
              {t('workspace.scrape.url.fullAlbums')}
            </label>
          )}
        </div>
      )}

      <label className={`flex items-start gap-2 rounded-lg border px-2.5 py-2 text-[0.75rem] ${
        rescueSmall
          ? 'border-indigo-400/50 bg-indigo-500/10 text-content'
          : 'border-border bg-white/[0.03] text-content-muted'} ${
        caps.engines?.klein === false ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}>
        <input type="checkbox" checked={rescueSmall}
          disabled={busy || importing || caps.engines?.klein === false}
          onChange={(e) => setRescueSmall(e.target.checked)}
          className="mt-0.5 h-4 w-4 shrink-0 rounded border-border-strong accent-indigo-500" />
        <span className="flex min-w-0 flex-col gap-0.5">
          <span className="font-semibold text-content">
            {t('workspace.scrape.rescue.title')}
          </span>
          <span className="text-[0.6875rem] leading-relaxed text-content-subtle">
            {t('workspace.scrape.rescue.description')}
            {caps.engines?.klein === false ? ` ${t('workspace.scrape.rescue.notReady')}` : ''}
          </span>
        </span>
      </label>

      {items.length > 0 && (() => {
        // Only live thumbnails are shown/pickable; dead source links are hidden.
        const liveItems = items.filter((it) => !broken.has(it.url));
        const deadCount = items.length - liveItems.length;
        const hasPexels = items.some((it) => it.platform === 'pexels');
        return (
        <>
          <div className="flex items-center gap-2 text-[0.6875rem] text-content-subtle flex-wrap">
            <span className="rounded-full border border-border bg-surface-raised px-2 py-0.5 font-semibold text-content"
              title={activeScanUrl || undefined}>
              {t('workspace.scrape.results.from', {
                source: platformLabel(activePlatform, t('workspace.scrape.results.source')),
              })}
            </span>
            <button type="button" onClick={() => setSelected(new Set(liveItems.map((it) => it.url)))}
              title={t('workspace.scrape.results.selectAllTitle')}
              className="px-2 py-0.5 rounded border border-border hover:text-content">
              {t('workspace.scrape.results.selectAll', { count: liveItems.length })}
            </button>
            {deadCount > 0 && (
              <span title={t('workspace.scrape.results.deadTitle')}>
                🚫 {t('workspace.scrape.results.deadHidden', { count: deadCount })}
              </span>
            )}
            {selected.size > 0 && (
              <button type="button" onClick={() => setSelected(new Set())}
                className="px-2 py-0.5 rounded border border-border hover:text-content">
                {t('workspace.scrape.results.clear')}
              </button>
            )}
            <label className="flex items-center gap-1.5" title={t('workspace.scrape.results.previewTitle')}>
              <span aria-hidden>🔍</span>
              <input type="range" min="72" max="300" step="4" value={tile}
                onChange={(e) => changeTile(Number(e.target.value))}
                aria-label={t('workspace.scrape.results.previewLabel')}
                className="w-24 sm:w-32 accent-indigo-500 cursor-pointer" />
            </label>
            <button type="button" onClick={resetScan} disabled={scanning || importing}
              className="px-2 py-0.5 rounded border border-border hover:text-content disabled:opacity-40">
              {t('workspace.scrape.results.reset')}
            </button>
            <span className="ml-auto">
              {t(rescueSmall
                ? 'workspace.scrape.results.filtersReview'
                : 'workspace.scrape.results.filtersSkip')}
            </span>
          </div>

          {hasPexels && (
            <a href="https://www.pexels.com/" target="_blank" rel="noreferrer"
              title={t('workspace.scrape.results.openPexels')}
              className="self-start text-xs font-medium text-content-muted underline decoration-white/20 underline-offset-2 hover:text-content">
              {t('workspace.scrape.results.providedBy')}
            </a>
          )}

          <div className="grid gap-1.5 overflow-y-auto max-h-[34rem] pr-1"
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${tile}px, 1fr))` }}>
            {liveItems.map((it) => {
              const on = selected.has(it.url);
              const imageLabel = it.title
                || (it.platform === 'pexels' && it.photographer
                  ? t('workspace.scrape.results.pexelsPhotoBy', { photographer: it.photographer })
                  : t('workspace.scrape.results.scrapedImage'));
              return (
                <div key={it.url} className="min-w-0">
                  <button type="button" onClick={() => toggle(it.url)}
                    aria-pressed={on}
                    aria-label={`${t(on
                      ? 'workspace.scrape.results.deselect'
                      : 'workspace.scrape.results.select')} ${imageLabel}`}
                    title={imageLabel}
                    className={`relative aspect-square w-full rounded-lg overflow-hidden border-2 transition-all
                      ${on ? 'border-indigo-400' : 'border-transparent hover:border-border-strong'}`}>
                    <img src={thumbFor(it)} alt="" loading="lazy" onError={() => markBroken(it.url)}
                      className="w-full h-full object-cover" />
                    <span aria-hidden
                      className={`absolute top-1 right-1 w-4 h-4 rounded-full text-[0.625rem] leading-4 text-center font-bold
                        ${on ? 'bg-indigo-500 text-white' : 'bg-black/50 text-white/70'}`}>
                      {on ? '✓' : ''}
                    </span>
                  </button>
                  <PexelsAttribution metadata={it}
                    className="mt-1 block px-0.5 text-[0.625rem] leading-tight text-content-subtle" />
                </div>
              );
            })}
          </div>

          {paginated && (
            <button type="button" onClick={() => runScan(page + 1)} disabled={scanning}
              className="self-start px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-xs disabled:opacity-40">
              {scanning
                ? t('workspace.scrape.results.loading')
                : t('workspace.scrape.results.loadMore')}
            </button>
          )}
        </>
        );
      })()}
    </section>
  );
}

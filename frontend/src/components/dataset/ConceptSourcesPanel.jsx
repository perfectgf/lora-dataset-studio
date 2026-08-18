/**
 * ConceptSourcesPanel — build a concept dataset from scraped media.
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
 *
 * The scan has always returned VIDEO items too, and they used to be dropped on
 * the floor here. What a destination can take now lives in
 * `scrapeDestinationMedia.js`, so the 🎬 video bank sees the clips and the two
 * image destinations keep seeing exactly what they saw before.
 */
import { useState, useCallback, useEffect } from 'react';
import { useToast } from '../common/Toast';
import { postJson } from '../../hooks/useDataset';
import { useCapabilities } from '../../context/CapabilitiesContext';
import InstallRunner from '../setup/InstallRunner';
import { clearScraperScanState, loadScraperScanState, saveScraperScanState } from './scraperState';
import { HelpBadge } from '../../help/HelpMode';
import PexelsAttribution from './PexelsAttribution';
import SettingsLink from '../common/SettingsLink';
import KleinModelSetting from '../shared/KleinModelSetting';
import { localEngineUnavailableReason } from '../../utils/localEngineReason';
import { scrapeDepsBanner } from '../../utils/scrapeDeps';
import {
  buildPexelsSearchUrl,
  buildWebSearchUrl,
  isPexelsUrl,
  loadPexelsAuthorization,
  normalizePexelsKeyword,
  normalizeWebSearchKeyword,
  resolveScanTarget,
  savePexelsAuthorization,
  scrapeItemToImportPayload,
} from './scraperSourceSearch';
import {
  destinationAcceptsItem,
  destinationMediaLabel,
  formatScanItemDuration,
  scrapeItemLabel,
  scrapeItemMediaKind,
  scrapeTileThumbUrl,
  setAsideNotice,
  sourceModesForDestination,
  splitScanItemsForDestination,
} from './scrapeDestinationMedia';

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
  ['websearch', 'Web images'],
  ['url', 'URL'],
];

const PEXELS_AUTH_ERROR = 'Confirm explicit Pexels authorization for dataset/ML use before scanning Pexels.';

const PLATFORM_LABELS = {
  civitai: 'Civitai', instagram: 'Instagram', pexels: 'Pexels', pornpics: 'PornPics',
  reddit: 'Reddit', sexcom: 'Sex.com', x: 'X / Twitter', generic: 'URL source',
  websearch: 'Web images',
};

const platformLabel = (platform) => PLATFORM_LABELS[platform]
  || (platform ? platform.charAt(0).toUpperCase() + platform.slice(1) : 'source');

/**
 * `destination` picks WHO receives the picked media, and nothing else about the
 * scan changes: 'dataset' (the historical outlet — training-grade filters and the
 * Klein rescue of small images apply at import), 'bank' (the triage pile — what
 * is downloaded is stored, and the bank's own passes rule on small/duplicate) or
 * 'video-bank' (the same pile, for clips: shots are cut and judged there).
 * It also decides WHICH scanned items are pickable — see scrapeDestinationMedia.
 * `stateKey` scopes the saved scan; it stays the dataset id by default so every
 * existing localStorage entry keeps its key.
 */
export default function ConceptSourcesPanel({ datasetId, onImport, busy,
  destination = 'dataset', stateKey = undefined }) {
  const toBank = destination === 'bank' || destination === 'video-bank';
  const toVideoBank = destination === 'video-bank';
  const scanKey = stateKey === undefined ? datasetId : stateKey;
  const toast = useToast();
  const { caps, refresh } = useCapabilities();
  // Why Klein can't rescue anything here, in the SAME words every other screen
  // uses for the same gap. Gated on the EXACT condition that disables the
  // checkbox (`=== false`, not "not true"): capabilities arrive after the first
  // paint, and a reason printed under a control that is still enabled would be
  // the same kind of lie in reverse.
  const kleinReason = caps.engines?.klein === false
    ? localEngineUnavailableReason('klein', caps)
    : null;
  const [restoredScan] = useState(() => loadScraperScanState(scanKey));
  const [sourceMode, setSourceMode] = useState(restoredScan.sourceMode);
  // The tabs this destination can use at all, and the one actually in force.
  // A video bank only shows URL (the other three are image-only by
  // construction), and a stored mode that is no longer offered — the empty
  // state defaults to 'reddit' — silently coerces to the first tab shown
  // instead of rendering a form whose search can never return anything.
  const shownModes = sourceModesForDestination(destination, SOURCE_MODES);
  const modeInUse = shownModes.some(([m]) => m === sourceMode)
    ? sourceMode : shownModes[0][0];
  // `url` is only the editable URL-mode draft. Pagination uses activeScanUrl.
  const [url, setUrl] = useState(restoredScan.url);
  const [kw, setKw] = useState(restoredScan.kw);       // Reddit keyword search
  const [sub, setSub] = useState(restoredScan.sub);     // optional subreddit scope
  const [pexelsKeyword, setPexelsKeyword] = useState(restoredScan.pexelsKeyword);
  const [pexelsLocale, setPexelsLocale] = useState(restoredScan.pexelsLocale);
  const [pexelsOrientation, setPexelsOrientation] = useState(restoredScan.pexelsOrientation);
  const [pexelsAuthorized, setPexelsAuthorized] = useState(() => loadPexelsAuthorization());
  const [websearchKeyword, setWebsearchKeyword] = useState(restoredScan.websearchKeyword);
  const [websearchSafe, setWebsearchSafe] = useState(restoredScan.websearchSafe);
  const [activeScanUrl, setActiveScanUrl] = useState(restoredScan.activeScanUrl);
  const [activePlatform, setActivePlatform] = useState(restoredScan.activePlatform);
  const [items, setItems] = useState(restoredScan.items);
  const [page, setPage] = useState(restoredScan.page);
  const [paginated, setPaginated] = useState(restoredScan.paginated);
  // Time-budget truncation (backend `partial`, cf. gdl.enumerate): the listing
  // was cut short before every album/page could be explored. Deliberately NOT
  // persisted across reloads (scraperState.js) — it describes the scan that just
  // ran, not a durable property of the saved items; a stale "truncated" banner
  // surviving a page reload would be its own lie.
  const [partial, setPartial] = useState(false);
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
  // Video posters that 404ed. Kept APART from `broken`: an image tile whose
  // thumbnail is dead usually IS the dead media (the thumb falls back to the
  // item's own url), but a video's poster is a separate CDN asset from the
  // watch page / .mp4 the backend downloads — an expired poster says nothing
  // about the clip. Filing it in `broken` removed a downloadable clip from the
  // grid and silently deselected it.
  const [posterBroken, setPosterBroken] = useState(() => new Set());
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
    saveScraperScanState(scanKey, { sourceMode, url, kw, sub, pexelsKeyword,
      pexelsLocale, pexelsOrientation, websearchKeyword, websearchSafe,
      activeScanUrl, activePlatform,
      items, page, paginated, fullAlbums, rescueSmall, selected });
  }, [scanKey, sourceMode, url, kw, sub, pexelsKeyword, pexelsLocale,
    pexelsOrientation, websearchKeyword, websearchSafe, activeScanUrl,
    activePlatform, items, page, paginated,
    fullAlbums, rescueSmall, selected]);

  // Page zero uses the submitted form target. Later pages are deliberately pinned
  // to the last successful target, even if another tab/draft has since been edited.
  const runScan = useCallback(async (nextPage, explicitUrl) => {
    const target = resolveScanTarget({ nextPage, explicitUrl,
      draftUrl: url, activeScanUrl });
    if (!target || scanning) return;
    if (isPexelsUrl(target) && !pexelsAuthorized) {
      setSourceMode('pexels');
      toast.error(PEXELS_AUTH_ERROR);
      return;
    }
    setScanning(true);
    try {
      const body = await postJson('/api/scrape/scan',
        { url: target, page: nextPage, include_albums: fullAlbums });
      if (!body || !body.scannable) { toast.error((body && body.error) || 'Could not scan this URL.'); return; }
      // EVERY item is kept, images and videos alike. What this destination can
      // take is decided at render (scrapeDestinationMedia), so the count of what
      // was set aside survives "Load more" and can be stated instead of making a
      // scan that worked look like one that found nothing.
      const scanned = body.items || [];
      const responsePage = body.page;
      const isFreshScan = responsePage === 0;
      setItems((prev) => {
        if (isFreshScan) return scanned;
        const seenUrls = new Set(prev.map((it) => it.url));
        const additions = scanned.filter((it) => {
          if (!it.url || seenUrls.has(it.url)) return false;
          seenUrls.add(it.url);
          return true;
        });
        return [...prev, ...additions];
      });
      setPaginated(!!body.paginated);
      setPartial(!!body.partial);
      setPage(responsePage);
      if (isFreshScan) {
        setActiveScanUrl(target);
        setActivePlatform(typeof body.platform === 'string' ? body.platform : '');
        setSelected(new Set()); setBroken(new Set()); setPosterBroken(new Set());
        // "No images found" on a page full of clips would be a lie in a video
        // bank, so the sentence names what THIS destination was looking for.
        const { accepted } = splitScanItemsForDestination(destination, scanned);
        if (accepted.length === 0) {
          toast.info(`No ${destinationMediaLabel(destination)} found on this page.`);
        }
      }  // fresh scan resets; "Load more" keeps it
    } finally {
      setScanning(false);
    }
  }, [url, activeScanUrl, scanning, fullAlbums, pexelsAuthorized, toast, destination]);

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
    if (!pexelsAuthorized) { toast.error(PEXELS_AUTH_ERROR); return; }
    const built = buildPexelsSearchUrl(
      pexelsKeyword, pexelsLocale, pexelsOrientation);
    if (!built) return;
    runScan(0, built);
  }, [pexelsKeyword, pexelsLocale, pexelsOrientation,
    pexelsAuthorized, scanning, runScan, toast]);

  const runWebSearch = useCallback(() => {
    if (scanning) return;
    const built = buildWebSearchUrl(websearchKeyword, websearchSafe);
    if (!built) return;
    runScan(0, built);
  }, [websearchKeyword, websearchSafe, scanning, runScan]);

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
    // The destination filter is applied to the SELECTION too, not only to the
    // grid: a scan restored from localStorage can carry a pick this destination
    // no longer takes, and sending it would earn a skip nobody could explain.
    const chosen = items
      .filter((it) => selected.has(it.url) && destinationAcceptsItem(destination, it))
      .map(scrapeItemToImportPayload);
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
    clearScraperScanState(scanKey);
    setSourceMode('reddit'); setUrl(''); setKw(''); setSub('');
    setPexelsKeyword(''); setPexelsLocale('fr-FR'); setPexelsOrientation('');
    setWebsearchKeyword(''); setWebsearchSafe(false);
    setActiveScanUrl(''); setActivePlatform(''); setItems([]); setPage(0);
    setPaginated(false); setPartial(false); setFullAlbums(false); setRescueSmall(false);
    setSelected(new Set()); setBroken(new Set()); setPosterBroken(new Set());
  };

  return (
    <section className="bg-surface rounded-xl border border-border p-3 flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <h2 className="text-content font-semibold text-sm">
          {toVideoBank ? '🕷️ Scrape videos into the bank'
            : toBank ? '🕷️ Scrape into the bank' : '🕷️ Build from scraped images'}
        </h2>
        <span className="text-content-subtle text-[0.6875rem]"
          title={toVideoBank
            ? 'A video bank is the triage step for footage: bring back more than you need, then cut it into shots and keep the ones worth training on.'
            : toBank
              ? 'A bank is the triage step: bring back MORE than you need, then let the quality, duplicate and framing passes cut it down.'
              : 'Research-backed: 20-50 curated images beat hundreds of mixed ones; keep at most ~10 per gallery (one gallery ≈ one shoot).'}>
          {toVideoBank ? 'clips only — shots are cut after'
            : toBank ? 'bring back plenty — you triage after' : 'aim for 20-50 varied images'}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2" aria-label="Compatible scraper sites">
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

      {/* Scrape extras live in the optional requirements-scrape.txt. The names
          are NOT written here: the banner used to recite three of them while the
          backend probe watches seven, so an install flagged because `ddgs` or
          `yt_dlp` was absent got a warning that mentioned neither. It now quotes
          what the probe actually reported missing (caps.scrape_deps_detail). */}
      {caps.scrape_deps === false && (
        <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 p-2 flex flex-col gap-1.5">
          <p className="text-amber-200 text-[0.6875rem]">
            {scrapeDepsBanner(caps.scrape_deps_detail)} Install them for image previews,
            imports, the keyless web image search and video sources. Pexels uses its
            official API for listing, but still needs curl_cffi to fetch images.
          </p>
          <InstallRunner action="scrape_extras" buttonLabel="⬇ Install scraper extras"
            onDone={() => refresh(true)} />
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {/* Several sources only return results once their credentials are filled in
            (Reddit client id, Civitai token, Pexels key) — the picker is where that
            becomes relevant, not the Settings page the user is not on. */}
        <SettingsLink section="scraping" focus="scrape-credentials" className="order-last ml-auto">
          Source credentials
        </SettingsLink>
        <div role="group" aria-label="Scraper source"
          className="inline-flex rounded-lg border border-border bg-surface-raised p-0.5">
          {shownModes.map(([mode, label]) => (
            <button key={mode} type="button"
              aria-pressed={modeInUse === mode}
              onClick={() => setSourceMode(mode)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-300 ${
                modeInUse === mode
                  ? 'bg-indigo-500 text-white shadow-sm'
                  : 'text-content-muted hover:bg-white/5 hover:text-content'}`}>
              {label}
            </button>
          ))}
        </div>
        <button type="button" onClick={handleImport} disabled={busy || importing || selected.size === 0}
          className="ml-auto px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          {importing ? 'Importing…' : `⬇ Import ${selected.size || ''}`}
        </button>
      </div>

      {modeInUse === 'reddit' && (
        <div className="rounded-lg border border-border bg-white/5 px-2 py-2 flex flex-col gap-1.5">
          <span className="text-content-subtle text-[0.6875rem] flex items-center gap-1">
            <span aria-hidden>🔎</span> Search Reddit
          </span>
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={kw} onChange={(e) => setKw(e.target.value)}
              aria-label="Reddit search keyword"
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runRedditSearch(); } }}
              placeholder="keyword — what to look for (e.g. film portrait)"
              title="The term to search for across Reddit. Leave empty to browse a subreddit's top posts."
              className="flex-[2] min-w-[9rem] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
            />
            <span className="text-content-subtle text-sm shrink-0">in r/</span>
            <input
              value={sub} onChange={(e) => setSub(e.target.value)}
              aria-label="Subreddit (optional)"
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runRedditSearch(); } }}
              placeholder="community, e.g. analog (optional)"
              title="A subreddit name (community). Restricts the search to it — cleaner results. This is also how you reach NSFW communities."
              className="flex-[1] min-w-[7rem] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
            />
            <button type="button" onClick={runRedditSearch}
              disabled={scanning || (!kw.trim() && !sub.trim())}
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40 shrink-0">
              {scanning ? 'Searching…' : 'Search Reddit'}
            </button>
          </div>
          <p className="text-content-muted text-[0.6875rem] leading-relaxed">
            <b className="text-content-subtle">Keyword</b> searches all of Reddit for a term.
            Add a <b className="text-content-subtle">subreddit</b> (the part after
            <code className="px-1 text-content-subtle">r/</code>) to search inside one community.
            Subreddit alone browses that community&apos;s top posts.
          </p>
        </div>
      )}

      {modeInUse === 'pexels' && (
        <div className="rounded-lg border border-border bg-white/5 px-2.5 py-2 flex flex-col gap-2">
          <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 p-2 text-[0.6875rem] leading-relaxed text-amber-100">
            <p>
              <b>Pexels authorization required.</b> An API key alone does not authorize
              dataset or machine-learning use. Search only if Pexels explicitly authorized
              your use case.{' '}
              <a href="https://help.pexels.com/hc/en-us/articles/900005880463-What-are-the-Terms-and-Conditions"
                target="_blank" rel="noreferrer"
                className="font-semibold underline underline-offset-2 hover:text-white">
                Read the official terms ↗
              </a>
            </p>
            <label className="mt-1.5 flex cursor-pointer items-start gap-2 font-semibold text-amber-50">
              <input type="checkbox" checked={pexelsAuthorized}
                onChange={(e) => changePexelsAuthorization(e.target.checked)}
                className="mt-0.5 h-4 w-4 shrink-0 rounded border-amber-300 accent-amber-500" />
              <span>I confirm I have explicit Pexels authorization for dataset/ML use.</span>
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input value={pexelsKeyword} onChange={(e) => setPexelsKeyword(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runPexelsSearch(); } }}
              placeholder="keyword — e.g. cinematic portrait"
              aria-label="Pexels search keyword"
              className="min-w-[12rem] flex-[2] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none" />
            <select value={pexelsLocale} onChange={(e) => setPexelsLocale(e.target.value)}
              aria-label="Pexels search language"
              className="px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm focus:border-indigo-500 outline-none">
              <option value="fr-FR">French (fr-FR)</option>
              <option value="en-US">English (en-US)</option>
            </select>
            <select value={pexelsOrientation} onChange={(e) => setPexelsOrientation(e.target.value)}
              aria-label="Pexels image orientation"
              className="px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm focus:border-indigo-500 outline-none">
              <option value="">Any orientation</option>
              <option value="portrait">Portrait</option>
              <option value="landscape">Landscape</option>
              <option value="square">Square</option>
            </select>
            <button type="button" onClick={runPexelsSearch}
              disabled={scanning || !pexelsAuthorized || !normalizePexelsKeyword(pexelsKeyword)}
              title={pexelsAuthorized ? 'Search the official Pexels API' : 'Confirm explicit Pexels authorization first'}
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40 shrink-0">
              {scanning ? 'Searching…' : 'Search Pexels'}
            </button>
          </div>
        </div>
      )}

      {modeInUse === 'websearch' && (
        <div className="rounded-lg border border-border bg-white/5 px-2.5 py-2 flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <input value={websearchKeyword} onChange={(e) => setWebsearchKeyword(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runWebSearch(); } }}
              placeholder="keyword — e.g. curly hair portrait"
              aria-label="Web image search keyword"
              className="min-w-[12rem] flex-[2] px-2.5 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none" />
            <label className="flex items-center gap-2 text-[0.6875rem] text-content-muted cursor-pointer shrink-0">
              <input type="checkbox" checked={websearchSafe}
                onChange={(e) => setWebsearchSafe(e.target.checked)}
                className="h-4 w-4 rounded border-border accent-indigo-500" />
              <span>SafeSearch</span>
            </label>
            <button type="button" onClick={runWebSearch}
              disabled={scanning || !normalizeWebSearchKeyword(websearchKeyword)}
              title="Search images across the open web"
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40 shrink-0">
              {scanning ? 'Searching…' : 'Search the web'}
            </button>
            <HelpBadge topic="action-scrape-websearch" className="self-center" />
          </div>
          <p className="text-content-muted text-[0.6875rem] leading-relaxed">
            Searches images across the open web — no account and no API key.
            Results come from third-party sites: check the licence before using an
            image, and expect a broader mix than a curated source like Pexels.
          </p>
        </div>
      )}

      {modeInUse === 'url' && (
        <div className="rounded-lg border border-border bg-white/5 px-2 py-2 flex flex-col gap-1.5">
          <form className="flex flex-wrap gap-2"
            onSubmit={(e) => { e.preventDefault(); runScan(0); }}>
            <input type="url" value={url} onChange={(e) => setUrl(e.target.value)}
              aria-label="Gallery or media URL"
              placeholder="Gallery, album, collection or direct photo URL"
              className="flex-1 min-w-[14rem] px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none" />
            <button type="submit" disabled={scanning || !url.trim()}
              className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40">
              {scanning ? 'Scanning…' : 'Scan URL'}
            </button>
            <HelpBadge topic="action-scrape-scan" className="self-center" />
          </form>
          <p className="text-content-muted text-[0.6875rem] leading-relaxed">
            Use this for supported galleries and albums, or direct Pexels photos and collections.
            Normal Pexels keyword searches belong in the Pexels tab.
          </p>
          {/pornpics\.com/i.test(url) && !/\/galleries\//i.test(url) && (
            <label className="flex items-center gap-2 text-[0.6875rem] text-content-muted cursor-pointer"
              title="Off: one listing cover per gallery. On: every photo from each matched gallery.">
              <input type="checkbox" checked={fullAlbums}
                onChange={(e) => setFullAlbums(e.target.checked)}
                className="h-3.5 w-3.5 rounded border-border-strong accent-indigo-500" />
              Scan full albums — off = one cover per gallery, on = every photo of each
            </label>
          )}
        </div>
      )}

      {!toBank && (
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
            Rescue images under 768 px with Klein (generative)
          </span>
          <span className="text-[0.6875rem] leading-relaxed text-content-subtle">
            Off by default. Only small images are sent to Klein. The original is preserved,
            and neither version enters training until you choose one in Curation.
            {/* This used to be a bare "not ready in this setup" — a verdict with no
                cause, so the one thing the sentence existed to give (what to do
                next) was the one thing missing. The shared reason knows whether it
                is a stopped ComfyUI, a named missing weight, a broken one, or a
                widget value this install does not have. */}
            {kleinReason ? ` ${kleinReason.replace(/^⚠\s*/, '')}` : ''}
          </span>
        </span>
      </label>
      )}
      {/* A rescued image lands in the SAME dataset as the improved ones, so it runs
          on the same Klein model — said here rather than discovered from the result.
          Only once the option is on (an off checkbox launches nothing), and OUTSIDE
          the <label> above: its picker is a control of its own, and inside a label
          every click on it would toggle the checkbox. */}
      {!toBank && rescueSmall && (
        <KleinModelSetting datasetId={datasetId} className="px-2.5" />
      )}

      {items.length > 0 && (() => {
        // What this destination takes first (a video bank does not take photos),
        // then the live ones — a dead source link is hidden either way.
        const { accepted, setAside } = splitScanItemsForDestination(destination, items);
        const liveItems = accepted.filter((it) => !broken.has(it.url));
        const deadCount = accepted.length - liveItems.length;
        const hasPexels = accepted.some((it) => it.platform === 'pexels');
        const asideNote = setAsideNotice(destination, setAside.length);
        return (
        <>
          <div className="flex items-center gap-2 text-[0.6875rem] text-content-subtle flex-wrap">
            <span className="rounded-full border border-border bg-surface-raised px-2 py-0.5 font-semibold text-content"
              title={activeScanUrl || undefined}>
              Results from {platformLabel(activePlatform)}
            </span>
            <button type="button" onClick={() => setSelected(new Set(liveItems.map((it) => it.url)))}
              title={`Selects all live (loaded) ${destinationMediaLabel(destination)}`}
              className="px-2 py-0.5 rounded border border-border hover:text-content">
              Select all ({liveItems.length})
            </button>
            {deadCount > 0 && (
              <span title="Items whose source link is dead/expired — hidden from the grid.">
                🚫 {deadCount} dead hidden
              </span>
            )}
            {asideNote && (
              <span title="The scan found them; this destination is not what they are for.">
                🎞 {asideNote}
              </span>
            )}
            {selected.size > 0 && (
              <button type="button" onClick={() => setSelected(new Set())}
                className="px-2 py-0.5 rounded border border-border hover:text-content">
                Clear
              </button>
            )}
            <label className="flex items-center gap-1.5" title="Preview size — enlarge to judge images faster">
              <span aria-hidden>🔍</span>
              <input type="range" min="72" max="300" step="4" value={tile}
                onChange={(e) => changeTile(Number(e.target.value))}
                aria-label="Preview size"
                className="w-24 sm:w-32 accent-indigo-500 cursor-pointer" />
            </label>
            <button type="button" onClick={resetScan} disabled={scanning || importing}
              className="px-2 py-0.5 rounded border border-border hover:text-content disabled:opacity-40">
              Reset scan
            </button>
            {/* The honest one-liner about what the chosen destination does to the
                picked images. They differ on purpose: a dataset is trained on, a
                bank is triaged — so the bank stores what it downloads and its own
                passes (quality, duplicate groups, semantic) rule on it. */}
            <span className="ml-auto">
              {toVideoBank
                ? 'Stored as downloaded — shot cuts, motion and sharpness are the bank’s passes to judge'
                : toBank
                  ? 'Stored as downloaded — small/duplicate/framing are the bank’s passes to judge'
                  : `Filters at import: duplicates, ${rescueSmall ? 'Klein review for' : 'skip'} short side < 768px, ratio > 3:1`}
            </span>
          </div>

          {hasPexels && (
            <a href="https://www.pexels.com/" target="_blank" rel="noreferrer"
              title="Open Pexels"
              className="self-start text-xs font-medium text-content-muted underline decoration-white/20 underline-offset-2 hover:text-content">
              Photos provided by Pexels
            </a>
          )}

          <div className="grid gap-1.5 overflow-y-auto max-h-[34rem] pr-1"
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${tile}px, 1fr))` }}>
            {liveItems.map((it) => {
              const on = selected.has(it.url);
              const itemLabel = scrapeItemLabel(it);
              const isVideo = scrapeItemMediaKind(it) === 'video';
              const thumb = scrapeTileThumbUrl(it);
              const duration = formatScanItemDuration(it.duration);
              return (
                <div key={it.url} className="min-w-0">
                  <button type="button" onClick={() => toggle(it.url)}
                    aria-pressed={on}
                    aria-label={`${on ? 'Deselect' : 'Select'} ${itemLabel}`}
                    title={itemLabel}
                    className={`relative aspect-square w-full rounded-lg overflow-hidden border-2 transition-all
                      ${on ? 'border-indigo-400' : 'border-transparent hover:border-border-strong'}`}>
                    {/* A video with no poster gets a placeholder rather than an
                        <img> pointed at its own .mp4: the thumbnail proxy only
                        restreams raster types, so that request 415s, the tile is
                        filed as a dead link and a perfectly good clip vanishes.
                        gallery-dl sources send no poster at all — the common
                        case, not the edge one. And a poster that 404s degrades
                        to the same placeholder instead of `markBroken`: the
                        poster is a separate asset from what the backend
                        downloads, so its death is not the clip's. */}
                    {thumb && !posterBroken.has(it.url) ? (
                      <img src={thumb} alt="" loading="lazy"
                        onError={() => (isVideo
                          ? setPosterBroken((prev) => new Set(prev).add(it.url))
                          : markBroken(it.url))}
                        className="w-full h-full object-cover" />
                    ) : (
                      <span aria-hidden
                        className="flex h-full w-full items-center justify-center bg-black/40 text-2xl">
                        🎬
                      </span>
                    )}
                    {isVideo && (
                      <span aria-hidden
                        className="absolute bottom-1 left-1 rounded bg-black/70 px-1 py-0.5 text-[0.625rem] font-semibold leading-none text-white">
                        ▶{duration ? ` ${duration}` : ''}
                      </span>
                    )}
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

          {partial && (
            // Backend `partial` now covers at least four causes (time budget,
            // item cap, page cap, blocked/rate-limited source) — state the
            // effect, not a specific cause. The items above are valid, but the
            // listing was cut short before it could be fully explored, and it
            // can happen even when `paginated` is false (a truncated album dive
            // still looks "final" otherwise, cf. the gdl.enumerate `from_albums`
            // note). Say so in plain English.
            <p className="text-[0.6875rem] text-amber-500">
              This scan stopped before the end of the listing — some results may be missing.
            </p>
          )}

          {paginated && (
            <button type="button" onClick={() => runScan(page + 1)} disabled={scanning}
              className="self-start px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-xs disabled:opacity-40">
              {scanning ? 'Loading…' : `Load more ${destinationMediaLabel(destination)}`}
            </button>
          )}
        </>
        );
      })()}
    </section>
  );
}

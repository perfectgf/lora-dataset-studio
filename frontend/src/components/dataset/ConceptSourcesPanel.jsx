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
 * ratio > 3:1) run at import — the counts are reported in the toast.
 */
import { useState, useCallback } from 'react';
import { useToast } from '../common/Toast';
import { postJson } from '../../hooks/useDataset';
import { useCapabilities } from '../../context/CapabilitiesContext';
import InstallRunner from '../setup/InstallRunner';

const thumbFor = (it) =>
  `/api/scrape/thumb?url=${encodeURIComponent(it.thumbnail || it.url)}`;

export default function ConceptSourcesPanel({ onImport, busy }) {
  const toast = useToast();
  const { caps, refresh } = useCapabilities();
  const [url, setUrl] = useState('');
  const [kw, setKw] = useState('');       // Reddit keyword search
  const [sub, setSub] = useState('');     // optional subreddit scope
  const [items, setItems] = useState([]);
  const [page, setPage] = useState(0);
  const [paginated, setPaginated] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
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

  // `explicitUrl` lets the Reddit keyword search scan a freshly-built URL without
  // waiting for the `url` state to flush; "Load more" omits it and reuses `url`.
  const runScan = useCallback(async (nextPage, explicitUrl) => {
    const target = (explicitUrl ?? url).trim();
    if (!target || scanning) return;
    setScanning(true);
    try {
      const body = await postJson('/api/scrape/scan', { url: target, page: nextPage });
      if (!body || !body.scannable) { toast.error((body && body.error) || 'Could not scan this URL.'); return; }
      // Images only (the dataset import rejects video/gif anyway).
      const imgs = (body.items || []).filter((it) => it.type === 'image');
      setItems((prev) => (nextPage === 0 ? imgs : [...prev, ...imgs]));
      setPaginated(!!body.paginated);
      setPage(nextPage);
      if (nextPage === 0) setSelected(new Set());  // fresh scan resets selection; "Load more" keeps it
      if (imgs.length === 0 && nextPage === 0) toast.info('No images found on this page.');
    } finally {
      setScanning(false);
    }
  }, [url, scanning, toast]);

  // Reddit keyword search: build a reddit search URL (global or subreddit-scoped)
  // and route it through the same scan pipeline. Reddit locked anonymous browsing,
  // so the backend RedditSource enumerates via the authenticated OAuth API.
  const runRedditSearch = useCallback(() => {
    const q = kw.trim();
    if (!q || scanning) return;
    const s = sub.trim().replace(/^\/?(r\/)?/i, '').replace(/[^A-Za-z0-9_]/g, '');
    const p = new URLSearchParams({ q, sort: 'top', t: 'all', type: 'link' });
    const built = s
      ? `https://www.reddit.com/r/${s}/search/?${p.toString()}&restrict_sr=1`
      : `https://www.reddit.com/search/?${p.toString()}`;
    setUrl(built);
    runScan(0, built);
  }, [kw, sub, scanning, runScan]);

  const toggle = (u) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(u)) next.delete(u); else next.add(u);
      return next;
    });
  };

  const handleImport = async () => {
    const chosen = items.filter((it) => selected.has(it.url))
      .map((it) => ({ url: it.url, title: it.title || '' }));
    if (chosen.length === 0) return;
    const d = await onImport?.(chosen);
    if (d?.ok) setSelected(new Set());
  };

  return (
    <section className="bg-surface rounded-xl border border-border p-3 flex flex-col gap-2">
      <div className="flex items-center gap-2 flex-wrap">
        <h2 className="text-content font-semibold text-sm">🕷️ Build from scraped images</h2>
        <span className="text-content-subtle text-[0.6875rem]"
          title="Research-backed: 20-50 curated images beat hundreds of mixed ones; keep at most ~10 per gallery (one gallery ≈ one shoot).">
          aim for 20-50 varied images
        </span>
      </div>

      {/* Scrape extras (curl_cffi, gallery-dl, cloudscraper…) live in the
          optional requirements-scrape.txt — without them most sources fail
          ("curl_cffi non disponible"). One-click install into THIS interpreter. */}
      {caps.scrape_deps === false && (
        <div className="rounded-lg border border-amber-400/40 bg-amber-500/10 p-2 flex flex-col gap-1.5">
          <p className="text-amber-200 text-[0.6875rem]">
            ⚠ The scraper&apos;s Python packages are not installed (curl_cffi, gallery-dl,
            cloudscraper…) — most sources (Picazor included) need them.
          </p>
          <InstallRunner action="scrape_extras" buttonLabel="⬇ Install scraper extras"
            onDone={() => refresh(true)} />
        </div>
      )}

      {/* URL → scan. Chosen images are downloaded straight into THIS dataset. */}
      <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); runScan(0); }}>
        <input
          type="url" value={url} onChange={(e) => setUrl(e.target.value)}
          placeholder="Gallery URL (e.g. https://www.pornpics.com/galleries/...)"
          className="flex-1 min-w-0 px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
        />
        <button type="submit" disabled={scanning || !url.trim()}
          className="px-3 py-1.5 rounded-lg bg-surface-raised border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40">
          {scanning && page === 0 ? 'Scanning…' : 'Scan'}
        </button>
        <button type="button" onClick={handleImport} disabled={busy || selected.size === 0}
          className="px-3 py-1.5 rounded-lg bg-gradient-primary text-white text-sm font-semibold disabled:opacity-40">
          ⬇ Import {selected.size || ''}
        </button>
      </form>

      {/* Reddit keyword search — a keyword (optionally scoped to a subreddit) is
          turned into a reddit search URL and scanned through the same pipeline.
          Label sits on its own line so it can never crowd the inputs. */}
      <div className="rounded-lg border border-border bg-surface-raised/40 px-2 py-2 flex flex-col gap-1.5">
        <span className="text-content-subtle text-[0.6875rem] flex items-center gap-1.5 flex-wrap">
          <span className="flex items-center gap-1"><span aria-hidden>🔎</span> Search Reddit by keyword</span>
          <span className="text-content-subtle/70">— scope to a subreddit for cleaner, on-topic results</span>
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={kw} onChange={(e) => setKw(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runRedditSearch(); } }}
            placeholder="keyword (e.g. film portrait)"
            className="flex-[2] min-w-[9rem] px-2.5 py-1.5 rounded-lg bg-surface border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
          />
          <span className="text-content-subtle text-sm shrink-0">in r/</span>
          <input
            value={sub} onChange={(e) => setSub(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); runRedditSearch(); } }}
            placeholder="subreddit (optional)"
            className="flex-[1] min-w-[7rem] px-2.5 py-1.5 rounded-lg bg-surface border border-border text-content text-sm placeholder:text-content-subtle focus:border-indigo-500 outline-none"
          />
          <button type="button" onClick={runRedditSearch} disabled={scanning || !kw.trim()}
            className="px-3 py-1.5 rounded-lg bg-surface border border-border text-content text-sm hover:bg-white/10 disabled:opacity-40 shrink-0">
            Search
          </button>
        </div>
      </div>

      {items.length > 0 && (
        <>
          <div className="flex items-center gap-2 text-[0.6875rem] text-content-subtle flex-wrap">
            <button type="button" onClick={() => setSelected(new Set(items.map((it) => it.url)))}
              title="Selects all loaded images"
              className="px-2 py-0.5 rounded border border-border hover:text-content">
              Select all ({items.length})
            </button>
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
            <span className="ml-auto">Filters at import: duplicates, short side &lt; 768px, ratio &gt; 3:1</span>
          </div>

          <div className="grid gap-1.5 overflow-y-auto max-h-[34rem] pr-1"
            style={{ gridTemplateColumns: `repeat(auto-fill, minmax(${tile}px, 1fr))` }}>
            {items.map((it) => {
              const on = selected.has(it.url);
              return (
                <button type="button" key={it.url} onClick={() => toggle(it.url)}
                  aria-pressed={on} title={it.title || it.url}
                  className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all
                    ${on ? 'border-indigo-400' : 'border-transparent hover:border-border-strong'}`}>
                  <img src={thumbFor(it)} alt="" loading="lazy" className="w-full h-full object-cover" />
                  <span aria-hidden
                    className={`absolute top-1 right-1 w-4 h-4 rounded-full text-[0.625rem] leading-4 text-center font-bold
                      ${on ? 'bg-indigo-500 text-white' : 'bg-black/50 text-white/70'}`}>
                    {on ? '✓' : ''}
                  </span>
                </button>
              );
            })}
          </div>

          {paginated && (
            <button type="button" onClick={() => runScan(page + 1)} disabled={scanning}
              className="self-start px-3 py-1.5 rounded-lg border border-border bg-surface text-content-muted hover:text-content text-xs disabled:opacity-40">
              {scanning ? 'Loading…' : 'Load more galleries'}
            </button>
          )}
        </>
      )}
    </section>
  );
}

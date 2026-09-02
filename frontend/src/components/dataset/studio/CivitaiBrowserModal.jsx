// 🌐 Civitai top prompts — browse the most-reacted Civitai images and reuse the
// generation prompt an image was posted with, right next to the image itself.
// Not every image publishes its prompt; the backend walks the listing and, when
// a Civitai API key is configured, keeps only the prompt-bearing ones (the
// toggle below can widen that). Without a key the top images still show, with a
// banner explaining what the (free) key unlocks — the credential is the same
// one Settings → Scraping & sources already stores for the scraper.
import { useCallback, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router';
import { apiFetch } from '../../../api/fetchClient';
import { useToast } from '../../common/Toast';
import { useFocusTrap } from '../../../hooks/useFocusTrap';
import { HelpBadge } from '../../../help/HelpMode';

const PERIODS = [
  ['day', 'Today'], ['week', 'This week'], ['month', 'This month'],
  ['year', 'This year'], ['alltime', 'All time'],
];
const SORTS = [['reactions', 'Top reactions'], ['newest', 'Newest']];
// Browsing ceiling — mirrors the API's levels, mildest first.
const LEVELS = [
  ['none', 'Safe'], ['soft', '+ Soft'], ['mature', '+ Mature'], ['x', 'Everything (18+)'],
];
const WANT = 12;

const readPref = (key, fallback, allowed) => {
  try {
    const v = localStorage.getItem(key);
    return allowed.includes(v) ? v : fallback;
  } catch { return fallback; }
};

// 📝 `picks` / `onTogglePick`: the host's prompt batch. Given a handler, every
// prompt-bearing card grows a tick box — ticking adds that prompt as one more
// pass of the next run and leaves the browser OPEN, because the whole point
// is to collect several. Without a handler (the comparison surface has no
// batch) the cards keep their two verbs and nothing else appears.
export default function CivitaiBrowserModal({ open, onClose, onUse, picks = null, onTogglePick = null }) {
  const toast = useToast();
  const ref = useRef(null);
  useFocusTrap(ref, open);
  const batchable = typeof onTogglePick === 'function';
  const picked = Array.isArray(picks) ? picks : [];

  // Filters survive reopen and restart — new localStorage keys, safe defaults.
  const [period, setPeriod] = useState(() => readPref('civitaiBrowse_period', 'week', PERIODS.map(([v]) => v)));
  const [sort, setSort] = useState(() => readPref('civitaiBrowse_sort', 'reactions', SORTS.map(([v]) => v)));
  const [level, setLevel] = useState(() => readPref('civitaiBrowse_level', 'none', LEVELS.map(([v]) => v)));
  const [withPrompt, setWithPrompt] = useState(() => {
    try { return localStorage.getItem('civitaiBrowse_withPrompt') !== '0'; } catch { return true; }
  });
  useEffect(() => {
    try {
      localStorage.setItem('civitaiBrowse_period', period);
      localStorage.setItem('civitaiBrowse_sort', sort);
      localStorage.setItem('civitaiBrowse_level', level);
      localStorage.setItem('civitaiBrowse_withPrompt', withPrompt ? '1' : '0');
    } catch { /* private mode */ }
  }, [period, sort, level, withPrompt]);

  const [items, setItems] = useState([]);
  const [hasKey, setHasKey] = useState(null);       // null = not known yet
  const [keyRejected, setKeyRejected] = useState(false);
  const [exhausted, setExhausted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(() => new Set());
  // Continuation lives in a ref: `load` reads the LATEST cursor even when the
  // button that triggered it closed over an older render.
  const cont = useRef({ cursor: null, skip: 0 });

  const load = useCallback(async (reset) => {
    setLoading(true);
    setError(null);
    if (reset) cont.current = { cursor: null, skip: 0 };
    try {
      const params = new URLSearchParams({ period, sort, level, want: String(WANT) });
      if (!withPrompt) params.set('require_prompt', '0');
      if (cont.current.cursor) params.set('cursor', cont.current.cursor);
      if (cont.current.skip) params.set('skip', String(cont.current.skip));
      const d = await apiFetch(`/api/studio/civitai/images?${params.toString()}`);
      cont.current = { cursor: d.next_cursor || null, skip: d.next_skip || 0 };
      setHasKey(!!d.has_key);
      setKeyRejected(!!d.key_rejected);
      setExhausted(!!d.exhausted);
      setItems((prev) => {
        // A re-walked page may overlap what is already shown — dedup by id so
        // “Load more” only ever APPENDS new cards.
        const base = reset ? [] : prev;
        const seen = new Set(base.map((c) => c.id));
        return [...base, ...(d.items || []).filter((c) => !seen.has(c.id))];
      });
    } catch (e) {
      setError(e.message || 'Could not reach Civitai.');
    } finally {
      setLoading(false);
    }
  }, [period, sort, level, withPrompt]);

  // First open and every filter change restart the browse from the top.
  useEffect(() => { if (open) load(true); }, [open, load]);

  if (!open) return null;

  const copyPrompt = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success('Prompt copied');
    } catch {
      toast.error('Could not copy — select the text and copy it manually.');
    }
  };
  const toggleExpand = (id) => setExpanded((cur) => {
    const next = new Set(cur);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const sel = 'rounded border border-border bg-app/60 px-1.5 py-1 text-content text-[0.6875rem]';

  /* PORTAILLÉE SUR `document.body`, et ce n'est pas une préférence.
     Cette modale est montée depuis StudioRunSetup, qui vit dans l'`<aside
     lg:sticky lg:overflow-auto>` de ComparisonStudio. `position: sticky` OUVRE un
     contexte d'empilement : le `z-[9999]` ci-dessous y est plafonné et ne peut pas
     passer au-dessus de la grille de résultats, sœur de l'aside et plus loin dans
     le DOM — d'où les 👍/👎 des cellules peints PAR-DESSUS les prompts. Et
     `overflow-auto` la DÉCOUPE en prime. Le portail sort du contexte fautif ; il
     n'y a pas de z-index qui répare ça de l'intérieur.
     ⚠️ Invisible aux suites : ni un test de source ni un rendu SSR n'a de layout.
     Seule une capture tranche. Même piège, même fix que CaptionEditorDialog. */
  return createPortal(
    <div className="fixed inset-0 z-[9999] bg-black/70 flex items-center justify-center p-4"
      role="dialog" aria-modal="true" aria-label="Browse top Civitai prompts" ref={ref}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      {/* `data-probe-layer` : cette modale couvre la page PAR DESIGN — non budgétée,
          appariée avec rien dans le test de chevauchement. Sans marqueur ni état de
          sonde, la rangée d'actions d'une carte — passée de deux boutons à trois
          avec le lot — n'est mesurée à aucune taille. */}
      <div data-probe-layer data-probe-panel="civitai-browser"
        className="w-full max-w-4xl max-h-[88vh] rounded-2xl border border-border bg-surface-overlay p-4 flex flex-col gap-3 shadow-xl">
        <div className="flex items-center justify-between gap-2">
          <h2 className="text-content text-sm font-semibold flex items-center gap-1.5">
            <span aria-hidden>🌐</span> Civitai top prompts
            <HelpBadge topic="studio-civitai-browser" />
          </h2>
          <button type="button" onClick={onClose} aria-label="Close"
            className="w-8 h-8 rounded-lg border border-border bg-app text-content-muted hover:text-content">×</button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select value={period} onChange={(e) => setPeriod(e.target.value)}
            aria-label="Time period" className={sel}>
            {PERIODS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <select value={sort} onChange={(e) => setSort(e.target.value)}
            aria-label="Sort order" className={sel}>
            {SORTS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          <select value={level} onChange={(e) => setLevel(e.target.value)}
            aria-label="Content level" className={sel}>
            {LEVELS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          {hasKey !== false && (
            <label className="flex items-center gap-1.5 text-content-muted text-[0.6875rem]">
              <input type="checkbox" checked={withPrompt}
                onChange={(e) => setWithPrompt(e.target.checked)} />
              Only images with a prompt
            </label>
          )}
        </div>

        {hasKey === false && (
          <p className="m-0 rounded-lg border border-amber-400/30 bg-amber-400/5 px-3 py-2 text-[0.6875rem] leading-snug text-amber-300/90">
            Browsing works without a key, but reading the prompts needs a free
            Civitai API key.{' '}
            <Link to="/settings/scraping" onClick={onClose}
              className="underline decoration-dotted hover:text-amber-200">
              Add it in Settings → Scraping &amp; sources
            </Link>{' '}
            and reopen this browser.
          </p>
        )}
        {keyRejected && (
          <p className="m-0 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-red-300 text-[0.6875rem]" role="alert">
            Civitai refused the API key — check it in{' '}
            <Link to="/settings/scraping" onClick={onClose}
              className="underline decoration-dotted">Settings → Scraping &amp; sources</Link>.
          </p>
        )}
        {error && (
          <p className="m-0 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-red-300 text-[0.6875rem]" role="alert">
            {error}
          </p>
        )}

        <div className="min-h-0 flex-1 overflow-y-auto flex flex-col gap-2 pr-1">
          {items.map((c) => (
            <div key={c.id} className="flex gap-2.5 rounded-xl border border-border bg-surface p-2">
              <a href={c.page_url} target="_blank" rel="noreferrer" className="shrink-0"
                title="Open this image on Civitai">
                <img src={c.thumb_url || c.image_url} alt="" loading="lazy" decoding="async"
                  className="w-28 sm:w-36 h-40 sm:h-48 object-cover rounded-lg border border-border" />
              </a>
              <div className="flex flex-col gap-1.5 min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[0.625rem] text-content-subtle">
                  <span title="Reactions">❤️ {c.reactions}</span>
                  <span title="Comments">💬 {c.comments}</span>
                  {c.nsfw_level && c.nsfw_level !== 'None' && (
                    <span className="rounded bg-red-500/20 px-1 py-px text-red-300">{c.nsfw_level}</span>
                  )}
                  {c.username && <span className="truncate">by {c.username}</span>}
                </div>
                {c.prompt ? (
                  <button type="button" onClick={() => toggleExpand(c.id)}
                    title={expanded.has(c.id) ? 'Collapse the prompt' : 'Show the full prompt'}
                    className={`m-0 text-left text-content text-[0.75rem] leading-snug whitespace-pre-wrap break-words ${
                      expanded.has(c.id) ? '' : 'line-clamp-5'}`}>
                    {c.prompt}
                  </button>
                ) : (
                  <p className="m-0 text-content-subtle text-[0.75rem] italic">
                    No prompt published for this image.
                  </p>
                )}
                <div className="mt-auto flex flex-wrap items-center gap-1.5">
                  {c.model && (
                    <span className="rounded bg-app/60 border border-border px-1.5 py-0.5 text-[0.625rem] text-content-muted max-w-[10rem] truncate"
                      title={`Model: ${c.model}`}>{c.model}</span>
                  )}
                  {c.steps != null && (
                    <span className="text-[0.625rem] text-content-subtle">{c.steps} steps</span>
                  )}
                  {c.cfg != null && (
                    <span className="text-[0.625rem] text-content-subtle">CFG {c.cfg}</span>
                  )}
                  {c.prompt && (
                    <span className="ml-auto flex items-center gap-1.5">
                      {batchable && (() => {
                        const inBatch = picked.includes(c.prompt);
                        return (
                          <button type="button" role="checkbox" aria-checked={inBatch}
                            onClick={() => onTogglePick(c.prompt)}
                            data-testid="civitai-batch-toggle"
                            title={inBatch
                              ? 'Remove this prompt from the batch'
                              : 'Add this prompt to the batch — one more pass of the next run, the field stays as it is'}
                            className={`px-2 py-1 min-h-10 lg:min-h-0 rounded border text-[0.6875rem] ${inBatch
                              ? 'border-purple-400 bg-purple-500/25 text-purple-100'
                              : 'border-border bg-app text-content-muted hover:text-content'}`}>
                            {inBatch ? '☑ In batch' : '☐ Batch'}
                          </button>
                        );
                      })()}
                      <button type="button" onClick={() => copyPrompt(c.prompt)}
                        title="Copy this prompt"
                        className="px-2 py-1 min-h-10 lg:min-h-0 rounded border border-border bg-app text-content-muted text-[0.6875rem] hover:text-content">
                        📋 Copy
                      </button>
                      <button type="button" onClick={() => onUse(c.prompt)}
                        title="Use this prompt as the test prompt"
                        className="px-2.5 py-1 min-h-10 lg:min-h-0 rounded-lg bg-gradient-primary text-gray-950 text-[0.6875rem] font-semibold">
                        ⤵ Use prompt
                      </button>
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}

          {!loading && !error && items.length === 0 && (
            <p className="m-0 rounded-lg border border-border bg-surface px-3 py-6 text-center text-content-subtle text-[0.75rem]">
              Nothing to show — Civitai returned no{withPrompt && hasKey ? ' prompt-bearing' : ''} images
              for these filters. Try a longer period{withPrompt && hasKey ? ' or untick “Only images with a prompt”' : ''}.
            </p>
          )}

          {loading && (
            <p className="m-0 flex items-center justify-center gap-2 py-4 text-content-subtle text-[0.75rem]" role="status">
              <span className="inline-block w-4 h-4 border-2 border-purple-400/40 border-t-purple-400 rounded-full animate-spin" aria-hidden />
              Reading Civitai…
            </p>
          )}

          {!loading && !exhausted && items.length > 0 && (
            <button type="button" onClick={() => load(false)}
              className="self-center px-4 py-1.5 min-h-10 lg:min-h-0 rounded-lg border border-border bg-surface text-content-muted text-[0.75rem] hover:text-content">
              Load more
            </button>
          )}
        </div>

        {/* 📝 The batch's running count, pinned under the list so it is read
            without scrolling back: what the next run will replay, and the way
            out once enough is ticked. */}
        {batchable && picked.length > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-2"
            data-testid="civitai-batch-footer">
            <span className="text-[0.75rem] text-content">
              <span className="rounded bg-purple-500/20 px-1.5 py-0.5 font-semibold text-purple-200 tabular-nums">
                {picked.length} prompt{picked.length === 1 ? '' : 's'}
              </span>
              {' '}in the batch — one pass each on the next run
            </span>
            <button type="button" onClick={onClose}
              className="px-3 py-1 min-h-10 lg:min-h-0 rounded-lg bg-gradient-primary text-gray-950 text-[0.6875rem] font-semibold">
              Done
            </button>
          </div>
        )}
      </div>
    </div>,
    document.body,
  );
}

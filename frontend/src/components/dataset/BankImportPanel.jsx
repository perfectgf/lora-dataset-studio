/** 🗃 Import from a bank or a dataset — a third source next to "📥 Import real
 *  photos" and "🕸 Scrape images from the web".
 *
 *  One chooser lists the banks and the library's other datasets; the picker
 *  below shows the chosen source's images and copies only the ones clicked.
 *  A bank copies through its server-side promote path (webp normalization +
 *  perceptual dedup against this dataset) as a background job (bank_jobs),
 *  followed through the very snapshot the bank page follows — the one embedded
 *  in the banks payload. A dataset copies through /import-from-dataset, in
 *  batches counted on the button. "Import all kept" keeps the whole-bank path
 *  for a bank that is already triaged. */
import { useCallback, useEffect, useRef, useState } from 'react';
import { Archive } from 'lucide-react';
import { useNavigate } from 'react-router';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import SelectionMark from '../shared/SelectionMark';
import { imageVersionQuery } from '../bank/bankEdits.js';
import { datasetThumbUrl } from '../../utils/datasetThumbUrl.js';
import {
  bankImportOptions, banksUrl, promotableCounts, promoteUrl, promoteAllBody,
  isBankJobLive, bankActivity, promoteOutcome,
  datasetsUrl, datasetUrl, sourceValue, parseSource, datasetSourceOptions,
  bankImagesUrl, pickableDatasetImages, promoteSelectionBody,
  importFromDatasetUrl, idBatches, datasetCopyOutcome,
} from './bankImport.js';

const BANK_CURRENT_KEY = 'bankCurrentId';   // BankPage's stored "open this bank" handle

export default function BankImportPanel({ datasetId, onImported, disabled = false }) {
  const toast = useToast();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [banks, setBanks] = useState(null);
  const [datasets, setDatasets] = useState(null);
  const [counts, setCounts] = useState({});
  const [chosen, setChosen] = useState('');
  // The picker: the chosen source's images (null while they load), the kept
  // ones unless showAll, and for a bank how many match server-side (it pages).
  const [showAll, setShowAll] = useState(false);
  const [items, setItems] = useState(null);
  const [itemsTotal, setItemsTotal] = useState(0);
  const [loadingMore, setLoadingMore] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  const [starting, setStarting] = useState(false);
  // The bank whose promote job we are following; null when nothing is in flight.
  const [watching, setWatching] = useState(null);
  // { done, total } while a dataset copy runs.
  const [copying, setCopying] = useState(null);
  const wasLive = useRef(false);
  // What the picker shows right now, so a page that lands after the source or
  // the filter changed is dropped instead of appended to the wrong grid.
  const view = useRef('');

  const source = parseSource(chosen);
  const sourceKind = source?.kind;
  const sourceId = source?.id;

  // ONE request for the whole bank chooser: the counts are per-target, so they
  // ride along on /api/banks?dataset_id= instead of costing a /promotable per
  // bank. Refetched whenever the panel reopens (a triage pass changes them).
  const loadBanks = useCallback(async () => {
    try {
      const d = await apiFetch(banksUrl(datasetId));
      const rows = d.banks || [];
      setBanks(rows);
      setCounts(promotableCounts(rows));
      return rows;
    } catch (e) {
      toast.error(e?.message || 'Could not load the banks.');
      setBanks([]);
      return [];
    }
  }, [toast, datasetId]);

  const loadDatasets = useCallback(async () => {
    try {
      const d = await apiFetch(datasetsUrl);
      setDatasets(datasetSourceOptions(d.datasets, datasetId));
    } catch (e) {
      toast.error(e?.message || 'Could not load the datasets.');
      setDatasets([]);
    }
  }, [toast, datasetId]);

  useEffect(() => {
    if (!open) return;
    setCounts({});      // back to "Counting…" rather than the previous target's numbers
    loadBanks();
    loadDatasets();
  }, [open, loadBanks, loadDatasets]);

  // ---- the picker's images: first page of a bank, or a whole dataset ---------
  useEffect(() => {
    view.current = `${chosen}|${showAll}`;
    setItems(null);
    setItemsTotal(0);
    if (!sourceKind) return undefined;
    let live = true;
    const load = sourceKind === 'bank'
      ? apiFetch(bankImagesUrl(sourceId, { all: showAll }))
        .then((d) => [d.images || [], Number(d.total || 0)])
      : apiFetch(datasetUrl(sourceId)).then((d) => {
        const rows = pickableDatasetImages(d.images, { all: showAll });
        return [rows, rows.length];
      });
    load.then(([rows, total]) => {
      if (!live) return;
      setItems(rows);
      setItemsTotal(total);
    }).catch((e) => {
      if (!live) return;
      toast.error(e?.message || 'Could not load the images of this source.');
      setItems([]);
    });
    return () => { live = false; };
  }, [chosen, sourceKind, sourceId, showAll, toast]);

  // ---- follow the background job (same snapshot the bank page polls) --------
  useEffect(() => {
    if (watching == null) return undefined;
    let live = true;
    const tick = async () => {
      let rows;
      try {
        // Same URL as the initial load: the poll refreshes the counts for free,
        // so the chooser is already right the moment the job reports done.
        const d = await apiFetch(banksUrl(datasetId));
        rows = d.banks || [];
      } catch {
        return;   // transient — the next tick retries
      }
      if (!live) return;
      setBanks(rows);
      setCounts(promotableCounts(rows));
      const act = bankActivity(rows, watching);
      if (isBankJobLive(act)) { wasLive.current = true; return; }
      // Gone or finished: the job is over. Report it with the SERVER's own words.
      if (wasLive.current || act) {
        wasLive.current = false;
        const outcome = promoteOutcome(act);
        if (outcome) toast[outcome.kind]?.(outcome.text);
        onImported?.();
      }
      setWatching(null);
    };
    const t = setInterval(tick, 2000);
    tick();
    return () => { live = false; clearInterval(t); };
  }, [watching, toast, onImported, datasetId]);

  const bankRows = bankImportOptions(banks, counts);
  const currentBank = sourceKind === 'bank'
    ? bankRows.find((r) => String(r.id) === String(sourceId)) || null
    : null;
  const busy = starting || watching != null || copying != null;

  const pickSource = (value) => {
    setChosen(value);
    setSelected(new Set());
  };

  const toggle = (id) => setSelected((prev) => {
    const next = new Set(prev);
    if (next.has(id)) next.delete(id); else next.add(id);
    return next;
  });

  const selectShown = () => setSelected((prev) => new Set([...prev, ...(items || []).map((img) => img.id)]));

  const loadMore = async () => {
    if (sourceKind !== 'bank' || loadingMore || !items) return;
    const asked = `${chosen}|${showAll}`;
    setLoadingMore(true);
    try {
      const d = await apiFetch(bankImagesUrl(sourceId, { all: showAll, offset: items.length }));
      if (view.current !== asked) return;
      setItems((prev) => [...(prev || []), ...(d.images || [])]);
      setItemsTotal(Number(d.total || 0));
    } catch (e) {
      toast.error(e?.message || 'Could not load more images.');
    } finally {
      setLoadingMore(false);
    }
  };

  const thumbOf = (img) => (sourceKind === 'bank'
    ? `/api/bank/${sourceId}/thumb/${img.id}${imageVersionQuery(img)}`
    : datasetThumbUrl(`/api/dataset/${sourceId}/img/${encodeURIComponent(img.filename)}`, 192));

  // The whole kept set of a bank, as before the picker existed.
  const startAll = async () => {
    if (busy || !currentBank?.ready) return;
    setStarting(true);
    try {
      await postJson(promoteUrl(currentBank.id), promoteAllBody(datasetId));
      toast.success('Import started — the images arrive as they are copied.');
      wasLive.current = false;
      setWatching(currentBank.id);
    } catch (e) {
      // postJson THROWS on 400/409 (409 = another job already runs on that bank);
      // without this the click would look like it did nothing.
      toast.error(e?.message || 'Could not start the import from this bank.');
    } finally {
      setStarting(false);
    }
  };

  const importSelected = async () => {
    if (busy || !source || selected.size === 0) return;
    const ids = [...selected];
    if (source.kind === 'bank') {
      setStarting(true);
      try {
        await postJson(promoteUrl(source.id), promoteSelectionBody(datasetId, ids));
        toast.success('Import started — the images arrive as they are copied.');
        wasLive.current = false;
        setWatching(source.id);
        setSelected(new Set());
      } catch (e) {
        toast.error(e?.message || 'Could not start the import from this bank.');
      } finally {
        setStarting(false);
      }
      return;
    }
    const totals = { imported: 0, duplicates: 0, failed: 0, small: 0 };
    setCopying({ done: 0, total: ids.length });
    try {
      for (const batch of idBatches(ids)) {
        const d = await postJson(importFromDatasetUrl(datasetId),
          { source_dataset_id: source.id, image_ids: batch });
        totals.imported += Number(d.imported || 0);
        totals.duplicates += Number(d.duplicates || 0);
        totals.failed += Number(d.failed || 0);
        totals.small += Number(d.small || 0);
        setCopying((c) => (c ? { ...c, done: c.done + batch.length } : c));
      }
      toast.success(datasetCopyOutcome(totals));
      if (totals.small) toast.warning(`${totals.small} image(s) under 768 px — they will stay soft in training.`);
      setSelected(new Set());
    } catch (e) {
      const partial = totals.imported ? ` — ${totals.imported} copied before the error` : '';
      toast.error(`${e?.message || 'Could not copy the images from this dataset.'}${partial}`);
    } finally {
      setCopying(null);
      if (totals.imported) onImported?.();
    }
  };

  const openBank = (id) => {
    try { localStorage.setItem(BANK_CURRENT_KEY, String(id)); } catch { /* ignore */ }
    navigate('/bank');
  };

  if (!open) {
    return (
      <button type="button" onClick={() => setOpen(true)} disabled={disabled}
        className="flex w-full items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2 text-left text-content-muted hover:text-content hover:bg-surface-raised transition-colors disabled:opacity-50">
        <Archive aria-hidden="true" className="h-4 w-4" />
        <span className="text-sm font-medium">Import from a bank or a dataset</span>
        <span className="text-content-subtle text-[0.6875rem]">pick images from a triaged bank or another dataset</span>
        <span aria-hidden className="ml-auto text-content-subtle">→</span>
      </button>
    );
  }

  const loadingSources = banks == null || datasets == null;
  const noSource = !loadingSources && bankRows.length === 0 && datasets.length === 0;
  const importLabel = copying ? `Copying ${copying.done}/${copying.total}…`
    : watching != null ? 'Importing…'
      : starting ? 'Starting…'
        : selected.size ? `Import ${selected.size} selected` : 'Import selected';

  return (
    <div id="ds-add-bank-import" tabIndex={-1}
      className="scroll-mt-20 flex flex-col gap-2 rounded-lg border border-border bg-surface px-3 py-2">
      <div className="flex items-center gap-2">
        <Archive aria-hidden="true" className="h-4 w-4" />
        <span className="text-sm font-medium text-content">Import from a bank or a dataset</span>
        <button type="button" onClick={() => setOpen(false)}
          className="ml-auto text-content-subtle text-[0.6875rem] hover:text-content">
          close
        </button>
      </div>
      <p className="text-content-subtle text-[0.6875rem]">
        Choose a source, then click the images to copy here. Captions travel with them;
        near-duplicates of what is already here are skipped. The source is left untouched.
      </p>

      {loadingSources ? (
        <p className="text-content-subtle text-xs">Loading sources…</p>
      ) : noSource ? (
        <p className="text-content-subtle text-xs">
          No bank and no other dataset yet —{' '}
          <button type="button" onClick={() => navigate('/bank')} className="underline hover:text-content">
            create a bank on the Bank page
          </button>{' '}
          to triage a big folder first.
        </p>
      ) : (
        <>
          <div>
            <label htmlFor="ds-bank-import-select" className="block text-content-muted text-[0.6875rem]">
              Source
            </label>
            <select id="ds-bank-import-select" value={chosen} disabled={busy}
              onChange={(e) => pickSource(e.target.value)}
              className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content disabled:opacity-50">
              <option value="">Choose a bank or a dataset…</option>
              {bankRows.length > 0 && (
                <optgroup label="Banks">
                  {bankRows.map((r) => (
                    <option key={`bank-${r.id}`} value={sourceValue('bank', r.id)}>
                      {r.name} — {r.count == null ? 'counting…' : `${r.count} to import`} ({r.keep}/{r.total} kept)
                    </option>
                  ))}
                </optgroup>
              )}
              {datasets.length > 0 && (
                <optgroup label="Datasets">
                  {datasets.map((d) => (
                    <option key={`dataset-${d.id}`} value={sourceValue('dataset', d.id)}>
                      {d.name} — {d.total} image{d.total === 1 ? '' : 's'} ({d.kept} kept)
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
          </div>
          {currentBank && (
            <p className={`text-[0.6875rem] ${currentBank.ready ? 'text-emerald-300' : 'text-amber-300'}`}>
              {currentBank.hint}
              {!currentBank.ready && currentBank.reason !== 'loading' && (
                <>
                  {' '}
                  <button type="button" onClick={() => openBank(currentBank.id)}
                    className="underline hover:text-content">
                    Open this bank →
                  </button>
                </>
              )}
            </p>
          )}

          {source && (
            <>
              <div className="flex flex-wrap items-center gap-2 text-[0.6875rem]">
                <div role="group" aria-label="Images to show"
                  className="flex shrink-0 rounded-md border border-border overflow-hidden">
                  {[[false, 'Kept'], [true, 'All']].map(([all, label]) => (
                    <button key={label} type="button" aria-pressed={showAll === all}
                      onClick={() => setShowAll(all)}
                      className={`px-2 py-1 text-[0.6875rem] font-semibold ${showAll === all
                        ? 'bg-surface-raised text-content'
                        : 'bg-transparent text-content-muted hover:text-content'}`}>
                      {label}
                    </button>
                  ))}
                </div>
                <span className="text-content-muted">{selected.size} selected</span>
                <button type="button" onClick={selectShown} disabled={busy || !items?.length}
                  className="text-content-subtle underline hover:text-content disabled:opacity-40">
                  Select all shown
                </button>
                <button type="button" onClick={() => setSelected(new Set())}
                  disabled={busy || selected.size === 0}
                  className="text-content-subtle underline hover:text-content disabled:opacity-40">
                  Clear
                </button>
              </div>

              {items == null ? (
                <p className="text-content-subtle text-xs">Loading images…</p>
              ) : items.length === 0 ? (
                <p className="text-content-subtle text-xs">
                  {showAll
                    ? `This ${sourceKind} has no images yet.`
                    : `No kept image in this ${sourceKind} — switch to All to pick among every image.`}
                </p>
              ) : (
                <ul className="m-0 grid max-h-[22rem] list-none grid-cols-4 gap-1.5 overflow-y-auto p-0 sm:grid-cols-6 lg:grid-cols-8">
                  {items.map((img) => {
                    const on = selected.has(img.id);
                    const name = img.name || img.filename || `image ${img.id}`;
                    return (
                      <li key={img.id} className="relative">
                        <button type="button" onClick={() => toggle(img.id)} disabled={busy}
                          aria-pressed={on} aria-label={`${on ? 'Deselect' : 'Select'} ${name}`}
                          title={img.caption || name}
                          className="block w-full overflow-hidden rounded-lg border border-border bg-black hover:border-indigo-400/60 disabled:opacity-60">
                          <img src={thumbOf(img)} alt="" loading="lazy"
                            className="aspect-square w-full object-cover" />
                        </button>
                        {on && (
                          <>
                            <span aria-hidden className="pointer-events-none absolute inset-0 rounded-lg ring-2 ring-inset ring-primary" />
                            <SelectionMark />
                          </>
                        )}
                      </li>
                    );
                  })}
                </ul>
              )}
              {sourceKind === 'bank' && items && items.length < itemsTotal && (
                <button type="button" onClick={loadMore} disabled={loadingMore}
                  className="self-start text-content-subtle text-[0.6875rem] underline hover:text-content disabled:opacity-40">
                  {loadingMore ? 'Loading…' : `Show more (${items.length} of ${itemsTotal})`}
                </button>
              )}
            </>
          )}

          <div className="flex flex-wrap items-center gap-2">
            <button type="button" onClick={importSelected} disabled={busy || !source || selected.size === 0}
              className="rounded-md bg-gradient-primary px-3 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-40">
              {importLabel}
            </button>
            {currentBank && (
              <button type="button" onClick={startAll} disabled={busy || !currentBank.ready}
                className="rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:text-content hover:bg-surface-raised disabled:opacity-40">
                {currentBank.ready ? `Import all ${currentBank.count} kept` : 'Import all kept'}
              </button>
            )}
            {watching != null && (
              <span className="text-content-subtle text-[0.6875rem]">
                running in the background — the grid fills in when it ends
              </span>
            )}
          </div>
        </>
      )}
    </div>
  );
}

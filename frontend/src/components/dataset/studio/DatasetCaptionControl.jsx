import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../../api/fetchClient';
import { useFocusTrap } from '../../../hooks/useFocusTrap';
import {
  BANK, DATASET, SOURCE_ICON, SOURCE_LABEL, captionSourceBody, emptyDrawMessage,
  lockedLabel, normaliseSource, sourceMeta,
} from './captionSource.js';

// Kept separate from the active dataset: the global Studio can compare LoRAs
// from several datasets, while the caption source must stay deliberately chosen.
// The KEY IS UNCHANGED now that banks are a source too — it holds a choice made
// in people's browsers, and a rename would silently reset every one of them.
// captionSource.normaliseSource reads the old kind-less shape as a dataset.
const STORAGE_KEY = 'studioCaptionDataset_v1';

function readLockedDataset() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return normaliseSource(JSON.parse(raw));
  } catch {
    return null;
  }
}

function saveLockedDataset(dataset) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(dataset));
  } catch {
    // A private browser context may refuse storage. The current page still works.
  }
}

function forgetLockedDataset() {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Storage is only a convenience, never a reason to block caption picking.
  }
}

function DatasetCaptionDialog({ open, onClose, onChoose }) {
  const dialogRef = useRef(null);
  const requestRef = useRef(0);
  const [datasets, setDatasets] = useState([]);
  const [banks, setBanks] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  useFocusTrap(dialogRef, open);

  /* Both libraries, in ONE settle: a bank library that is slow (or absent on a
     fresh install) must not keep the datasets — the half that already worked —
     off the screen. /api/banks is asked WITHOUT ?rescan=1 on purpose: the
     rescan re-walks every bank's folder, which is seconds of work this picker
     has no use for (it reads counts, not files). */
  const loadSources = useCallback(async () => {
    const request = ++requestRef.current;
    setLoading(true);
    setError(null);
    const [ds, bk] = await Promise.allSettled([
      apiFetch('/api/dataset/list'),
      apiFetch('/api/banks'),
    ]);
    if (request !== requestRef.current) return;
    setDatasets(ds.status === 'fulfilled' ? (ds.value?.datasets || []) : []);
    setBanks(bk.status === 'fulfilled' ? (bk.value?.banks || []) : []);
    if (ds.status === 'rejected' && bk.status === 'rejected') {
      setError(ds.reason?.message
        || 'Could not load your datasets or banks. Check your connection and try again.');
    } else if (ds.status === 'rejected') {
      setError('Your banks loaded, but the datasets did not. Showing banks only.');
    } else if (bk.status === 'rejected') {
      setError('Your datasets loaded, but the banks did not. Showing datasets only.');
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (!open) return undefined;
    loadSources();
    return () => { requestRef.current += 1; };
  }, [open, loadSources]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 p-3 sm:p-4"
      role="dialog" aria-modal="true" aria-labelledby="caption-dataset-title" ref={dialogRef}
      onClick={(event) => { if (event.target === event.currentTarget) onClose(); }}>
      <div className="flex max-h-[calc(100vh-1.5rem)] w-full max-w-lg flex-col gap-3 overflow-y-auto rounded-2xl border border-border bg-surface-overlay p-4 shadow-xl sm:max-h-[calc(100vh-2rem)]">
        <div className="flex items-start justify-between gap-3">
          <div>
            <h2 id="caption-dataset-title" className="text-sm font-semibold text-content">
              🎲 Pick a caption source
            </h2>
            <p className="mt-1 text-[0.6875rem] leading-snug text-content-subtle">
              A dataset or a bank. The choice stays locked for future runs; selecting one
              draws a random kept caption now.
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close caption source picker"
            className="h-8 w-8 shrink-0 rounded-lg border border-border bg-app text-content-muted hover:text-content">
            ×
          </button>
        </div>

        {loading && (
          <p className="m-0 flex items-center gap-2 rounded-lg border border-border bg-app/60 px-3 py-2 text-[0.75rem] text-content-muted" role="status">
            <span className="inline-block h-4 w-4 rounded-full border-2 border-purple-400/40 border-t-purple-400 animate-spin" aria-hidden />
            Loading your datasets and banks…
          </p>
        )}

        {error && (
          <div className="rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-2 text-[0.75rem] text-red-200" role="alert">
            <p className="m-0">{error}</p>
            <button type="button" onClick={loadSources}
              className="mt-2 rounded border border-red-300/40 px-2 py-1 text-[0.6875rem] font-semibold hover:bg-red-500/10">
              Try again
            </button>
          </div>
        )}

        {!loading && !error && datasets.length === 0 && banks.length === 0 && (
          <p className="m-0 rounded-lg border border-amber-400/40 bg-amber-500/10 px-3 py-2 text-[0.75rem] text-amber-200" role="status">
            No datasets or banks yet. Caption some kept images in either one before using
            this shortcut.
          </p>
        )}

        {!loading && (
          <div className="flex flex-col gap-3">
            {/* TWO SECTIONS, not one merged list. A bank and the dataset it
                promotes into routinely carry the SAME name on this machine, so a
                flat list would offer two identical-looking rows that draw from
                different piles. The heading is what tells them apart. */}
            {[[DATASET, datasets], [BANK, banks]].map(([kind, rows]) => (
              rows.length === 0 ? null : (
                <section key={kind} className="flex flex-col gap-2">
                  <h3 className="m-0 flex items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-content-subtle">
                    <span aria-hidden>{SOURCE_ICON[kind]}</span>
                    {SOURCE_LABEL[kind]}s
                    <span className="font-normal normal-case tracking-normal opacity-70">
                      ({rows.length})
                    </span>
                  </h3>
                  <div className="flex flex-col gap-2"
                    aria-label={`Available caption ${SOURCE_LABEL[kind].toLowerCase()}s`}>
                    {rows.map((row) => {
                      const choice = normaliseSource({ ...row, kind });
                      if (!choice) return null;
                      const meta = sourceMeta(row, kind);
                      return (
                        <button type="button" key={`${kind}-${choice.id}`}
                          onClick={() => onChoose(choice)}
                          className="flex min-w-0 items-center gap-3 rounded-xl border border-border bg-app/60 px-3 py-2.5 text-left hover:border-purple-400/60 hover:bg-purple-500/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400">
                          <span className="text-base" aria-hidden>🔒</span>
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-sm font-semibold text-content">{choice.name}</span>
                            {meta && <span className="block truncate text-[0.6875rem] text-content-subtle">{meta}</span>}
                          </span>
                          <span className="shrink-0 text-[0.6875rem] font-semibold text-purple-200">Use &amp; draw</span>
                        </button>
                      );
                    })}
                  </div>
                </section>
              )
            ))}
          </div>
        )}

        <div className="flex flex-col-reverse gap-2 pt-1 sm:flex-row sm:justify-end">
          <button type="button" onClick={onClose}
            className="rounded-lg border border-border bg-app px-3 py-1.5 text-[0.75rem] text-content-muted hover:text-content">
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

export default function DatasetCaptionControl({ onCaption }) {
  const [lockedDataset, setLockedDataset] = useState(readLockedDataset);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [error, setError] = useState(null);

  const clearLockedDataset = useCallback(() => {
    setLockedDataset(null);
    forgetLockedDataset();
  }, []);

  const openPicker = useCallback(() => {
    setError(null);
    setPickerOpen(true);
  }, []);

  const drawCaption = useCallback(async (dataset = lockedDataset) => {
    const target = normaliseSource(dataset);
    const body = captionSourceBody(target);
    if (!target || !body) {
      openPicker();
      return;
    }

    setDrawing(true);
    setError(null);
    try {
      const result = await postJson('/api/studio/random-caption', body);
      const caption = typeof result?.caption === 'string' ? result.caption.trim() : '';
      if (!caption) {
        setError({ message: emptyDrawMessage(target), action: 'choose' });
        return;
      }
      onCaption(caption);
    } catch (err) {
      if (err?.status === 404 || err?.status === 400) {
        clearLockedDataset();
        setError({
          message: `The locked ${SOURCE_LABEL[target.kind].toLowerCase()} is no longer available.`
            + ' Choose one from your library and try again.',
          action: 'choose',
        });
      } else if (err?.status === 422) {
        setError({
          message: err.message || emptyDrawMessage(target),
          action: 'choose',
        });
      } else {
        setError({
          message: err?.message || 'Could not draw a caption. Check your connection and try again.',
          action: 'retry',
        });
      }
    } finally {
      setDrawing(false);
    }
  }, [clearLockedDataset, lockedDataset, onCaption, openPicker]);

  const chooseDataset = useCallback((dataset) => {
    const choice = normaliseSource(dataset);
    if (!choice) return;
    saveLockedDataset(choice);
    setLockedDataset(choice);
    setPickerOpen(false);
    drawCaption(choice);
  }, [drawCaption]);

  const retry = useCallback(() => {
    if (error?.action === 'choose') openPicker();
    else drawCaption();
  }, [drawCaption, error?.action, openPicker]);

  return (
    <>
      <span className="inline-flex max-w-full items-center gap-1">
        <button type="button" onClick={() => drawCaption()} disabled={drawing}
          aria-busy={drawing}
          title={lockedDataset
            ? 'Draw a random kept caption from ' + lockedLabel(lockedDataset)
            : 'Choose a dataset or a bank, then draw a random caption'}
          className="min-h-7 rounded-l border border-border bg-surface px-2 py-0.5 text-[0.625rem] text-content-subtle hover:text-content disabled:opacity-50">
          {drawing ? '🎲 Drawing…' : '🎲 Caption'}
        </button>
        <button type="button" onClick={openPicker} disabled={drawing}
          aria-label="Choose or change the caption source" aria-haspopup="dialog"
          aria-expanded={pickerOpen}
          title={lockedDataset ? 'Change caption source' : 'Choose caption source'}
          className="-ml-1 min-h-7 rounded-r border border-border bg-surface px-1.5 py-0.5 text-[0.625rem] text-content-subtle hover:text-content disabled:opacity-50">
          <span aria-hidden>▾</span>
        </button>
      </span>

      {lockedDataset && (
        <span className="inline-flex max-w-full items-center gap-1 rounded border border-purple-400/30 bg-purple-500/10 px-1.5 py-0.5 text-[0.625rem] text-purple-100"
          title={'Caption source locked — ' + lockedLabel(lockedDataset)} aria-live="polite">
          <span aria-hidden>{SOURCE_ICON[lockedDataset.kind]}</span>
          <span className="max-w-36 truncate">{lockedDataset.name}</span>
        </span>
      )}

      {error && (
        <span className="basis-full rounded-lg border border-red-400/40 bg-red-500/10 px-2 py-1.5 text-[0.6875rem] leading-snug text-red-200" role="alert">
          {error.message}{' '}
          <button type="button" onClick={retry}
            className="font-semibold underline underline-offset-2 hover:text-white">
            {error.action === 'choose' ? 'Choose a source' : 'Try again'}
          </button>
        </span>
      )}

      <DatasetCaptionDialog open={pickerOpen} onClose={() => setPickerOpen(false)}
        onChoose={chooseDataset} />
    </>
  );
}

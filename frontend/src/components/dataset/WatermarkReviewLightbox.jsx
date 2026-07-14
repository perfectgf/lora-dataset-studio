/**
 * Full-screen watermark REVIEW mode. Walk the flagged (🚩 detected) images one by one,
 * see the detected bbox drawn over the photo (crucial to spot a false positive at a
 * glance), and rule on each: 🧽 Clean (apply this image's routed removal now), ✓ Not a
 * watermark (dismiss — badge clears, future scans skip it), or ✕ Reject (drop it from
 * the kept set). Big tap targets + keyboard: ← → navigate, c/d/x act, Esc closes.
 *
 * Clean does NOT auto-advance: the user asked to actually SEE the cleaned pixels before
 * moving on. A successful clean reloads the same tile (existing nonce/cache-bust), hides
 * the now-stale bbox, and shows a "Cleaned — cropped/inpainted" badge; the user then
 * presses → themselves. Dismiss/Reject don't touch pixels — nothing to look at — so they
 * keep the original auto-advance. Navigation is held (arrows + buttons) while an action
 * is in flight so the "Cleaning…" spinner can't end up drawn over the wrong image.
 *
 * The queue is FROZEN on open (a snapshot of the currently-detected images): actions
 * remove images from the live 'detected' set, but the filmstrip stays stable so the
 * user walks it once. Per-image outcomes are tracked locally; the parent refreshes the
 * grid counts underneath and shows a recap toast on close.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { displayLabel } from '../../utils/labels';
import {
  MAX_WATERMARK_REGIONS,
  buildWatermarkReviewState,
  cloneWatermarkRegions,
  deleteSelectedWatermarkRegion,
} from '../../utils/watermarkRegions';
import WatermarkRegionEditor from './WatermarkRegionEditor';

// The action Clean WILL take, per backend route (watermark_route in the payload).
const ROUTE_LABEL = {
  crop: { icon: '✂', text: 'Crop the watermarked border', cls: 'text-sky-300' },
  lama: { icon: '🖌', text: 'Inpaint the mark (LaMa)', cls: 'text-emerald-300' },
  review: { icon: '👁', text: 'On the subject — needs manual review', cls: 'text-amber-300' },
};

// Per-image outcome after an action. Terminal ones leave the 'detected' set (badge
// gone) and hide the (now stale) bbox overlay; the rest keep the image flagged so the
// user can still reject. Of the terminal outcomes, only dismissed/rejected auto-advance
// (see AUTO_ADVANCE below) — cleaned holds so the user can see the result.
const OUTCOME = {
  cleaned: { icon: '✨', text: 'Cleaned', cls: 'text-emerald-300', terminal: true },
  dismissed: { icon: '⊘', text: 'Marked “not a watermark”', cls: 'text-content-subtle', terminal: true },
  rejected: { icon: '✕', text: 'Rejected — removed from the set', cls: 'text-red-300', terminal: true },
  review: { icon: '👁', text: 'Left for manual review', cls: 'text-amber-300', terminal: false },
  skipped: { icon: '⬇', text: 'Skipped — inpainting not installed', cls: 'text-amber-300', terminal: false },
  failed: { icon: '⚠', text: 'Clean failed', cls: 'text-red-300', terminal: false },
};

// Which terminal outcomes auto-advance to the next image. Cleaned is deliberately
// excluded — the user needs to see the cleaned pixels first.
const AUTO_ADVANCE = new Set(['dismissed', 'rejected']);

// Clean's outcome text, refined with which route actually ran (from the clean API's
// per-request counts) once known.
const CLEAN_DETAIL_TEXT = { cropped: 'Cleaned — cropped', inpainted: 'Cleaned — inpainted' };

const RECAP_ORDER = ['cleaned', 'dismissed', 'rejected', 'review', 'skipped', 'failed'];
const RECAP_WORD = { cleaned: 'cleaned', dismissed: 'dismissed', rejected: 'rejected',
  review: 'need review', skipped: 'skipped', failed: 'failed' };

export function buildWatermarkRecap(outcomes) {
  const c = {};
  for (const k of Object.values(outcomes || {})) c[k] = (c[k] || 0) + 1;
  const parts = RECAP_ORDER.filter((k) => c[k]).map((k) => `${c[k]} ${RECAP_WORD[k]}`);
  return parts.join(' · ');
}

function apiErrorText(error, fallback = 'Could not save correction zones') {
  const value = error?.error ?? error;
  if (typeof value === 'string' && value) return value;
  if (typeof value?.detail === 'string' && value.detail) return value.detail;
  if (typeof value?.message === 'string' && value.message) return value.message;
  return fallback;
}

export default function WatermarkReviewLightbox({ datasetId, queue, caps, nonces = {},
                                                  onSaveRegions, onClean, onDismiss,
                                                  onReject, onClose }) {
  const initialReviewStateRef = useRef(null);
  if (!initialReviewStateRef.current) {
    initialReviewStateRef.current = buildWatermarkReviewState(queue);
  }
  const initialReviewState = initialReviewStateRef.current;
  const [idx, setIdx] = useState(0);
  const [outcomes, setOutcomes] = useState({});   // id -> OUTCOME key
  const [cleanDetail, setCleanDetail] = useState({}); // id -> 'cropped' | 'inpainted' (cleaned outcomes only)
  const [regionsById, setRegionsById] = useState(initialReviewState.regionsById);
  const [manualById, setManualById] = useState(initialReviewState.manualById);
  const [selectedById, setSelectedById] = useState(initialReviewState.selectedById);
  const [addModeById, setAddModeById] = useState(initialReviewState.addModeById);
  const [saveStateById, setSaveStateById] = useState(initialReviewState.saveStateById);
  const [working, setWorking] = useState(false);
  const [workingKind, setWorkingKind] = useState(null); // 'clean' | 'dismiss' | 'reject' — which action is in flight
  const [note, setNote] = useState(null);         // transient inline note {tone, text}
  const dialogRef = useRef(null);
  const workingRef = useRef(false);               // re-entrancy guard (double keypress)
  const saveJobsRef = useRef({});                 // id -> latest serialized save job

  useFocusTrap(dialogRef, queue.length > 0);

  const total = queue.length;
  const item = idx >= 0 && idx < total ? queue[idx] : null;
  const outcome = item ? outcomes[item.id] : null;
  const regions = item ? (regionsById[item.id] || []) : [];
  const manual = item ? Boolean(manualById[item.id]) : false;
  const selectedRegion = item ? selectedById[item.id] : null;
  const addMode = item ? Boolean(addModeById[item.id]) : false;
  const saveState = item
    ? (saveStateById[item.id] || { status: 'saved', error: null })
    : { status: 'saved', error: null };
  const saveBlocked = saveState.status === 'saving' || saveState.status === 'failed';
  const manualLamaMissing = manual && caps?.watermark_inpaint === false;
  const allDone = total > 0 && Object.keys(outcomes).length >= total
    && queue.every((q) => outcomes[q.id]);

  const setRegionsFor = useCallback((id, nextRegions) => {
    const next = cloneWatermarkRegions(nextRegions);
    setRegionsById((current) => ({ ...current, [id]: next }));
  }, []);
  const setManualFor = useCallback((id, nextManual) => {
    setManualById((current) => ({ ...current, [id]: Boolean(nextManual) }));
  }, []);
  const setSelectedFor = useCallback((id, nextSelected) => {
    setSelectedById((current) => ({ ...current, [id]: nextSelected }));
  }, []);
  const setAddModeFor = useCallback((id, nextAddMode) => {
    setAddModeById((current) => ({ ...current, [id]: Boolean(nextAddMode) }));
  }, []);
  const setSaveStateFor = useCallback((id, nextSaveState) => {
    setSaveStateById((current) => ({ ...current, [id]: nextSaveState }));
  }, []);

  const persistRegions = useCallback((id, regionsOrNull, visibleRegions, nextManual) => {
    const visible = cloneWatermarkRegions(visibleRegions);
    const payload = regionsOrNull === null ? null : cloneWatermarkRegions(regionsOrNull);
    const previous = saveJobsRef.current[id];
    const waitForPrevious = previous?.promise
      ? previous.promise.catch(() => undefined)
      : Promise.resolve();
    const job = {
      status: 'saving', error: null, payload, visible, manual: Boolean(nextManual), promise: null,
    };

    saveJobsRef.current[id] = job;
    setRegionsFor(id, visible);
    setManualFor(id, nextManual);
    setSaveStateFor(id, { status: 'saving', error: null });

    job.promise = waitForPrevious
      .then(async () => {
        const response = await onSaveRegions(
          id,
          payload === null ? null : cloneWatermarkRegions(payload),
        );
        if (!response || response.ok === false) {
          throw new Error(apiErrorText(response));
        }
        return response;
      })
      .then((response) => {
        if (saveJobsRef.current[id] !== job) return response;
        job.status = 'saved';
        const effective = Array.isArray(response.effective_watermark_regions)
          ? response.effective_watermark_regions
          : visible;
        setRegionsFor(id, effective);
        const responseHasOverride = Object.prototype.hasOwnProperty.call(
          response,
          'watermark_regions',
        );
        setManualFor(id, responseHasOverride
          ? Array.isArray(response.watermark_regions)
          : job.manual);
        setSelectedById((current) => ({
          ...current,
          [id]: effective.length
            ? Math.min(current[id] ?? 0, effective.length - 1)
            : null,
        }));
        setSaveStateFor(id, { status: 'saved', error: null });
        return response;
      })
      .catch((error) => {
        if (saveJobsRef.current[id] === job) {
          const message = apiErrorText(error);
          job.status = 'failed';
          job.error = message;
          setSaveStateFor(id, { status: 'failed', error: message });
        }
        throw error;
      });

    return job.promise;
  }, [onSaveRegions, setManualFor, setRegionsFor, setSaveStateFor]);

  const isSaveBlocked = useCallback((id) => {
    const job = saveJobsRef.current[id];
    return Boolean(job && job.status !== 'saved');
  }, []);

  const waitForLatestSave = useCallback(async (id) => {
    while (true) {
      const job = saveJobsRef.current[id];
      if (!job) return true;
      if (job.status === 'failed') return false;
      try { await job.promise; } catch { /* status below is authoritative */ }
      if (saveJobsRef.current[id] === job) return job.status === 'saved';
    }
  }, []);

  const recap = useMemo(() => buildWatermarkRecap(outcomes), [outcomes]);
  const close = useCallback(() => {
    if (workingRef.current) return;
    if (item && isSaveBlocked(item.id)) {
      setNote({ tone: 'err', text: 'Retry or reset the correction zones before leaving this image.' });
      return;
    }
    onClose(recap);
  }, [isSaveBlocked, item, onClose, recap]);

  const go = useCallback((delta) => {
    if (workingRef.current) return;   // hold navigation while an action is in flight
    if (item && isSaveBlocked(item.id)) {
      setNote({ tone: 'err', text: 'Correction zones must be saved before navigating.' });
      return;
    }
    setNote(null);
    setIdx((i) => Math.min(total - 1, Math.max(0, i + delta)));
  }, [isSaveBlocked, item, total]);
  const advance = useCallback(() => setIdx((i) => Math.min(total - 1, i + 1)), [total]);

  const run = useCallback(async (kind, fn) => {
    if (!item || workingRef.current) return;
    workingRef.current = true;
    setWorking(true);
    setWorkingKind(kind);
    setNote(null);
    try {
      const { key, note: n, detail } = await fn(item);
      if (key) setOutcomes((m) => ({ ...m, [item.id]: key }));
      if (detail) setCleanDetail((m) => ({ ...m, [item.id]: detail }));
      if (n) setNote(n);
      if (key && AUTO_ADVANCE.has(key)) advance();
    } finally {
      workingRef.current = false;
      setWorking(false);
      setWorkingKind(null);
    }
  }, [item, advance]);

  const commitRegions = useCallback((nextRegions) => {
    if (!item || outcome === 'cleaned') return;
    const next = cloneWatermarkRegions(nextRegions);
    setNote(null);
    setAddModeFor(item.id, false);
    void persistRegions(item.id, next, next, true).catch(() => undefined);
  }, [item, outcome, persistRegions, setAddModeFor]);

  const deleteSelectedRegion = useCallback(() => {
    if (!item || isSaveBlocked(item.id) || workingRef.current) return;
    const next = deleteSelectedWatermarkRegion(regions, selectedRegion);
    if (next.selectedIndex === null && selectedRegion === null) return;
    setSelectedFor(item.id, next.selectedIndex);
    commitRegions(next.regions);
  }, [commitRegions, isSaveBlocked, item, regions, selectedRegion, setSelectedFor]);

  const resetDetection = useCallback(() => {
    if (!item || saveState.status === 'saving' || workingRef.current) return;
    const detected = initialReviewState.detectionRegionsById[item.id] || [];
    setNote(null);
    setAddModeFor(item.id, false);
    setSelectedFor(item.id, detected.length ? 0 : null);
    void persistRegions(item.id, null, detected, false).catch(() => undefined);
  }, [initialReviewState.detectionRegionsById, item, persistRegions, saveState.status,
    setAddModeFor, setSelectedFor]);

  const retrySave = useCallback(() => {
    if (!item || workingRef.current) return;
    const job = saveJobsRef.current[item.id];
    if (!job || job.status !== 'failed') return;
    setNote(null);
    void persistRegions(item.id, job.payload, job.visible, job.manual).catch(() => undefined);
  }, [item, persistRegions]);

  const doClean = useCallback(() => {
    if (!item || outcome === 'cleaned' || !regions.length || manualLamaMissing) return;
    if (saveJobsRef.current[item.id]?.status === 'failed') return;
    return run('clean', async (it) => {
      if (!await waitForLatestSave(it.id)) {
        return { note: { tone: 'err', text: 'Correction zones could not be saved. Retry or reset them before cleaning.' } };
      }
      const d = await onClean(it.id);
      if (!d || d.ok === false) {
        return { key: 'failed', note: { tone: 'err', text: (d && d.error && (d.error.detail || d.error)) || 'Clean failed' } };
      }
      if (d.error) {
        return { key: 'failed', note: { tone: 'err',
          text: d.error.kind === 'unavailable'
            ? 'Inpainting isn’t installed — install it (next to the 🧽 tools) or reject/crop this one.'
            : `Inpainting failed: ${d.error.detail || d.error.kind}` } };
      }
      if (d.cropped || d.inpainted) return { key: 'cleaned', detail: d.cropped ? 'cropped' : 'inpainted' };
      if (d.needs_review) {
        return { key: 'review', note: { tone: 'warn',
          text: 'On the subject — auto crop/inpaint would damage the photo. Reject it or crop it manually.' } };
      }
      if (d.skipped) {
        return { key: 'skipped', note: { tone: 'warn',
          text: 'Off-center mark, but inpainting isn’t installed — install it or reject/crop this one.' } };
      }
      return { key: 'cleaned' };   // nothing to do reported → treat as resolved
    });
  }, [item, manualLamaMissing, onClean, outcome, regions.length, run, waitForLatestSave]);

  const doDismiss = useCallback(() => {
    if (!item || isSaveBlocked(item.id)) return;
    return run('dismiss', async (it) => {
      const d = await onDismiss(it.id);
      if (!d || d.ok === false) return { note: { tone: 'err', text: (d && d.error) || 'Could not dismiss' } };
      return { key: 'dismissed' };
    });
  }, [isSaveBlocked, item, onDismiss, run]);

  const doReject = useCallback(() => {
    if (!item || isSaveBlocked(item.id)) return;
    return run('reject', async (it) => {
      await onReject(it.id);
      return { key: 'rejected' };
    });
  }, [isSaveBlocked, item, onReject, run]);

  // Keyboard: ← → navigate · Esc close · c Clean · d Dismiss · x Reject.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') { e.preventDefault(); close(); return; }
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(1); return; }
      const k = e.key.toLowerCase();
      if (k === 'c') { e.preventDefault(); doClean(); }
      else if (k === 'd') { e.preventDefault(); doDismiss(); }
      else if (k === 'x') { e.preventDefault(); doReject(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close, go, doClean, doDismiss, doReject]);

  if (!total) return null;

  const alt = item ? (displayLabel(item.variation_label) || 'dataset image') : '';
  const nonce = item ? (nonces[item.id] || 0) : 0;
  const url = item && item.filename
    ? `/api/dataset/${datasetId}/img/${encodeURIComponent(item.filename)}${nonce ? `?v=${nonce}` : ''}`
    : null;
  const route = manual
    ? {
        icon: '🖌',
        text: `Inpaint ${regions.length} selected zone${regions.length === 1 ? '' : 's'}`,
        cls: 'text-emerald-300',
      }
    : (item ? ROUTE_LABEL[item.watermark_route] : null);
  const oc = outcome ? OUTCOME[outcome] : null;
  const ocText = outcome === 'cleaned' && cleanDetail[item?.id]
    ? CLEAN_DETAIL_TEXT[cleanDetail[item.id]] || oc.text
    : oc?.text;
  const cleaning = working && workingKind === 'clean';   // navigation is held while true, so this always tracks `item`
  const showEditor = !(oc && oc.terminal) && !cleaning;
  const automaticLamaMissing = !manual && item?.watermark_route === 'lama'
    && caps?.watermark_inpaint === false;
  const editorDisabled = working || saveBlocked;
  const actionBlocked = working || saveBlocked;
  const cleanDisabled = actionBlocked || outcome === 'cleaned' || regions.length === 0
    || manualLamaMissing;
  const atRegionLimit = regions.length >= MAX_WATERMARK_REGIONS;
  const selectedRegionExists = Number.isInteger(selectedRegion)
    && selectedRegion >= 0 && selectedRegion < regions.length;
  const saveLabel = saveState.status === 'saving'
    ? 'Saving…'
    : saveState.status === 'failed' ? 'Save failed' : 'Saved';
  const saveCls = saveState.status === 'saving'
    ? 'text-amber-200'
    : saveState.status === 'failed' ? 'text-red-300' : 'text-emerald-300';

  const btn = 'flex-1 min-w-[7rem] min-h-[3rem] px-3 rounded-lg text-sm font-semibold flex items-center justify-center gap-1.5 disabled:opacity-40';

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Review flagged watermarks"
      className="fixed inset-0 z-[9997] bg-black/90 flex flex-col" onClick={close}>

      {/* Top bar: counter + title + close */}
      <div onClick={(e) => e.stopPropagation()}
        className="shrink-0 flex items-center gap-2 px-3 py-2 bg-black/60 border-b border-white/10">
        <span className="text-white font-semibold text-sm tabular-nums">{idx + 1} / {total}</span>
        <span className="px-1.5 py-0.5 rounded text-[10px] bg-white/10 text-white/80">
          {item?.source === 'import' ? 'real' : 'generated'}{item?.framing ? ` · ${item.framing}` : ''}
        </span>
        <span className="text-white/70 text-xs truncate">{alt}</span>
        <button type="button" onClick={close} disabled={working || saveBlocked}
          title="Close (Esc)" aria-label="Close review"
          className="ml-auto w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-white text-lg leading-none disabled:opacity-40">✕</button>
      </div>

      {/* Image + editable correction-zone overlays */}
      <div onClick={(e) => e.stopPropagation()}
        className="flex-1 min-h-0 flex items-center justify-center p-3">
        {url ? (
          showEditor ? (
            <WatermarkRegionEditor
              src={url}
              alt={alt}
              regions={regions}
              disabled={editorDisabled}
              addMode={addMode}
              selectedIndex={selectedRegion}
              onAddModeChange={(next) => setAddModeFor(item.id, next)}
              onSelectedIndexChange={(next) => setSelectedFor(item.id, next)}
              onCommit={commitRegions}
            >
              {oc && (
                <div className="absolute inset-x-0 bottom-0 z-50 flex justify-center pb-2 pointer-events-none">
                  <span className={`px-2 py-1 rounded-lg bg-black/75 text-xs font-semibold leading-normal ${oc.cls}`}>
                    {oc.icon} {ocText}
                  </span>
                </div>
              )}
            </WatermarkRegionEditor>
          ) : (
            <div className="relative">
              <img src={url} alt={alt} className="block max-h-[70vh] max-w-[92vw] select-none" />
              {cleaning ? (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-sm">
                  <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/75 text-amber-200 text-sm font-semibold">
                    <span aria-hidden className="w-4 h-4 rounded-full border-2 border-amber-200/40 border-t-amber-200 animate-spin" />
                    Cleaning…
                  </span>
                </div>
              ) : oc && (
                <div className="absolute inset-x-0 bottom-0 flex justify-center pb-2 pointer-events-none">
                  <span className={`px-2 py-1 rounded-lg bg-black/75 text-xs font-semibold ${oc.cls}`}>
                    {oc.icon} {ocText}
                  </span>
                </div>
              )}
            </div>
          )
        ) : (
          <span className="text-white/60 text-sm">image unavailable</span>
        )}
      </div>

      {/* Bottom: planned action, note, action buttons, nav, legend */}
      <div onClick={(e) => e.stopPropagation()}
        className="shrink-0 bg-black/70 border-t border-white/10 px-3 py-2.5 flex flex-col gap-2">

        <div className="flex items-center justify-center gap-2 text-xs flex-wrap">
          <span className="text-white/70 tabular-nums">
            {regions.length} correction zone{regions.length === 1 ? '' : 's'}
          </span>
          <button type="button" aria-pressed={addMode}
            onClick={() => setAddModeFor(item.id, !addMode)}
            disabled={editorDisabled || Boolean(oc?.terminal) || atRegionLimit}
            title={atRegionLimit ? `Maximum of ${MAX_WATERMARK_REGIONS} zones reached` : 'Drag on the image to add a correction zone'}
            className={`min-h-9 px-3 rounded-lg border text-xs font-semibold disabled:opacity-40 ${addMode
              ? 'border-sky-300 bg-sky-500/25 text-sky-100'
              : 'border-white/20 bg-white/10 text-white hover:bg-white/20'}`}>
            + Add zone
          </button>
          <button type="button" onClick={deleteSelectedRegion}
            disabled={editorDisabled || Boolean(oc?.terminal) || !selectedRegionExists}
            title={selectedRegionExists ? `Delete selected zone ${selectedRegion + 1}` : 'Select a zone to delete it'}
            className="min-h-9 px-3 rounded-lg border border-white/20 bg-white/10 text-white hover:bg-white/20 text-xs font-semibold disabled:opacity-40">
            Delete zone
          </button>
          <button type="button" onClick={resetDetection}
            disabled={working || saveState.status === 'saving' || Boolean(oc?.terminal) || !manual}
            title="Discard manual zones and restore the detected rectangle"
            className="min-h-9 px-3 rounded-lg border border-white/20 bg-white/10 text-white hover:bg-white/20 text-xs font-semibold disabled:opacity-40">
            Reset detection
          </button>
          <span aria-live="polite" className={`font-semibold ${saveCls}`}>
            {saveState.status === 'saved' ? '✓ ' : saveState.status === 'failed' ? '⚠ ' : ''}{saveLabel}
          </span>
        </div>

        {saveState.status === 'failed' && (
          <div role="alert" className="flex items-center justify-center gap-2 text-xs text-red-300 flex-wrap">
            <span>{saveState.error || 'Could not save correction zones.'}</span>
            <button type="button" onClick={retrySave} disabled={working}
              className="min-h-8 px-2.5 rounded-md border border-red-300/50 bg-red-500/15 text-red-100 hover:bg-red-500/25 disabled:opacity-40">
              Retry save
            </button>
            <span className="text-white/50">or reset detection</span>
          </div>
        )}

        <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
          <span className="text-white/50">Planned:</span>
          {route ? (
            <span className={`font-semibold ${route.cls}`}>{route.icon} {route.text}</span>
          ) : (
            <span className="text-white/60">— unknown (missing box)</span>
          )}
          {manual && regions.length > 0 && (
            <span className="text-emerald-200/70 text-xs">· one composite LaMa pass</span>
          )}
          {manualLamaMissing && (
            <span className="text-amber-300/90 text-xs">· LaMa inpainting isn’t installed → install it from the 🧽 tools before Clean</span>
          )}
          {automaticLamaMissing && (
            <span className="text-amber-300/90 text-xs">· inpainting not installed → Clean will skip</span>
          )}
        </div>

        {note && (
          <p className={`text-center text-xs ${note.tone === 'err' ? 'text-red-300' : 'text-amber-300'}`}>
            {note.text}
          </p>
        )}

        <div className="flex gap-2 flex-wrap">
          <button type="button" onClick={doClean} disabled={cleanDisabled}
            title={outcome === 'cleaned'
              ? 'Already cleaned'
              : regions.length === 0
                ? 'Add at least one correction zone before cleaning'
                : manualLamaMissing
                  ? 'Install LaMa inpainting before cleaning manual zones'
                  : saveBlocked
                    ? 'Save correction zones before cleaning'
              : "Apply this image's watermark removal now (crop / inpaint / manual review) — shortcut c"}
            className={`${btn} bg-amber-500/20 border border-amber-400/50 text-amber-100 hover:bg-amber-500/30`}>
            {cleaning ? '🧽 Cleaning…' : <>🧽 Clean <kbd className="text-[10px] text-white/50">c</kbd></>}
          </button>
          <button type="button" onClick={doDismiss} disabled={actionBlocked}
            title="This is NOT a watermark (false positive) — clears the flag, future scans skip it — shortcut d"
            className={`${btn} bg-emerald-600/20 border border-emerald-400/40 text-emerald-100 hover:bg-emerald-600/30`}>
            ✓ Not a watermark <kbd className="text-[10px] text-white/50">d</kbd>
          </button>
          <button type="button" onClick={doReject} disabled={actionBlocked}
            title="Reject this image — it leaves the kept set (for watermarks that can't be recovered) — shortcut x"
            className={`${btn} bg-red-600/20 border border-red-400/40 text-red-100 hover:bg-red-600/30`}>
            ✕ Reject <kbd className="text-[10px] text-white/50">x</kbd>
          </button>
        </div>

        <div className="flex items-center justify-between gap-2">
          <button type="button" onClick={() => go(-1)} disabled={idx <= 0 || actionBlocked}
            title="Previous (←)" aria-label="Previous image"
            className="px-3 min-h-[2.5rem] rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm disabled:opacity-30">← Prev</button>
          <span className="text-white/40 text-[11px] text-center hidden sm:block">
            ← → navigate · <kbd>c</kbd> clean · <kbd>d</kbd> dismiss · <kbd>x</kbd> reject · Esc close
          </span>
          {allDone ? (
            <button type="button" onClick={close} disabled={actionBlocked}
              className="px-4 min-h-[2.5rem] rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold disabled:opacity-40">
              Done ✓
            </button>
          ) : (
            <button type="button" onClick={() => go(1)} disabled={idx >= total - 1 || actionBlocked}
              title="Next (→)" aria-label="Next image"
              className="px-3 min-h-[2.5rem] rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm disabled:opacity-30">Next →</button>
          )}
        </div>
      </div>
    </div>
  );
}

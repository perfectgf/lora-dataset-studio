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
 * presses → themselves. If they don't like the result, ↩ Restore original (shortcut r)
 * takes the Clean button's place: it brings the preserved original back, drops the
 * cleaned outcome so the editor returns, and lets them re-clean straight away — often
 * with the other engine. Dismiss/Reject don't touch pixels — nothing to look at — so they
 * keep the original auto-advance. Navigation is held (arrows + buttons) while an action
 * is in flight so the "Cleaning…"/"Restoring…" spinner can't end up drawn over the wrong image.
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
  klein: { icon: '🎨', text: 'Masked Klein inpaint (crop-and-stitch)', cls: 'text-emerald-300' },
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
const CLEAN_DETAIL_TEXT = { cropped: 'Cleaned — cropped', inpainted: 'Cleaned — inpainted',
  inpainted_klein: 'Cleaned — Klein inpaint' };

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

function isInteractiveShortcutTarget(target) {
  return Boolean(target?.closest?.(
    'button, input, select, textarea, [contenteditable="true"], [role="button"], [role="slider"]',
  ));
}

export default function WatermarkReviewLightbox({ datasetId, queue, caps, nonces = {},
                                                  onSaveRegions, onClean, onRestore, onDismiss,
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
  // Per-image inpaint engine (the batch's LaMa|Klein toggle, mirrored here). Klein is
  // the ONLY engine that can clean an on-subject ('review') mark, so it makes those
  // actionable; LaMa stays the fast default. Greyed when Klein isn't ready.
  const kleinReady = caps?.watermark_klein !== false;
  const [method, setMethod] = useState('lama');
  // Per-image crop-vs-inpaint choice (id -> 'crop' | 'inpaint'). Only offered when a
  // SAFE border crop exists for the image (watermark_route === 'crop') and no manual
  // zones are drawn. Unset entries fall back to the persisted "Allow auto-crop" default.
  const [cropChoiceById, setCropChoiceById] = useState({});
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
  const kleinSelected = method === 'klein';
  // Crop-vs-inpaint for THIS image. `canCrop` = a safe border crop exists (and no manual
  // zones, which are always repainted). The choice defaults to the persisted "Allow
  // auto-crop" preference; `useCrop` is the resolved decision. `effectiveRoute` is what
  // Clean will actually do — crop, or the crop-disabled fallback (watermark_route_nocrop)
  // — and drives both the planned-action label and the LaMa-needed gating below.
  const canCrop = Boolean(item) && !manual && item.watermark_route === 'crop';
  const cropChoice = item ? cropChoiceById[item.id] : undefined;
  const defaultUseCrop = caps?.watermark_allow_crop !== false;
  const useCrop = canCrop && (cropChoice ? cropChoice === 'crop' : defaultUseCrop);
  const effectiveRoute = !item || manual
    ? null
    : useCrop ? 'crop' : (canCrop ? item.watermark_route_nocrop : item.watermark_route);
  // What Clean forwards as allow_crop: an explicit per-image override only when a crop is
  // actually on the table; otherwise undefined → the backend uses the persisted default.
  const allowCropArg = canCrop ? useCrop : undefined;
  // Which engine the planned Clean actually needs, and whether it's installed. LaMa is
  // required only for a real LaMa inpaint (manual zones or an effective 'lama' route);
  // a crop needs no engine, and the 'review' route under LaMa is a no-op (needs_review).
  const lamaNeeded = !kleinSelected && (manual || effectiveRoute === 'lama');
  const engineMissing = kleinSelected
    ? !kleinReady
    : (lamaNeeded && caps?.watermark_inpaint === false);
  const manualLamaMissing = engineMissing;   // drives the existing Clean gating + copy
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
    const activeSave = saveJobsRef.current[item.id];
    if (activeSave && activeSave.status !== 'saved') return;
    return run('clean', async (it) => {
      if (!await waitForLatestSave(it.id)) {
        return { note: { tone: 'err', text: 'Correction zones could not be saved. Retry or reset them before cleaning.' } };
      }
      const d = await onClean(it.id, method, allowCropArg);
      if (!d || d.ok === false) {
        return { key: 'failed', note: { tone: 'err', text: (d && d.error && (d.error.detail || d.error)) || 'Clean failed' } };
      }
      if (d.error) {
        return { key: 'failed', note: { tone: 'err',
          text: d.error.kind === 'unavailable'
            ? 'Inpainting isn’t installed — install it (next to the 🧽 tools) or reject/crop this one.'
            : `Inpainting failed: ${d.error.detail || d.error.kind}` } };
      }
      if (d.cropped || d.inpainted || d.inpainted_klein) {
        return { key: 'cleaned',
          detail: d.cropped ? 'cropped' : d.inpainted_klein ? 'inpainted_klein' : 'inpainted' };
      }
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
  }, [item, manualLamaMissing, method, allowCropArg, onClean, outcome, regions.length, run, waitForLatestSave]);

  // Undo a Clean: the user didn't like the result (LaMa vs Klein, a bad crop…). Bring
  // the preserved original back, drop the 'cleaned' outcome so the editor returns, and
  // let them re-clean straight away (often with the OTHER engine). Only offered once a
  // real pixel edit happened (cleanDetail set) — the backend still 404s if no .orig.
  const doRestore = useCallback(() => {
    if (!item || outcome !== 'cleaned' || !cleanDetail[item.id]) return;
    return run('restore', async (it) => {
      const d = await onRestore(it.id);
      if (!d || d.ok === false) {
        return { note: { tone: 'err',
          text: (d && d.error && (d.error.detail || d.error)) || 'Could not restore the original' } };
      }
      setOutcomes((m) => { const n = { ...m }; delete n[it.id]; return n; });
      setCleanDetail((m) => { const n = { ...m }; delete n[it.id]; return n; });
      return { note: { tone: 'warn', text: 'Original restored — re-clean it, or switch engine and clean again.' } };
    });
  }, [item, outcome, cleanDetail, onRestore, run]);

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
      if (isInteractiveShortcutTarget(e.target)) return;
      if (e.key === 'Escape') { e.preventDefault(); close(); return; }
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(-1); return; }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(1); return; }
      const k = e.key.toLowerCase();
      if (k === 'c') { e.preventDefault(); doClean(); }
      else if (k === 'r') { e.preventDefault(); doRestore(); }
      else if (k === 'd') { e.preventDefault(); doDismiss(); }
      else if (k === 'x') { e.preventDefault(); doReject(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [close, go, doClean, doRestore, doDismiss, doReject]);

  if (!total) return null;

  const alt = item ? (displayLabel(item.variation_label) || 'dataset image') : '';
  const nonce = item ? (nonces[item.id] || 0) : 0;
  const url = item && item.filename
    ? `/api/dataset/${datasetId}/img/${encodeURIComponent(item.filename)}${nonce ? `?v=${nonce}` : ''}`
    : null;
  // Planned action, method-aware. A border ('crop') mark is always cropped (invents no
  // pixel) regardless of engine; every OTHER mark follows the selected engine — and
  // under Klein the on-subject ('review') mark becomes actionable instead of blocked.
  const kleinInpaintLabel = (n) => ({
    icon: '🎨',
    text: manual ? `Klein inpaint ${n} selected zone${n === 1 ? '' : 's'}` : ROUTE_LABEL.klein.text,
    cls: 'text-emerald-300',
  });
  let route;
  if (manual) {
    route = kleinSelected
      ? kleinInpaintLabel(regions.length)
      : { icon: '🖌', text: `Inpaint ${regions.length} selected zone${regions.length === 1 ? '' : 's'}`,
          cls: 'text-emerald-300' };
  } else if (item) {
    // effectiveRoute already folds in the per-image crop/inpaint choice ('crop' when the
    // user keeps the crop, else the crop-disabled fallback). Crop needs no engine; every
    // other route follows the selected engine, and Klein makes 'review' actionable.
    const r = effectiveRoute;
    route = (r !== 'crop' && kleinSelected && (r === 'lama' || r === 'review'))
      ? kleinInpaintLabel(regions.length)
      : ROUTE_LABEL[r];
  } else {
    route = null;
  }
  const oc = outcome ? OUTCOME[outcome] : null;
  const ocText = outcome === 'cleaned' && cleanDetail[item?.id]
    ? CLEAN_DETAIL_TEXT[cleanDetail[item.id]] || oc.text
    : oc?.text;
  const cleaning = working && workingKind === 'clean';   // navigation is held while true, so this always tracks `item`
  const restoring = working && workingKind === 'restore';
  // A real pixel edit ran (cleanDetail set on crop/inpaint/Klein) → a .orig exists to
  // undo. The "nothing to do" cleaned fallback sets no detail, so Restore stays hidden.
  const restorable = outcome === 'cleaned' && Boolean(cleanDetail[item?.id]);
  const showEditor = !(oc && oc.terminal) && !cleaning;
  const automaticLamaMissing = !kleinSelected && !manual && effectiveRoute === 'lama'
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

      {/* Image + editable correction-zone overlays.
          [container-type:size] turns this cell into a size-query container so the
          media below can cap its height to the cell (100cqh) — without it a portrait
          image keeps its natural height, overflows this flex-1 cell on a short mobile
          viewport, and the absolutely-positioned region box/handles paint over (and
          steal pointer events from) the controls bar underneath. */}
      <div onClick={(e) => e.stopPropagation()}
        className="flex-1 min-h-0 flex items-center justify-center p-3 [container-type:size]">
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
            />
          ) : (
            <div className="relative">
              <img src={url} alt={alt} className="block max-h-[min(70vh,calc(100cqh_-_1.5rem))] max-w-[min(92vw,100cqw)] select-none" />
              {(cleaning || restoring) && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-sm">
                  <span className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-black/75 text-amber-200 text-sm font-semibold">
                    <span aria-hidden className="w-4 h-4 rounded-full border-2 border-amber-200/40 border-t-amber-200 animate-spin" />
                    {cleaning ? 'Cleaning…' : 'Restoring…'}
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
          {/* Outcome badge lives HERE, off the photo — on the image it covered the
              exact pixels the user needs to judge after a clean. */}
          {oc && (
            <span className={`px-2 py-1 rounded-lg bg-black/60 font-semibold ${oc.cls}`}>
              {oc.icon} {ocText}
            </span>
          )}
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
            <span className="text-emerald-200/70 text-xs">
              · {kleinSelected ? 'one Klein inpaint per zone' : 'one composite LaMa pass'}
            </span>
          )}
          {engineMissing && (
            <span className="text-amber-300/90 text-xs">
              {kleinSelected
                ? '· Klein inpaint isn’t ready → start ComfyUI + install the Klein models (Setup ▸ ComfyUI)'
                : '· LaMa inpainting isn’t installed → install it from the 🧽 tools before Clean'}
            </span>
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

        {/* Crop-vs-inpaint for THIS image — shown only when a safe border crop exists.
            Crop cuts the band off (invents no pixel); Inpaint repaints the mark with the
            chosen engine. Overrides the persisted "Allow auto-crop" default for this one. */}
        {canCrop && !(oc && oc.terminal) && (
          <div role="group" aria-label="Removal method"
            className="flex items-center justify-center gap-1 text-xs">
            <span className="text-white/50">Method:</span>
            <div className="flex items-center rounded-lg border border-white/15 bg-white/5 p-0.5">
              <button type="button" aria-pressed={useCrop} disabled={working}
                onClick={() => setCropChoiceById((m) => ({ ...m, [item.id]: 'crop' }))}
                title="Crop the watermarked border off — invents no pixel (aspect ratio changes; ai-toolkit buckets it)."
                className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${useCrop
                  ? 'bg-sky-500/25 text-sky-100' : 'text-white/60 hover:text-white'}`}>
                ✂ Crop
              </button>
              <button type="button" aria-pressed={!useCrop} disabled={working}
                onClick={() => setCropChoiceById((m) => ({ ...m, [item.id]: 'inpaint' }))}
                title="Repaint the mark instead of cropping — keeps the full frame (uses the engine below)."
                className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${!useCrop
                  ? 'bg-amber-500/25 text-amber-100' : 'text-white/60 hover:text-white'}`}>
                🖌 Inpaint
              </button>
            </div>
          </div>
        )}

        {/* Inpaint engine for THIS image: Klein is the only one that can clean an
            on-subject ('review') mark, so it makes those actionable. Greyed until
            ComfyUI + the Klein models are ready, or while Crop is the chosen method
            (a crop uses no engine). */}
        <div role="group" aria-label="Inpaint method"
          className="flex items-center justify-center gap-1 text-xs">
          <span className={useCrop ? 'text-white/30' : 'text-white/50'}>Engine:</span>
          <div className="flex items-center rounded-lg border border-white/15 bg-white/5 p-0.5">
            <button type="button" aria-pressed={!kleinSelected} onClick={() => setMethod('lama')}
              disabled={working || useCrop}
              title="LaMa: fast, non-generative (border crop + small off-center marks). On-subject marks stay manual review."
              className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${!kleinSelected
                ? 'bg-amber-500/25 text-amber-100' : 'text-white/60 hover:text-white'}`}>
              LaMa
            </button>
            <button type="button" aria-pressed={kleinSelected} onClick={() => setMethod('klein')}
              disabled={working || useCrop || !kleinReady}
              title={kleinReady
                ? 'Klein: masked Flux.2 inpaint (crop-and-stitch). Cleans complex texture and marks ON the subject; only the mark changes.'
                : 'Klein inpaint needs ComfyUI running + the Klein models installed (Setup ▸ ComfyUI).'}
              className={`px-2.5 py-1 rounded-md font-semibold disabled:opacity-40 ${kleinSelected
                ? 'bg-amber-500/25 text-amber-100' : 'text-white/60 hover:text-white'}`}>
              Klein
            </button>
          </div>
        </div>

        <div className="flex gap-2 flex-wrap">
          {restorable ? (
            <button type="button" onClick={doRestore} disabled={working}
              title="Undo the clean — bring the watermarked original back so you can re-clean it (e.g. with the other engine) — shortcut r"
              className={`${btn} bg-sky-500/20 border border-sky-400/50 text-sky-100 hover:bg-sky-500/30`}>
              {restoring ? '↩ Restoring…' : <>↩ Restore original <kbd className="text-[10px] text-white/50">r</kbd></>}
            </button>
          ) : (
            <button type="button" onClick={doClean} disabled={cleanDisabled}
              title={outcome === 'cleaned'
                ? 'Already cleaned'
                : regions.length === 0
                  ? 'Add at least one correction zone before cleaning'
                  : engineMissing
                    ? (kleinSelected
                        ? 'Start ComfyUI and install the Klein models before cleaning with Klein'
                        : 'Install LaMa inpainting before cleaning manual zones')
                    : saveBlocked
                      ? 'Save correction zones before cleaning'
                : "Apply this image's watermark removal now (crop / inpaint / manual review) — shortcut c"}
              className={`${btn} bg-amber-500/20 border border-amber-400/50 text-amber-100 hover:bg-amber-500/30`}>
              {cleaning ? '🧽 Cleaning…' : <>🧽 Clean <kbd className="text-[10px] text-white/50">c</kbd></>}
            </button>
          )}
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
            ← → navigate · <kbd>c</kbd> clean · <kbd>r</kbd> restore · <kbd>d</kbd> dismiss · <kbd>x</kbd> reject · Esc close
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

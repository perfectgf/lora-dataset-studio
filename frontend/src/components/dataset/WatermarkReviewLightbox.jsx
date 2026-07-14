/**
 * Full-screen watermark REVIEW mode. Walk the flagged (🚩 detected) images one by one,
 * see the detected bbox drawn over the photo (crucial to spot a false positive at a
 * glance), and rule on each: 🧽 Clean (apply this image's routed removal now), ✓ Not a
 * watermark (dismiss — badge clears, future scans skip it), or ✕ Reject (drop it from
 * the kept set). Big tap targets + keyboard: ← → navigate, c/d/x act, Esc closes.
 *
 * The queue is FROZEN on open (a snapshot of the currently-detected images): actions
 * remove images from the live 'detected' set, but the filmstrip stays stable so the
 * user walks it once. Per-image outcomes are tracked locally; the parent refreshes the
 * grid counts underneath and shows a recap toast on close.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { displayLabel } from '../../utils/labels';

// The action Clean WILL take, per backend route (watermark_route in the payload).
const ROUTE_LABEL = {
  crop: { icon: '✂', text: 'Crop the watermarked border', cls: 'text-sky-300' },
  lama: { icon: '🖌', text: 'Inpaint the mark (LaMa)', cls: 'text-emerald-300' },
  review: { icon: '👁', text: 'On the subject — needs manual review', cls: 'text-amber-300' },
};

// Per-image outcome after an action. Terminal ones leave the 'detected' set (badge
// gone) and auto-advance; the rest keep the image flagged so the user can still reject.
const OUTCOME = {
  cleaned: { icon: '✨', text: 'Cleaned', cls: 'text-emerald-300', terminal: true },
  dismissed: { icon: '⊘', text: 'Marked “not a watermark”', cls: 'text-content-subtle', terminal: true },
  rejected: { icon: '✕', text: 'Rejected — removed from the set', cls: 'text-red-300', terminal: true },
  review: { icon: '👁', text: 'Left for manual review', cls: 'text-amber-300', terminal: false },
  skipped: { icon: '⬇', text: 'Skipped — inpainting not installed', cls: 'text-amber-300', terminal: false },
  failed: { icon: '⚠', text: 'Clean failed', cls: 'text-red-300', terminal: false },
};

const RECAP_ORDER = ['cleaned', 'dismissed', 'rejected', 'review', 'skipped', 'failed'];
const RECAP_WORD = { cleaned: 'cleaned', dismissed: 'dismissed', rejected: 'rejected',
  review: 'need review', skipped: 'skipped', failed: 'failed' };

export function buildWatermarkRecap(outcomes) {
  const c = {};
  for (const k of Object.values(outcomes || {})) c[k] = (c[k] || 0) + 1;
  const parts = RECAP_ORDER.filter((k) => c[k]).map((k) => `${c[k]} ${RECAP_WORD[k]}`);
  return parts.join(' · ');
}

export default function WatermarkReviewLightbox({ datasetId, queue, caps, nonces = {},
                                                  onClean, onDismiss, onReject, onClose }) {
  const [idx, setIdx] = useState(0);
  const [outcomes, setOutcomes] = useState({});   // id -> OUTCOME key
  const [working, setWorking] = useState(false);
  const [note, setNote] = useState(null);         // transient inline note {tone, text}
  const dialogRef = useRef(null);
  const workingRef = useRef(false);               // re-entrancy guard (double keypress)

  useFocusTrap(dialogRef, queue.length > 0);

  const total = queue.length;
  const item = idx >= 0 && idx < total ? queue[idx] : null;
  const outcome = item ? outcomes[item.id] : null;
  const allDone = total > 0 && Object.keys(outcomes).length >= total
    && queue.every((q) => outcomes[q.id]);

  const recap = useMemo(() => buildWatermarkRecap(outcomes), [outcomes]);
  const close = useCallback(() => onClose(recap), [onClose, recap]);

  const go = useCallback((delta) => {
    setNote(null);
    setIdx((i) => Math.min(total - 1, Math.max(0, i + delta)));
  }, [total]);
  const advance = useCallback(() => setIdx((i) => Math.min(total - 1, i + 1)), [total]);

  const run = useCallback(async (fn) => {
    if (!item || workingRef.current) return;
    workingRef.current = true;
    setWorking(true);
    setNote(null);
    try {
      const { key, note: n } = await fn(item);
      if (key) setOutcomes((m) => ({ ...m, [item.id]: key }));
      if (n) setNote(n);
      if (key && OUTCOME[key]?.terminal) advance();
    } finally {
      workingRef.current = false;
      setWorking(false);
    }
  }, [item, advance]);

  const doClean = useCallback(() => run(async (it) => {
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
    if (d.cropped || d.inpainted) return { key: 'cleaned' };
    if (d.needs_review) {
      return { key: 'review', note: { tone: 'warn',
        text: 'On the subject — auto crop/inpaint would damage the photo. Reject it or crop it manually.' } };
    }
    if (d.skipped) {
      return { key: 'skipped', note: { tone: 'warn',
        text: 'Off-center mark, but inpainting isn’t installed — install it or reject/crop this one.' } };
    }
    return { key: 'cleaned' };   // nothing to do reported → treat as resolved
  }), [run, onClean]);

  const doDismiss = useCallback(() => run(async (it) => {
    const d = await onDismiss(it.id);
    if (!d || d.ok === false) return { note: { tone: 'err', text: (d && d.error) || 'Could not dismiss' } };
    return { key: 'dismissed' };
  }), [run, onDismiss]);

  const doReject = useCallback(() => run(async (it) => {
    await onReject(it.id);
    return { key: 'rejected' };
  }), [run, onReject]);

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
  const bbox = item && Array.isArray(item.watermark_bbox) && item.watermark_bbox.length === 4
    ? item.watermark_bbox : null;
  const route = item ? ROUTE_LABEL[item.watermark_route] : null;
  const oc = outcome ? OUTCOME[outcome] : null;
  const showBox = bbox && !(oc && oc.terminal);
  const lamaMissing = item && item.watermark_route === 'lama' && caps && !caps.watermark_inpaint;

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
        <button type="button" onClick={close}
          title="Close (Esc)" aria-label="Close review"
          className="ml-auto w-9 h-9 rounded-full bg-white/10 hover:bg-white/20 text-white text-lg leading-none">✕</button>
      </div>

      {/* Image + detected bbox overlay */}
      <div onClick={(e) => e.stopPropagation()}
        className="flex-1 min-h-0 flex items-center justify-center p-3">
        {url ? (
          <div className="relative">
            <img src={url} alt={alt} className="block max-h-[70vh] max-w-[92vw] select-none" />
            {showBox && (
              <div aria-hidden
                className="absolute border-2 border-amber-400 bg-amber-400/15 pointer-events-none rounded-sm"
                style={{
                  left: `${bbox[0] * 100}%`, top: `${bbox[1] * 100}%`,
                  width: `${(bbox[2] - bbox[0]) * 100}%`, height: `${(bbox[3] - bbox[1]) * 100}%`,
                }} />
            )}
            {oc && (
              <div className="absolute inset-x-0 bottom-0 flex justify-center pb-2 pointer-events-none">
                <span className={`px-2 py-1 rounded-lg bg-black/75 text-xs font-semibold ${oc.cls}`}>
                  {oc.icon} {oc.text}
                </span>
              </div>
            )}
          </div>
        ) : (
          <span className="text-white/60 text-sm">image unavailable</span>
        )}
      </div>

      {/* Bottom: planned action, note, action buttons, nav, legend */}
      <div onClick={(e) => e.stopPropagation()}
        className="shrink-0 bg-black/70 border-t border-white/10 px-3 py-2.5 flex flex-col gap-2">

        <div className="flex items-center justify-center gap-2 text-sm flex-wrap">
          <span className="text-white/50">Planned:</span>
          {route ? (
            <span className={`font-semibold ${route.cls}`}>{route.icon} {route.text}</span>
          ) : (
            <span className="text-white/60">— unknown (missing box)</span>
          )}
          {lamaMissing && (
            <span className="text-amber-300/90 text-xs">· inpainting not installed → Clean will skip</span>
          )}
        </div>

        {note && (
          <p className={`text-center text-xs ${note.tone === 'err' ? 'text-red-300' : 'text-amber-300'}`}>
            {note.text}
          </p>
        )}

        <div className="flex gap-2 flex-wrap">
          <button type="button" onClick={doClean} disabled={working}
            title="Apply this image's watermark removal now (crop / inpaint / manual review) — shortcut c"
            className={`${btn} bg-amber-500/20 border border-amber-400/50 text-amber-100 hover:bg-amber-500/30`}>
            🧽 Clean <kbd className="text-[10px] text-white/50">c</kbd>
          </button>
          <button type="button" onClick={doDismiss} disabled={working}
            title="This is NOT a watermark (false positive) — clears the flag, future scans skip it — shortcut d"
            className={`${btn} bg-emerald-600/20 border border-emerald-400/40 text-emerald-100 hover:bg-emerald-600/30`}>
            ✓ Not a watermark <kbd className="text-[10px] text-white/50">d</kbd>
          </button>
          <button type="button" onClick={doReject} disabled={working}
            title="Reject this image — it leaves the kept set (for watermarks that can't be recovered) — shortcut x"
            className={`${btn} bg-red-600/20 border border-red-400/40 text-red-100 hover:bg-red-600/30`}>
            ✕ Reject <kbd className="text-[10px] text-white/50">x</kbd>
          </button>
        </div>

        <div className="flex items-center justify-between gap-2">
          <button type="button" onClick={() => go(-1)} disabled={idx <= 0}
            title="Previous (←)" aria-label="Previous image"
            className="px-3 min-h-[2.5rem] rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm disabled:opacity-30">← Prev</button>
          <span className="text-white/40 text-[11px] text-center hidden sm:block">
            ← → navigate · <kbd>c</kbd> clean · <kbd>d</kbd> dismiss · <kbd>x</kbd> reject · Esc close
          </span>
          {allDone ? (
            <button type="button" onClick={close}
              className="px-4 min-h-[2.5rem] rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-semibold">
              Done ✓
            </button>
          ) : (
            <button type="button" onClick={() => go(1)} disabled={idx >= total - 1}
              title="Next (→)" aria-label="Next image"
              className="px-3 min-h-[2.5rem] rounded-lg bg-white/10 hover:bg-white/20 text-white text-sm disabled:opacity-30">Next →</button>
          )}
        </div>
      </div>
    </div>
  );
}

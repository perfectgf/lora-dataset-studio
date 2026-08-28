/**
 * 🔤 Find text — the dataset's launch window. FULL PARITY with the bank's
 * dialog is the point (maintainer's rule: shared features ship at full parity
 * on both surfaces): the same two dials — "try on a sample first" and the
 * stored Sensitivity — priced with the same measured counts, before anything
 * runs. The pass itself is the same engine, the same merge rules, the same
 * funnel; this window only decides how much of it runs and at what floor.
 *
 * The scope story differs from the bank BY MECHANIC, not by behaviour: a
 * dataset pass has always read the KEPT images (its watermark scan does the
 * same), so instead of the bank's pile picker this window says that out
 * loud, with the number it means.
 */
import { useEffect, useRef, useState } from 'react';
import { putJson } from '../../api/fetchClient';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import DatasetTextScanPreview from './DatasetTextScanPreview.jsx';

export default function TextScanDialog({
  onClose, onLaunch, toRead = 0, rereadable = 0, sensitivity = 0.5, live = false,
  datasetId,
}) {
  const [redo, setRedo] = useState(false);
  const [sampleOn, setSampleOn] = useState(false);
  const [sampleSize, setSampleSize] = useState(20);
  const [level, setLevel] = useState(
    Number.isFinite(Number(sensitivity)) ? Number(sensitivity) : 0.5);
  const [busy, setBusy] = useState(false);
  /* A run happened FROM THIS WINDOW. The window then stays open showing the
     flagged pages with their zones (the strip below) — closing on launch would
     hide exactly what a sample run was started to show. Only relabels the exit
     and arms the strip's "nothing found" line; every dial keeps working, so
     adjust-and-rerun needs no reopen. Bank parity: its window does the same. */
  const [ran, setRan] = useState(false);
  const dialogRef = useRef(null);
  useFocusTrap(dialogRef, true);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === 'Escape' && !busy) onClose();
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [busy, onClose]);

  const pool = redo ? rereadable : toRead;
  const willRead = sampleOn
    ? Math.min(pool, Math.max(1, Math.round(Number(sampleSize) || 20)))
    : pool;

  const saveSensitivity = async (value) => {
    try {
      await putJson('/api/settings', { config: { text_scan: { score_min: value } } });
    } catch { /* the run still uses the stored value */ }
  };

  const launch = async () => {
    if (busy || willRead === 0) return;
    setBusy(true);
    try {
      await onLaunch({
        rescan: redo,
        ...(sampleOn
          ? { limit: Math.max(1, Math.min(10000, Math.round(Number(sampleSize) || 20))) }
          : {}),
      });
      setRan(true);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[9991] flex items-center justify-center bg-black/80 p-3 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose(); }}>
      <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Find text"
        className="w-full max-w-lg space-y-3 overflow-y-auto rounded-xl border border-border bg-surface p-4 shadow-xl"
        style={{ maxHeight: 'min(38rem, calc(100vh - 2rem))' }}>
        <h2 className="m-0 text-base font-bold text-content">🔤 Find text</h2>
        <p className="m-0 text-[11px] leading-snug text-content-subtle">
          Reads burned-in text — speech bubbles, subtitles, captions, sound effects —
          and marks each zone so 🧽 Clean can repaint it. CPU only, never the GPU.
        </p>
        <p className="m-0 rounded-md border border-border bg-surface-raised px-2 py-1.5 text-[11px] leading-snug text-content-muted">
          This run reads the <span className="font-semibold text-content">kept</span> images
          that still need reading — <span className="font-semibold text-content">{toRead}</span> waiting.
          Images you dismissed as “not watermarked” keep their ruling and are never re-examined.
        </p>
        <label className="flex items-start gap-2 text-[11px] text-content-subtle">
          <input type="checkbox" className="mt-0.5" checked={redo}
            onChange={(e) => setRedo(e.target.checked)} disabled={live || busy} />
          <span>
            <span className="font-medium text-content">Also re-read images that were already scanned</span>
            {' '}({rereadable} kept). With the sample below ticked, this re-reads the
            SAME first images — the way to judge a new sensitivity on known pages.
          </span>
        </label>
        <div className="space-y-2 rounded-md border border-border bg-surface-raised p-2">
          <p className="m-0 text-[11px] font-semibold uppercase tracking-wide text-content-muted">
            Options for this run
          </p>
          <label className="flex items-start gap-2 text-[11px] text-content-subtle">
            <input type="checkbox" className="mt-0.5" checked={sampleOn}
              onChange={(e) => setSampleOn(e.target.checked)} disabled={live || busy} />
            <span>
              <span className="font-medium text-content">Try on a sample first</span>
              {' — read only the first '}
              <input type="number" min="1" max="10000" value={sampleSize}
                onChange={(e) => setSampleSize(e.target.value)}
                disabled={live || busy || !sampleOn} aria-label="Sample size"
                className="mx-1 w-16 rounded border border-border bg-app px-1 py-0.5 text-content" />
              {' images. Judge the zones in the 🔍 review of flagged images, then '}
              {'run again for the rest.'}
            </span>
          </label>
          <label className="block text-[11px] text-content-subtle">
            <span className="font-medium text-content">Sensitivity</span>
            {' — the OCR confidence a line needs to become a zone. Lower catches '}
            {'fainter or stylised lettering, at the cost of false zones. Stored: '}
            {'the bank scan reads the same value.'}
            <span className="mt-1 flex items-center gap-2">
              <input type="range" min="0.30" max="0.70" step="0.05" value={level}
                disabled={live || busy} aria-label="Text sensitivity"
                onChange={(e) => setLevel(Number(e.target.value))}
                onMouseUp={() => saveSensitivity(level)}
                onTouchEnd={() => saveSensitivity(level)}
                onKeyUp={() => saveSensitivity(level)}
                className="w-40" />
              <span className="tabular-nums text-content">{level.toFixed(2)}</span>
              <span className="text-content-subtle">(default 0.50)</span>
            </span>
          </label>
        </div>
        {/* The run's RESULT, in the window that launched it: the flagged pages
            with their zones, POLLED while the scan runs so they fill in as
            text is found (the pass commits per image). It used to read the
            workspace payload instead, which only refreshes when the run
            returns — a whole 106-page scan with the strip sitting on
            "nothing flagged yet" the entire time. Bank parity: same strip,
            same poll, off this surface's own /text/preview. */}
        <DatasetTextScanPreview datasetId={datasetId} live={busy || live}
          emptyLine={ran ? 'No text found on the scanned images.' : null} />
        <div className="flex items-center justify-end gap-2 pt-1">
          <button type="button" onClick={onClose} disabled={busy}
            className="rounded-lg border border-border bg-surface px-3 py-1.5 text-sm text-content disabled:opacity-40">
            {ran ? 'Close' : 'Cancel'}
          </button>
          <button type="button" onClick={launch}
            disabled={busy || live || willRead === 0}
            title={willRead === 0 ? 'Nothing left to read in this scope.' : undefined}
            className="rounded-lg bg-amber-500/90 px-3 py-1.5 text-sm font-bold text-black disabled:opacity-40">
            {busy ? 'Scanning…' : `Scan ${willRead} image${willRead === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>
    </div>
  );
}

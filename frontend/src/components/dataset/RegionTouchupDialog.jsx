/**
 * 🖌 Region touch-up — paint a mask, type a removal prompt, Klein inpaints
 * only those pixels on the FULL image (same contract as the Flux.2 Klein
 * inpaint graph). Does NOT flag the image as a watermark.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { HelpBadge } from '../../help/HelpMode';
import { localEngineUnavailableReason } from '../../utils/localEngineReason.js';
import KleinModelSetting from '../shared/KleinModelSetting';
import InpaintBrushEditor, { maskPngFromCanvas } from './InpaintBrushEditor';
import { useCapabilities } from '../../context/CapabilitiesContext';

const CHIPS = [
  'remove necklace',
  'remove earrings',
  'remove jewelry',
  'remove skin blemish',
  'remove makeup',
];

const btn = 'min-h-11 rounded-lg border border-white/20 bg-white/10 px-3 text-xs '
  + 'font-semibold text-white hover:bg-white/20 disabled:opacity-40';

export default function RegionTouchupDialog({
  datasetId, image, nonce = 0, onApply, onRestore, onClose, busy = false, busyReason = null,
}) {
  const { caps } = useCapabilities();
  const [prompt, setPrompt] = useState('');
  const [working, setWorking] = useState(false);
  const [applied, setApplied] = useState(() => Boolean(image?.has_region_touchup));
  const [error, setError] = useState(null);
  const [painted, setPainted] = useState(false);
  const [eraser, setEraser] = useState(false);
  const [brushCss, setBrushCss] = useState(24);
  const canvasRef = useRef(null);
  const dialogRef = useRef(null);

  useFocusTrap(dialogRef, true);

  useEffect(() => {
    setApplied(Boolean(image?.has_region_touchup));
  }, [image?.id, image?.has_region_touchup]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (!working) onClose?.();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onClose, working]);

  const kleinReady = caps?.watermark_klein !== false;
  const kleinReason = kleinReady ? null : localEngineUnavailableReason('klein', caps);
  const refused = (busy && !working) ? busyReason : null;
  const applying = working || busy;
  const canApply = kleinReady && !applying && painted && prompt.trim() && !refused;

  const src = `/api/dataset/${datasetId}/img/${encodeURIComponent(image?.filename || '')}`
    + (nonce ? `?v=${nonce}` : '');

  const clearMask = useCallback(() => {
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    setPainted(false);
  }, []);

  const apply = useCallback(async () => {
    if (!canApply || !onApply) return;
    const mask = canvasRef.current ? maskPngFromCanvas(canvasRef.current) : null;
    if (!mask) {
      setError('Paint the area to change first');
      setPainted(false);
      return;
    }
    setWorking(true);
    setError(null);
    try {
      const result = await onApply(mask, prompt.trim());
      if (!result || result.ok === false) {
        setError(result?.error || result?.detail || refused || 'Touch-up failed');
        return;
      }
      setApplied(true);
      clearMask();
    } catch (e) {
      setError(e?.message || 'Touch-up failed');
    } finally {
      setWorking(false);
    }
  }, [canApply, clearMask, onApply, prompt, refused]);

  const restore = useCallback(async () => {
    if (!onRestore || applying) return;
    setWorking(true);
    setError(null);
    try {
      const result = await onRestore();
      if (!result || result.ok === false) {
        setError(result?.error || result?.detail || 'Could not reset the touch-up');
        return;
      }
      setApplied(Boolean(result.has_region_touchup));
    } catch (e) {
      setError(e?.message || 'Could not reset the touch-up');
    } finally {
      setWorking(false);
    }
  }, [applying, onRestore]);

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Touch up a region"
      className="fixed inset-0 z-[9998] flex flex-col bg-black">
      <div className="flex flex-wrap gap-2 items-center px-3 py-2 border-b shrink-0 border-white/10">
        <span className="text-sm font-semibold text-white">🖌 Touch up</span>
        <span className="max-w-[14rem] truncate text-xs text-white/60" title={image?.filename}>
          {image?.filename}
        </span>
        <HelpBadge topic="action-dataset-region-touchup" className="self-center" />
        <button type="button" onClick={onClose} disabled={working}
          title="Close" aria-label="Close the touch-up editor"
          className="ml-auto w-11 h-11 text-lg leading-none text-white rounded-full bg-white/10 hover:bg-white/20 disabled:opacity-40">
          ✕
        </button>
      </div>

      <div className="flex min-h-0 flex-1 items-center justify-center p-2 [container-type:size]">
        <InpaintBrushEditor
          src={src}
          alt={image?.filename || `Dataset image ${image?.id}`}
          disabled={applying}
          eraser={eraser}
          brushCss={brushCss}
          onDirty={setPainted}
          canvasRef={canvasRef}
        />
      </div>

      <div className="shrink-0 space-y-2 border-t border-white/10 bg-black/70 px-3 py-2.5">
        <p className="text-xs text-center text-white/70">
          Paint over what you want gone, then say what to remove. Klein sees the whole
          photo and only changes the painted pixels.
        </p>

        <div className="flex flex-wrap gap-2 justify-center items-center">
          <button type="button" aria-pressed={!eraser} onClick={() => setEraser(false)}
            disabled={applying}
            className={!eraser
              ? 'px-3 text-xs font-semibold text-sky-100 rounded-lg border border-sky-300 min-h-11 bg-sky-500/25'
              : btn}>
            Brush
          </button>
          <button type="button" aria-pressed={eraser} onClick={() => setEraser(true)}
            disabled={applying}
            className={eraser
              ? 'px-3 text-xs font-semibold text-sky-100 rounded-lg border border-sky-300 min-h-11 bg-sky-500/25'
              : btn}>
            Eraser
          </button>
          <label className="flex gap-2 items-center text-xs text-white/80">
            Size
            <input type="range" min="6" max="80" value={brushCss} disabled={applying}
              onChange={(e) => setBrushCss(Number(e.target.value))}
              className="w-28 accent-sky-300" />
          </label>
          <button type="button" onClick={clearMask} disabled={applying || !painted}
            className={btn}>
            Clear
          </button>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-1.5">
          {CHIPS.map((chip) => (
            <button key={chip} type="button" disabled={applying}
              onClick={() => setPrompt(chip)}
              className={`min-h-9 rounded-full border px-2.5 text-[11px] font-semibold
                ${prompt === chip
                  ? 'border-sky-300 bg-sky-500/25 text-sky-100'
                  : 'border-white/20 bg-white/5 text-white/80 hover:bg-white/15'}`}>
              {chip}
            </button>
          ))}
        </div>

        <label className="block mx-auto max-w-xl">
          <span className="sr-only">Removal prompt</span>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            disabled={applying}
            rows={2}
            maxLength={500}
            placeholder="remove necklace"
            className="px-3 py-2 w-full text-sm text-white rounded-lg border resize-none border-white/20 bg-black/40 placeholder:text-white/35 disabled:opacity-50"
          />
        </label>

        <KleinModelSetting datasetId={datasetId} className="mx-auto max-w-xl text-center" />

        {!kleinReady && (
          <p role="note" className="text-xs text-center text-amber-200">
            ⚠ {kleinReason || 'Klein is not ready — this tool cannot use LaMa (it smears the patch).'}
          </p>
        )}
        {refused && (
          <p role="note" className="text-xs text-center text-amber-200">⚠ {refused}</p>
        )}
        {error && (
          <p role="alert" className="text-xs text-center text-rose-300">⚠ {error}</p>
        )}
        {applied && !error && (
          <p className="text-xs text-center text-emerald-300">
            Reset restores the file from before any touch-up on this image. Captions are not rewritten; re-caption if they named what you removed.
          </p>
        )}

        <div className="flex flex-wrap gap-2 justify-center items-center">
          {applied && (
            <button type="button" onClick={restore} disabled={applying}
              className="px-4 text-xs font-semibold text-white rounded-lg border min-h-11 border-white/25 bg-white/10 hover:bg-white/20 disabled:opacity-40">
              {working ? 'Resetting…' : '↩ Reset'}
            </button>
          )}
          <button type="button" onClick={apply} disabled={!canApply}
            title={!kleinReady ? (kleinReason || 'Klein is not ready')
              : !painted ? 'Paint the area to change first'
                : !prompt.trim() ? 'Write what to remove'
                  : refused || 'Klein inpaints the painted pixels'}
            className="px-4 text-xs font-semibold text-emerald-100 rounded-lg border min-h-11 border-emerald-400/60 bg-emerald-500/20 hover:bg-emerald-500/30 disabled:opacity-40">
            {working ? 'Applying…' : '🖌 Apply'}
          </button>
        </div>
      </div>
    </div>
  );
}

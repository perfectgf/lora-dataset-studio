/**
 * 🖌 Region touch-up — paint a mask, type a removal prompt, Klein inpaints
 * only those pixels on the FULL image (same contract as the Flux.2 Klein
 * inpaint graph). Does NOT flag the image as a watermark.
 *
 * The control panel uses the same geometry as the dataset lightbox: a side
 * rail when the photo is height-limited (portrait on a wide window), a bottom
 * bar when it is width-limited, and a drawer on a phone so the picture is
 * not left a few dozen pixels tall.
 */
import { useCallback, useEffect, useId, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { HelpBadge } from '../../help/HelpMode';
import { localEngineUnavailableReason } from '../../utils/localEngineReason.js';
import KleinModelSetting from '../shared/KleinModelSetting';
import InpaintBrushEditor, { maskPngFromCanvas } from './InpaintBrushEditor';
import {
  decideActionPlacement, rememberImageRatio, readImageRatio,
} from './lightboxActionPlacement';
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

function ControlsHost({ sheet, open, panelId, closeRef, onDone, children }) {
  if (!sheet) return children;
  if (!open) return null;
  return (
    <div id={panelId} role="dialog" aria-label="Touch-up controls"
      className="absolute inset-x-0 bottom-0 z-30 flex max-h-[70vh] flex-col
        rounded-t-2xl border-t border-white/15 bg-neutral-950 shadow-2xl">
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-white/10 px-4 py-2">
        <h2 className="min-w-0 truncate text-sm font-semibold text-white">Touch-up controls</h2>
        <button type="button" ref={closeRef} onClick={onDone}
          title="Close the controls (Esc)" aria-label="Close the controls"
          className="min-h-9 shrink-0 rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold text-white hover:bg-white/20">
          Done
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">{children}</div>
    </div>
  );
}

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
  const [controlsOpen, setControlsOpen] = useState(false);
  const canvasRef = useRef(null);
  const dialogRef = useRef(null);
  const controlsBtnRef = useRef(null);
  const panelCloseRef = useRef(null);
  const panelId = `touchup-controls${useId()}`;
  const imageId = image?.id;
  const [ratio, setRatio] = useState(() => readImageRatio(imageId));
  const [placement, setPlacement] = useState(() => decideActionPlacement({
    viewportWidth: typeof window === 'undefined' ? 0 : window.innerWidth,
    viewportHeight: typeof window === 'undefined' ? 0 : window.innerHeight,
    imageWidth: ratio?.imageWidth,
    imageHeight: ratio?.imageHeight,
  }));

  useFocusTrap(dialogRef, true);

  useEffect(() => {
    setApplied(Boolean(image?.has_region_touchup));
  }, [image?.id, image?.has_region_touchup]);

  useEffect(() => { setRatio(readImageRatio(imageId)); }, [imageId]);

  const kleinReady = caps?.watermark_klein !== false;
  const kleinReason = kleinReady ? null : localEngineUnavailableReason('klein', caps);
  const refused = (busy && !working) ? busyReason : null;
  const applying = working || busy;
  const canApply = kleinReady && !applying && painted && prompt.trim() && !refused;
  const rail = placement === 'rail';
  const sheet = placement === 'sheet';
  const panelOpen = sheet && controlsOpen;
  const stacked = rail || sheet;

  const closeControls = useCallback(() => {
    setControlsOpen(false);
    controlsBtnRef.current?.focus();
  }, []);

  useEffect(() => { if (panelOpen) panelCloseRef.current?.focus(); }, [panelOpen]);

  const onNaturalSize = useCallback((w, h) => {
    rememberImageRatio(imageId, w, h);
    setRatio((prev) => (prev && prev.imageWidth === w && prev.imageHeight === h
      ? prev
      : { imageWidth: w, imageHeight: h }));
  }, [imageId]);

  useEffect(() => {
    let frame = 0;
    const applyPlacement = () => setPlacement((current) => decideActionPlacement({
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      imageWidth: ratio?.imageWidth,
      imageHeight: ratio?.imageHeight,
      current,
      locked: applying,
    }));
    applyPlacement();
    const onResize = () => {
      if (frame) return;
      frame = requestAnimationFrame(() => { frame = 0; applyPlacement(); });
    };
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [ratio, applying]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      e.preventDefault();
      e.stopImmediatePropagation();
      if (panelOpen) { closeControls(); return; }
      if (!working) onClose?.();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [closeControls, onClose, panelOpen, working]);

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

  const controls = (
    <div className={rail
      ? 'flex w-[17rem] shrink-0 flex-col gap-2 overflow-y-auto border-l border-white/10 bg-black/70 px-3 py-2.5'
      : sheet
        ? 'flex flex-col gap-2 px-4 py-3'
        : 'shrink-0 space-y-2 border-t border-white/10 bg-black/70 px-3 py-2.5'}>
      <p className={`text-xs text-white/70 ${stacked ? 'text-left' : 'text-center'}`}>
        Paint over what you want gone, then say what to remove. Klein sees the whole
        photo and only changes the painted pixels.
      </p>

      <div className={`flex flex-wrap items-center gap-2 ${stacked ? '' : 'justify-center'}`}>
        <button type="button" aria-pressed={!eraser} onClick={() => setEraser(false)}
          disabled={applying}
          className={!eraser
            ? 'min-h-11 rounded-lg border border-sky-300 bg-sky-500/25 px-3 text-xs font-semibold text-sky-100'
            : btn}>
          Brush
        </button>
        <button type="button" aria-pressed={eraser} onClick={() => setEraser(true)}
          disabled={applying}
          className={eraser
            ? 'min-h-11 rounded-lg border border-sky-300 bg-sky-500/25 px-3 text-xs font-semibold text-sky-100'
            : btn}>
          Eraser
        </button>
        <label className="flex items-center gap-2 text-xs text-white/80">
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

      <div className={`flex flex-wrap items-center gap-1.5 ${stacked ? '' : 'justify-center'}`}>
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

      <label className={stacked ? 'block w-full' : 'mx-auto block max-w-xl'}>
        <span className="sr-only">Removal prompt</span>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          disabled={applying}
          rows={2}
          maxLength={500}
          placeholder="remove necklace"
          className="w-full resize-none rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-white/35 disabled:opacity-50"
        />
      </label>

      <KleinModelSetting datasetId={datasetId}
        className={stacked ? 'w-full' : 'mx-auto max-w-xl text-center'} />

      {!kleinReady && (
        <p role="note" className={`text-xs text-amber-200 ${stacked ? '' : 'text-center'}`}>
          ⚠ {kleinReason || 'Klein is not ready — this tool cannot use LaMa (it smears the patch).'}
        </p>
      )}
      {refused && (
        <p role="note" className={`text-xs text-amber-200 ${stacked ? '' : 'text-center'}`}>⚠ {refused}</p>
      )}
      {error && (
        <p role="alert" className={`text-xs text-rose-300 ${stacked ? '' : 'text-center'}`}>⚠ {error}</p>
      )}
      {applied && !error && (
        <p className={`text-xs text-emerald-300 ${stacked ? '' : 'text-center'}`}>
          Reset restores the file from before any touch-up on this image. Captions are not rewritten; re-caption if they named what you removed.
        </p>
      )}

      <div className={`flex flex-wrap items-center gap-2 ${stacked ? '' : 'justify-center'}`}>
        {applied && (
          <button type="button" onClick={restore} disabled={applying}
            className="min-h-11 rounded-lg border border-white/25 bg-white/10 px-4 text-xs font-semibold text-white hover:bg-white/20 disabled:opacity-40">
            {working ? 'Resetting…' : '↩ Reset'}
          </button>
        )}
        <button type="button" onClick={apply} disabled={!canApply}
          title={!kleinReady ? (kleinReason || 'Klein is not ready')
            : !painted ? 'Paint the area to change first'
              : !prompt.trim() ? 'Write what to remove'
                : refused || 'Klein inpaints the painted pixels'}
          className="min-h-11 rounded-lg border border-emerald-400/60 bg-emerald-500/20 px-4 text-xs font-semibold text-emerald-100 hover:bg-emerald-500/30 disabled:opacity-40">
          {working ? 'Applying…' : '🖌 Apply'}
        </button>
      </div>
    </div>
  );

  return (
    <div ref={dialogRef} role="dialog" aria-modal="true" aria-label="Touch up a region"
      className="fixed inset-0 z-[9998] flex flex-col bg-black">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-white/10 px-3 py-2">
        <span className="text-sm font-semibold text-white">🖌 Touch up</span>
        <span className="max-w-[14rem] truncate text-xs text-white/60" title={image?.filename}>
          {image?.filename}
        </span>
        <HelpBadge topic="action-dataset-region-touchup" className="self-center" />
        <button type="button" onClick={onClose} disabled={working}
          title="Close" aria-label="Close the touch-up editor"
          className="ml-auto h-11 w-11 rounded-full bg-white/10 text-lg leading-none text-white hover:bg-white/20 disabled:opacity-40">
          ✕
        </button>
      </div>

      <div className={`relative flex min-h-0 flex-1 ${rail ? 'flex-row' : 'flex-col'}`}>
        <div className={`relative flex min-h-0 min-w-0 flex-1 items-center justify-center p-2 [container-type:size] ${
          sheet ? 'pb-16' : ''}`}>
          <InpaintBrushEditor
            src={src}
            alt={image?.filename || `Dataset image ${image?.id}`}
            disabled={applying}
            eraser={eraser}
            brushCss={brushCss}
            onDirty={setPainted}
            onNaturalSize={onNaturalSize}
            canvasRef={canvasRef}
          />
          {sheet && (
            <button type="button" ref={controlsBtnRef}
              onClick={() => setControlsOpen((open) => !open)}
              aria-expanded={controlsOpen} aria-controls={panelId}
              aria-label="Touch-up controls — brush, prompt, apply"
              title="Touch-up controls — brush, prompt, apply"
              className="absolute bottom-3 left-1/2 z-20 flex min-h-11 -translate-x-1/2 items-center
                rounded-full border border-white/25 bg-black/75 px-5 py-2 text-sm font-semibold
                text-white shadow-lg hover:bg-black/90">
              <span aria-hidden="true">☰&nbsp;</span>Controls
            </button>
          )}
        </div>
        <ControlsHost sheet={sheet} open={panelOpen} panelId={panelId}
          closeRef={panelCloseRef} onDone={closeControls}>
          {controls}
        </ControlsHost>
      </div>
    </div>
  );
}

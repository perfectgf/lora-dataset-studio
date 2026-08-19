/**
 * ✦ Repair — draw a zone on ONE image and repaint just that zone from your own
 * words. Everything outside it comes back byte-identical.
 *
 * WHY IT EXISTS HERE, on a GENERATED image. Asked for by .samexit on Discord:
 * "add the inpaint feature immediately after the first generation, to avoid
 * having to completely regenerate the image just to fix a small detail". Before
 * this, a stray finger or an unwanted object meant throwing the whole picture
 * away and rolling the dice again — the only prompted lane (✦ Edit) re-renders
 * everything and gives you a different image.
 *
 * TWO SHAPES, ONE BUTTON. A box is right for a mark in a corner: the server
 * crops a square around it and works at ~1 MP, which is quick and bounds VRAM.
 * A BRUSH is right for a necklace or a pair of glasses, where a rectangle would
 * hand the model a square full of face it was not asked to touch — that gesture
 * sends the whole frame with a painted mask, so Klein reconstructs while seeing
 * the picture around it. (Brush contributed by OneCodingDude on GitHub, PR #37; it
 * lives inside this dialog rather than behind a THIRD button in an action bar
 * that already carries two doors to the same place.)
 *
 * It reuses the DATASET's zone editor rather than growing a second one: the same
 * drawing gesture, the same normalised boxes, the same thing the server expects.
 * The only new part is the sentence you type and the call it posts.
 *
 * The caller owns the request (`onSubmit`), because the two surfaces address
 * their image differently — a generated image by its id, a dataset image by
 * dataset+image. This dialog only knows how to draw a box and take a sentence.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import WatermarkRegionEditor from '../dataset/WatermarkRegionEditor';
import InpaintBrushEditor, { maskPngFromCanvas } from './InpaintBrushEditor';

export default function RepairDialog({ open, src, alt = 'image', onClose, onSubmit, onUndo = null }) {
  const dialogRef = useRef(null);
  const [regions, setRegions] = useState([]);
  const [addMode, setAddMode] = useState(true);
  const [selected, setSelected] = useState(null);
  const [prompt, setPrompt] = useState('');
  /* 'box' stays the default: it is the cheaper lane, and it is the gesture that
     shipped — someone who learned it must not find it replaced. */
  const [mode, setMode] = useState('box');
  const [brushSize, setBrushSize] = useState(28);
  const [eraser, setEraser] = useState(false);
  const [painted, setPainted] = useState(false);
  const brushCanvasRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  /* The dialog STAYS OPEN on success. An inpaint is a dice roll: the gesture
     people actually make is look → not right → change the sentence → go again.
     Closing on success forced a full round trip for every attempt. `bust` is
     what makes the new pixels visible — the file is overwritten in place, so
     the URL does not move. (Asked for on Discord right after ✦ Repair shipped.) */
  const [done, setDone] = useState(false);
  const [bust, setBust] = useState(0);
  /* Computed BEFORE the early return and before `run`, which closes over it:
     hooks may not sit behind a conditional, and a `ready` declared after the
     `if (!open)` would be a stale capture inside the callback. */
  const brush = mode === 'brush';
  const ready = !!prompt.trim() && (brush ? painted : regions.length > 0);
  useFocusTrap(dialogRef, open);

  // A fresh dialog never inherits the previous image's zones or sentence.
  useEffect(() => {
    if (!open) return;
    setRegions([]); setPrompt(''); setError(null); setSelected(null); setAddMode(true);
    setDone(false); setBust(0); setMode('box'); setEraser(false); setPainted(false);
  }, [open, src]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, busy, onClose]);

  const run = useCallback(async () => {
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    try {
      /* The mask is read from the canvas at SUBMIT time, not tracked in state:
         it is up to a few megabytes of pixels, and keeping it in React would
         re-encode it on every stroke for a value only this call ever reads. */
      const mask = mode === 'brush' ? maskPngFromCanvas(brushCanvasRef.current) : null;
      if (mode === 'brush' && !mask) {
        setError('Paint the area to repair first.');
        return;
      }
      const d = await onSubmit(mode === 'brush'
        ? { mask, prompt: prompt.trim() }
        : { boxes: regions, prompt: prompt.trim() });
      if (!d || d.ok === false) {
        setError((d && (d.error?.detail || d.error)) || 'The repair failed.');
        return;
      }
      setDone(true);
      setBust((n) => n + 1);
    } catch (e) {
      setError(e?.message || 'The repair failed.');
    } finally {
      setBusy(false);
    }
  }, [ready, mode, regions, prompt, busy, onSubmit]);

  const undo = useCallback(async () => {
    if (!onUndo || busy) return;
    setBusy(true);
    setError(null);
    try {
      const d = await onUndo();
      if (!d || d.ok === false) {
        setError((d && (d.error?.detail || d.error)) || 'Could not put the previous image back.');
        return;
      }
      if (d.undone === false) {
        setError('There was nothing to undo — this image has not been repaired since.');
        return;
      }
      setDone(false);
      setBust((n) => n + 1);
    } catch (e) {
      setError(e?.message || 'Could not put the previous image back.');
    } finally {
      setBusy(false);
    }
  }, [onUndo, busy]);

  if (!open) return null;

  return (
    /* STOPS ITS OWN CLICKS. Every host mounts this INSIDE its overlay so it
       inherits the stacking context — and those overlays close when you click
       their backdrop. Without this, clicking the description field (which has no
       handler of its own to swallow the event) bubbled all the way up and shut
       the whole review down: reported from the watermark review, where it threw
       the user back to the dataset. Defended here rather than in each host, so a
       fourth surface cannot reintroduce it. */
    <div className="fixed inset-0 z-[9998] flex flex-col bg-black p-3 sm:p-4"
      role="dialog" aria-modal="true" aria-label="Repair an area of this image"
      onClick={(e) => e.stopPropagation()}
      ref={dialogRef}>
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-white">✦ Repair an area</span>
        <span className="text-[0.6875rem] text-white/60">
          Draw the zone, say what should be there. Everything outside it is left untouched.
        </span>
        <button type="button" onClick={() => onClose()} disabled={busy}
          aria-label="Close" className="ml-auto h-8 w-8 rounded-full bg-white/10 text-white hover:bg-white/20 disabled:opacity-40">
          ×
        </button>
      </div>

      {/* ONE row, and it only appears where it means something. The brush
          controls are the brush's own; showing them next to a box editor that
          cannot use them would be the clutter this dialog exists to avoid. */}
      <div className="mb-2 flex flex-wrap items-center gap-2 text-[0.6875rem] text-white/70">
        <div role="group" aria-label="Repair shape"
          className="flex items-center rounded-lg border border-white/15 bg-white/5 p-0.5">
          <button type="button" aria-pressed={!brush} disabled={busy}
            onClick={() => setMode('box')}
            title="Draw a rectangle — quickest, and the model works on a crop of it"
            className={`rounded-md px-2.5 py-1 font-semibold disabled:opacity-40 ${!brush
              ? 'bg-sky-500/25 text-sky-100' : 'text-white/60 hover:text-white'}`}>
            ▭ Box
          </button>
          <button type="button" aria-pressed={brush} disabled={busy}
            onClick={() => setMode('brush')}
            title="Paint over the thing to remove — the model sees the whole picture, better for jewelry, glasses or straps"
            className={`rounded-md px-2.5 py-1 font-semibold disabled:opacity-40 ${brush
              ? 'bg-pink-500/25 text-pink-100' : 'text-white/60 hover:text-white'}`}>
            🖌 Brush
          </button>
        </div>
        {brush && (
          <>
            <label className="flex items-center gap-1.5">
              Size
              <input type="range" min="8" max="120" step="2" value={brushSize}
                disabled={busy} aria-label="Brush size"
                onChange={(e) => setBrushSize(Number(e.target.value))}
                className="w-24 accent-pink-400" />
            </label>
            <button type="button" aria-pressed={eraser} disabled={busy}
              onClick={() => setEraser((v) => !v)}
              className={`min-h-8 rounded-lg border px-2.5 py-1 font-semibold disabled:opacity-40 ${eraser
                ? 'border-amber-300/60 bg-amber-500/20 text-amber-100'
                : 'border-white/20 bg-white/10 text-white hover:bg-white/20'}`}>
              ⌫ Erase
            </button>
            <button type="button" disabled={busy || !painted}
              onClick={() => {
                const c = brushCanvasRef.current;
                c?.getContext('2d')?.clearRect(0, 0, c.width, c.height);
                setPainted(false);
              }}
              className="min-h-8 rounded-lg border border-white/20 bg-white/10 px-2.5 py-1 font-semibold text-white hover:bg-white/20 disabled:opacity-40">
              Clear
            </button>
            <span className="text-white/45">Paint over what should go.</span>
          </>
        )}
      </div>

      {/* [container-type:size] turns this cell into a size-query container, so
          both editors' `100cqh/100cqw` caps mean THIS stage — the space left
          between the toolbars — instead of falling back to the whole viewport.
          Without it the picture could still be taller than the room it has and
          hide the prompt/Repair row below. Same construct the watermark review
          and the bank mask dialog already use around the same editors. */}
      <div className="flex min-h-0 flex-1 items-center justify-center [container-type:size]">
        {brush ? (
          <InpaintBrushEditor src={bust ? `${src}${src.includes('?') ? '&' : '?'}r=${bust}` : src}
            alt={alt} disabled={busy} eraser={eraser} brushCss={brushSize}
            canvasRef={brushCanvasRef} onDirty={setPainted} />
        ) : (
          <WatermarkRegionEditor src={bust ? `${src}${src.includes('?') ? '&' : '?'}r=${bust}` : src}
            alt={alt} regions={regions} disabled={busy}
            addMode={addMode} selectedIndex={selected}
            onAddModeChange={setAddMode} onSelectedIndexChange={setSelected}
            onCommit={setRegions} />
        )}
      </div>

      <div className="mt-2 flex flex-col gap-2">
        {done && !error && (
          <p role="status" className="m-0 rounded-lg border border-sky-400/40 bg-sky-500/10 px-3 py-1.5 text-[0.75rem] text-sky-100">
            ✦ Repaired — everything outside your zone is untouched. Not right? Change the
            description and repair again, or ↩ undo.
          </p>
        )}
        {error && (
          <p role="alert" className="m-0 rounded-lg border border-red-400/40 bg-red-500/10 px-3 py-1.5 text-[0.75rem] text-red-200">
            {error}
          </p>
        )}
        <div className="flex flex-wrap items-center gap-2">
          <input type="text" value={prompt} onChange={(e) => setPrompt(e.target.value)}
            disabled={busy} placeholder='What should be there? e.g. "remove the extra finger"'
            className="min-w-[16rem] flex-1 rounded-lg border border-white/20 bg-black/40 px-3 py-2 text-sm text-white placeholder:text-white/35 disabled:opacity-40" />
          {done && onUndo && (
            <button type="button" onClick={undo} disabled={busy}
              title="Put back the image from just before this repair, so you can try another description"
              className="rounded-lg border border-amber-400/50 bg-amber-500/20 px-4 py-2 text-sm font-semibold text-amber-100 disabled:opacity-40">
              ↩ Undo repair
            </button>
          )}
          <button type="button" onClick={() => onClose(done ? { ok: true } : undefined)} disabled={busy}
            className="rounded-lg border border-white/25 px-4 py-2 text-sm text-white disabled:opacity-40">
            {done ? 'Done' : 'Cancel'}
          </button>
          <button type="button" onClick={run} disabled={!ready || busy}
            title={brush && !painted
              ? 'Paint over the area to repair first'
              : !brush && !regions.length
                ? 'Draw the area to repair first'
                : !prompt.trim()
                  ? 'Say what should be painted in that area'
                  : 'Repaint only that area — everything outside it stays byte-identical'}
            className="rounded-lg border border-sky-400/60 bg-sky-500/25 px-5 py-2 text-sm font-semibold text-sky-50 disabled:opacity-40">
            {busy ? '✦ Repairing…' : done ? '✦ Repair again' : '✦ Repair'}
          </button>
        </div>
      </div>
    </div>
  );
}

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

export default function RepairDialog({ open, src, alt = 'image', onClose, onSubmit, onUndo = null }) {
  const dialogRef = useRef(null);
  const [regions, setRegions] = useState([]);
  const [addMode, setAddMode] = useState(true);
  const [selected, setSelected] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  /* The dialog STAYS OPEN on success. An inpaint is a dice roll: the gesture
     people actually make is look → not right → change the sentence → go again.
     Closing on success forced a full round trip for every attempt. `bust` is
     what makes the new pixels visible — the file is overwritten in place, so
     the URL does not move. (Asked for on Discord right after ✦ Repair shipped.) */
  const [done, setDone] = useState(false);
  const [bust, setBust] = useState(0);
  useFocusTrap(dialogRef, open);

  // A fresh dialog never inherits the previous image's zones or sentence.
  useEffect(() => {
    if (!open) return;
    setRegions([]); setPrompt(''); setError(null); setSelected(null); setAddMode(true);
    setDone(false); setBust(0);
  }, [open, src]);

  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, busy, onClose]);

  const run = useCallback(async () => {
    if (!regions.length || !prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const d = await onSubmit({ boxes: regions, prompt: prompt.trim() });
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
  }, [regions, prompt, busy, onSubmit]);

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
  const ready = regions.length > 0 && !!prompt.trim();

  return (
    <div className="fixed inset-0 z-[9998] flex flex-col bg-black/90 p-3 sm:p-4"
      role="dialog" aria-modal="true" aria-label="Repair an area of this image"
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

      <div className="flex min-h-0 flex-1 items-center justify-center">
        <WatermarkRegionEditor src={bust ? `${src}${src.includes('?') ? '&' : '?'}r=${bust}` : src}
          alt={alt} regions={regions} disabled={busy}
          addMode={addMode} selectedIndex={selected}
          onAddModeChange={setAddMode} onSelectedIndexChange={setSelected}
          onCommit={setRegions} />
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
            title={!regions.length
              ? 'Draw the area to repair first'
              : !prompt.trim()
                ? 'Say what should be painted in that area'
                : 'Repaint only the drawn zone — everything outside it stays byte-identical'}
            className="rounded-lg border border-sky-400/60 bg-sky-500/25 px-5 py-2 text-sm font-semibold text-sky-50 disabled:opacity-40">
            {busy ? '✦ Repairing…' : done ? '✦ Repair again' : '✦ Repair'}
          </button>
        </div>
      </div>
    </div>
  );
}

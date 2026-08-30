import { useEffect, useRef, useState } from 'react';
import { postJson } from '../../api/fetchClient';
import { useFocusTrap } from '../../hooks/useFocusTrap';
import { compareSeed, defaultTicked, toggleTicked } from './kleinCompare.js';

/* ⚖ Compare Klein models — the intermediate window the CLEAN was missing.
 *
 * Same shape as the detection scan's launch window on the maintainer's ask
 * ("un modal intermédiaire avant l'envoi, comme la détection") : judge on a
 * sample BEFORE committing the batch. Here the sample is one flagged image and
 * the judged variable is the MODEL: every candidate runs on the same image,
 * same zones, same seed — the server enforces the derivation (manual zones
 * first, else the detected bbox), the dialog enforces the seed.
 *
 * ONE component for both surfaces. What differs is only what "adopt" means,
 * and the parent says it via `onAdopt`:
 *  - dataset: save the per-dataset pick (the batch clean already honours it);
 *  - bank: arm a PER-RUN override on the next clean — a bank deliberately
 *    stores no Klein pick (one stored authority for the UNETLoader, and it is
 *    the dataset's).
 *
 * Results stream into the grid one model at a time: a 9B UNET swap costs tens
 * of seconds, and a spinner that hides three finished results behind a fourth
 * still running would waste exactly the time this window exists to save.
 */
export default function KleinCompareDialog({
  choices = [], stored = null, compareUrl, onAdopt, onClose,
  adoptLabel = 'Use this model',
}) {
  const [ticked, setTicked] = useState(() => defaultTicked(choices, stored));
  const [results, setResults] = useState([]);   // [{model, ok, after?, error?, seconds?}]
  const [before, setBefore] = useState(null);   // {image: b64, label}
  const [running, setRunning] = useState(null); // model currently in flight
  const [ran, setRan] = useState(false);
  const seedRef = useRef(compareSeed());
  const aliveRef = useRef(true);
  const dialogRef = useRef(null);
  useFocusTrap(dialogRef, true);

  useEffect(() => {
    aliveRef.current = true;
    const onKeyDown = (e) => { if (e.key === 'Escape' && !running) onClose(); };
    window.addEventListener('keydown', onKeyDown);
    return () => { aliveRef.current = false; window.removeEventListener('keydown', onKeyDown); };
  }, [running, onClose]);

  const run = async () => {
    if (running) return;
    const models = choices.filter((m) => ticked.includes(m));
    if (!models.length) return;
    setResults([]); setRan(true);
    // Sequential ON PURPOSE: ComfyUI holds one UNET at a time, and firing all
    // candidates at once would just queue them while hiding the progress.
    for (const model of models) {
      if (!aliveRef.current) return;
      setRunning(model);
      let out = null;
      try {
        out = await postJson(compareUrl, { model, seed: seedRef.current });
      } catch (e) {
        out = { ok: false, error: e.message || 'the compare call failed' };
      }
      if (!aliveRef.current) return;
      if (out?.ok && out.before && !before) {
        setBefore({ image: out.before, label: out.label || `#${out.image_id}` });
      }
      setResults((prev) => [...prev, {
        model, ok: !!out?.ok, after: out?.after || null,
        error: out?.ok ? null : (out?.error || 'failed'), seconds: out?.seconds,
      }]);
      setRunning(null);
    }
  };

  const busyText = running
    ? `Running ${running}… a model swap takes tens of seconds.`
    : null;

  return (
    <div className="fixed inset-0 z-[9990] flex items-center justify-center bg-black/80 p-3"
      role="dialog" aria-modal="true" aria-label="Compare Klein models">
      <div ref={dialogRef}
        className="max-h-[92vh] w-full max-w-3xl overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-content">⚖ Compare Klein models</h3>
          <button type="button" onClick={onClose} disabled={!!running} aria-label="Close"
            className="text-lg leading-none text-content-subtle hover:text-content disabled:opacity-40">×</button>
        </div>
        <p className="text-xs leading-relaxed text-content-muted">
          Each ticked model repaints the <b>same flagged image</b>, same zones, same
          seed — so what differs between the results is the model, nothing else. The
          original is never modified; this is a preview, exactly like the detection
          window&apos;s sample.
        </p>

        <div className="flex flex-wrap gap-2">
          {choices.map((m) => (
            <label key={m}
              className="flex items-center gap-1.5 rounded-md border border-border px-2 py-1 text-xs text-content-muted">
              <input type="checkbox" checked={ticked.includes(m)} disabled={!!running}
                onChange={() => setTicked((t) => toggleTicked(t, m))} />
              <span className="font-mono break-all">{m}{m === stored ? ' (current)' : ''}</span>
            </label>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <button type="button" onClick={run} disabled={!!running || !ticked.length}
            className="rounded-md bg-gradient-primary px-3 py-1.5 text-xs font-semibold text-gray-950 disabled:opacity-50">
            {ran ? 'Run again (same image & seed)' : `Run on ${ticked.length} model${ticked.length > 1 ? 's' : ''}`}
          </button>
          {busyText && (
            <span role="status" aria-live="polite" className="text-xs text-content-muted">{busyText}</span>
          )}
        </div>

        {before && (
          <figure className="space-y-1">
            <img src={`data:image/jpeg;base64,${before.image}`} alt="Before — the flagged image"
              className="max-h-64 w-auto rounded-md border border-border" />
            <figcaption className="text-[0.6875rem] text-content-subtle">
              Before — <span className="font-mono">{before.label}</span>
            </figcaption>
          </figure>
        )}

        {results.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2">
            {results.map((r) => (
              <figure key={r.model} className="space-y-1 rounded-md border border-border p-2">
                <figcaption className="flex flex-wrap items-center justify-between gap-1 text-[0.6875rem] text-content-muted">
                  <span className="font-mono break-all">{r.model}</span>
                  {r.ok && Number.isFinite(r.seconds) && <span>{r.seconds}s</span>}
                </figcaption>
                {r.ok ? (
                  <>
                    <img src={`data:image/jpeg;base64,${r.after}`}
                      alt={`Result with ${r.model}`}
                      className="w-full rounded-md" />
                    <button type="button" onClick={() => onAdopt?.(r.model)}
                      className="w-full rounded-md border border-border px-2 py-1 text-xs font-semibold text-content-muted hover:bg-surface-raised hover:text-content">
                      {adoptLabel}
                    </button>
                  </>
                ) : (
                  <p className="text-xs text-rose-300">{r.error}</p>
                )}
              </figure>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

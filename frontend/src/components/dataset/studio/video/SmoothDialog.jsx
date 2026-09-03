/**
 * ↗ Smooth asks for the rate before it runs.
 *
 * The interpolator (RIFE) works by WHOLE factors: the choices are the source
 * rate times 2, 3 and 4 — 48, 72 and 96 fps for a clip authored at 24 — and
 * not any number, because a 30 or 60 fps target would mean throwing frames
 * away unevenly after the pass, which reads as judder. The clip keeps its
 * length (frames are added between the existing ones, never slowed down) and
 * the work grows with the frames written: ×3 costs about twice ×2, ×4 about
 * three times. The result is a NEW clip in the list; the original stays.
 *
 * PORTALLED on `document.body`, like every Studio modal (the contract test
 * studioModalsArePortaled enumerates this folder): a `sticky` ancestor would
 * otherwise trap the overlay under the rail without a single test going red.
 */
import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { Waves } from 'lucide-react';
import { HelpBadge } from '../../../../help/HelpMode';
import { smoothTargets } from './videoStudioApi';

export default function SmoothDialog({ clip, busy, onSmooth, onClose }) {
  const targets = smoothTargets(clip);
  const [multiplier, setMultiplier] = useState(targets[0].multiplier);
  const picked = targets.find((t) => t.multiplier === multiplier) || targets[0];

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) onClose?.(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [busy, onClose]);

  const submit = (e) => {
    e.preventDefault();
    if (!busy) onSmooth?.(picked.multiplier);
  };

  const costLine = picked.cost === 1
    ? 'The quickest pass.'
    : `About ${picked.cost}× the time of the ×2 pass.`;

  return createPortal(
    <div role="dialog" aria-modal="true" aria-label={`Smooth clip #${clip.id}`}
      data-probe-chrome="smooth-dialog" data-probe-layer
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-2 sm:p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget && !busy) onClose?.(); }}>
      <form onSubmit={submit}
        className="flex w-full max-w-md max-h-[92vh] flex-col overflow-hidden rounded-xl border border-border bg-surface-overlay shadow-2xl">
        <header className="shrink-0 space-y-1 border-b border-border p-4">
          <h2 className="flex items-center gap-2 text-base font-bold text-content">
            <Waves aria-hidden="true" className="h-4 w-4" />
            Smooth clip #{clip.id}
            <HelpBadge topic="video-smooth-rate" />
          </h2>
          <p className="text-sm text-content-muted">
            Frames are added between the existing ones: the clip keeps its length and plays at the
            rate you pick. A NEW clip in the list; this one stays as it is.
          </p>
        </header>

        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3 sm:p-4">
          <div>
            <p className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-content-subtle">Playback rate</p>
            <div role="radiogroup" aria-label="Playback rate" className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-surface p-0.5">
              {targets.map((t) => (
                <button key={t.multiplier} type="button" role="radio" aria-checked={t.multiplier === multiplier}
                  onClick={() => setMultiplier(t.multiplier)}
                  className={`flex min-h-10 flex-col items-center justify-center rounded-md px-2 py-1.5 text-sm font-semibold lg:min-h-0 ${
                    t.multiplier === multiplier ? 'bg-primary text-white' : 'text-content-muted hover:text-content'}`}>
                  <span>{t.fps} fps</span>
                  <span className="text-[0.6875rem] font-normal opacity-80">×{t.multiplier}</span>
                </button>
              ))}
            </div>
          </div>
          <p className="text-xs text-content-muted">
            {picked.frames
              ? `${clip.frames} → ${picked.frames} frames, same length. `
              : `${picked.fps} fps, same length. `}
            {costLine}
          </p>
          <p className="text-[0.6875rem] text-content-subtle">
            Whole factors only: the interpolator writes 1, 2 or 3 frames between each pair. Any other
            rate would mean dropping frames unevenly afterwards, which reads as judder.
          </p>
        </div>

        <div className="shrink-0 border-t border-border p-3">
          <div className="flex flex-wrap justify-end gap-2">
            <button type="button" onClick={onClose} disabled={busy}
              className="min-h-10 rounded-md border border-border px-3 py-1.5 text-sm text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50 lg:min-h-0">
              Cancel
            </button>
            <button type="submit" disabled={busy}
              className="flex min-h-10 items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-50 lg:min-h-0">
              <Waves aria-hidden="true" className="h-3.5 w-3.5" />
              {busy ? 'Queuing…' : `Smooth to ${picked.fps} fps`}
            </button>
          </div>
        </div>
      </form>
    </div>,
    document.body,
  );
}

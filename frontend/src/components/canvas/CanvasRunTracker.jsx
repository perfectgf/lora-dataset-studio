import { canvasResultLabel, describeCanvasRun } from '../../utils/canvasRunResults';
import { pinBatchLabel } from '../../utils/canvasPinBatch';

/* 🎨 The board's own generation bar: what is being rendered, and where it went.

   It answers two things that were genuinely missing, both reported from real
   use. "If I leave this page I cannot get back to it" — the progress lives on
   the BOARD now, not inside the settings panel, so it is here whether that panel
   is open, closed, or was never reopened after a reload. And "once the image is
   generated I cannot see where it appears" — a launch that ends in silence is a
   launch you cannot trust, so the finished run names the checkpoints it filled
   and each one opens its gallery in a click.

   The vocabulary is the Test Studio's, word for word ("N generating · M queued",
   "Stop (resumable)"): it is the same engine, and inventing a second wording for
   one run would read as a second feature.

   400 px is the design target, not the fallback: the row wraps, the result
   buttons wrap under it, and nothing here is ever the reason the page scrolls
   sideways. */
export default function CanvasRunTracker({ run, targets, onStop, onResume, onOpenResult,
  onOpenPanel, onDismiss, pinCount = 0, pinBusy = false, pinSaid = '',
  onPinAll, onUndoPinAll }) {
  const s = describeCanvasRun(run);
  if (s.phase === 'idle') return null;
  // 📌 The one-click way onto the board. It says HOW MANY it will put down, and
  // it is simply not rendered once there is nothing left to put down — a lit
  // button that does nothing is a worse answer than no button.
  const pinLabel = pinBatchLabel(pinCount);

  const working = s.phase === 'working';
  const tone = working
    ? 'border-indigo-400/40 bg-indigo-500/10 '
    : s.phase === 'stopped'
      ? 'border-amber-400/40 bg-amber-500/10 '
      : 'border-emerald-400/40 bg-emerald-500/10 ';

  return (
    <div data-testid="canvas-run-tracker" role="status"
      // No bottom margin: the board's overlay owns the gap between its rows, and
      // this bar now sits INSIDE an opaque pill that provides its own padding.
      className={'flex flex-wrap items-center gap-x-2 gap-y-1.5 rounded-xl border px-2.5 py-1.5 text-[0.6875rem] ' + tone}>
      {working ? (
        <span aria-hidden className="inline-block h-3.5 w-3.5 shrink-0 animate-spin rounded-full border-2 border-indigo-400/40 border-t-indigo-400" />
      ) : (
        <span aria-hidden className="shrink-0">{s.phase === 'stopped' ? '⏸' : '✓'}</span>
      )}
      <span className="min-w-0 font-semibold text-content">{s.text}</span>

      {working && (
        <>
          <button type="button" onClick={onOpenPanel}
            className="rounded-md border border-border bg-app/60 px-2 py-0.5 text-content-muted hover:text-content">
            Settings
          </button>
          <button type="button" onClick={onStop}
            className="ml-auto rounded-md bg-red-600/80 px-2 py-0.5 font-semibold text-white">
            Stop (resumable)
          </button>
        </>
      )}

      {s.phase === 'stopped' && (
        <button type="button" onClick={onResume}
          className="ml-auto rounded-md bg-gradient-primary px-2 py-0.5 font-semibold text-gray-950">
          ▶ Resume
        </button>
      )}

      {/* The end of the gesture. Each button opens the gallery of the checkpoint
          it names — the same gallery the × N badge on the pill opens, which is
          also where the new image now shows up on the board. */}
      {s.phase === 'done' && (
        <>
          <span className="text-content-muted">
            {targets.length
              ? 'They joined the gallery of the checkpoints you picked:'
              : 'They joined the gallery of the checkpoint they were generated from.'}
          </span>
          <span className="flex min-w-0 flex-wrap gap-1">
            {targets.map((t) => (
              <button key={`${t.datasetId}:${t.recordId}:${t.step}`} type="button"
                onClick={() => onOpenResult(t)}
                title={`Open the images generated from run #${t.recordId} at step ${t.step}`}
                className="rounded-md border border-emerald-400/50 bg-emerald-500/10 px-1.5 py-0.5 text-emerald-100 tabular-nums hover:bg-emerald-500/25">
                <span aria-hidden>🖼</span> {canvasResultLabel(t)}
              </button>
            ))}
          </span>
          {/* 📌 …or put the whole lot on the board at once. Beside Dismiss
              because they are the two ends of the same sentence: take them, or
              let them go. Placement is utils/canvasPinBatch — one column per
              source checkpoint, in a band under the lane, guaranteed to overlap
              nothing that is already there. */}
          {pinLabel && (
            <button type="button" onClick={onPinAll} disabled={pinBusy}
              data-testid="canvas-pin-all"
              title="Put every image this run produced on the board, each under the checkpoint that made it"
              className="shrink-0 rounded-md border border-emerald-400/60 bg-emerald-500/20 px-2 py-0.5 font-semibold text-emerald-50 hover:bg-emerald-500/35 disabled:opacity-50">
              {pinBusy ? '📌 Pinning…' : pinLabel}
            </button>
          )}
          {onUndoPinAll && (
            <button type="button" onClick={onUndoPinAll}
              data-testid="canvas-pin-all-undo"
              title="Take those images back off the board"
              className="shrink-0 rounded-md border border-border bg-app/60 px-2 py-0.5 text-content-muted hover:text-content">
              ↩ Undo
            </button>
          )}
          <button type="button" onClick={onDismiss}
            className="ml-auto shrink-0 text-content-subtle underline decoration-dotted hover:text-content">
            Dismiss
          </button>
          {/* What actually happened, announced. A bulk action that reports
              nothing is a bulk action you have to go and audit by hand — and if
              some of the lot could NOT be placed, this is where it is said.
              NOT its own live region: this bar already IS one (`role="status"`
              on the wrapper), and a live region nested in a live region makes
              screen readers announce the same sentence twice. */}
          <span data-testid="canvas-pin-all-said"
            className="basis-full text-content-muted">
            {pinSaid}
          </span>
        </>
      )}
    </div>
  );
}

import { HelpBadge } from '../../help/HelpMode'
import {
  BURST_SHORTCUTS, BURST_HINT, burstProgressLine, burstEndNote, undoLine,
} from './videoBurstTriage'

/** ⌨ The burst-mode bar — the only chrome the keyboard run needs.
 *
 * Three things live here and nowhere else, each for a reason worth stating:
 *
 * THE UNDO OFFER IS A LINE, NOT A TOAST. A four-second toast under a
 * one-keystroke-a-second run is a stroboscope: it is replaced before it is
 * read, and the one time it matters it has already faded. The offer sits in the
 * bar, names the shot it would put back, says how many steps the net still has,
 * and costs no timer.
 *
 * THE ? PANEL DOES NOT TRAP FOCUS. It is a disclosure inside the bar, not a
 * modal — opening it must not move the focus, because the keys are the feature
 * and a focus trap would suspend every one of them. That is also why it can be
 * left open while you keep triaging.
 *
 * NOTHING HERE IS COLOUR-ONLY. The toggle prints On/Off, the queue prints its
 * count, the cursor on the grid carries a ▸ marker as well as its ring.
 */
export default function VideoBurstBar({
  on, autoAdvance, clips, index, hasMore, undoStack, saving,
  onToggle, onAutoAdvance, onUndo, helpOpen, onHelp,
}) {
  const end = on ? burstEndNote({ clips, index, hasMore }) : null
  const undo = undoLine(undoStack)

  return (
    <div className="rounded-lg border border-border bg-surface p-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button type="button" onClick={onToggle} aria-pressed={on}
          title="Judge shots from the keyboard: one key per shot, and the cursor moves to the next untriaged one on its own."
          className={`rounded-md border px-2.5 py-1 font-semibold transition-colors ${
            on
              ? 'border-primary bg-primary/20 text-content'
              : 'border-border bg-surface-raised text-content-muted hover:bg-surface'}`}>
          ⌨ Burst mode <span className="font-mono">{on ? 'On' : 'Off'}</span>
        </button>
        <HelpBadge topic="video-burst-triage" />

        {on && (
          <>
            <span className="tabular-nums text-content-muted">{burstProgressLine(clips)}</span>
            <label className="flex items-center gap-1.5 text-content-muted"
              title="After each decision, jump straight to the next shot you have not judged yet. Off, the cursor stays put so you can correct yourself.">
              <input type="checkbox" checked={!!autoAdvance}
                onChange={(e) => onAutoAdvance(e.target.checked)}
                className="accent-indigo-500" />
              Auto-advance
            </label>
            {saving > 0 && (
              /* A run that has ENDED is not a run that is SAVED. Saying so is
                 what makes leaving the page a decision rather than a surprise. */
              <span role="status" className="text-amber-300">⏳ saving {saving}…</span>
            )}
          </>
        )}

        <button type="button" onClick={onHelp} aria-expanded={!!helpOpen}
          aria-controls="video-burst-shortcuts"
          className="ml-auto rounded-md border border-border bg-surface-raised px-2 py-1 font-semibold text-content-muted hover:bg-surface">
          ? Shortcuts
        </button>
      </div>

      {on && (
        <>
          {undo && (
            <div className="mt-1.5 flex flex-wrap items-center gap-2 text-xs">
              <button type="button" onClick={onUndo}
                className="rounded-md border border-amber-500/60 bg-amber-500/10 px-2 py-1 font-semibold text-amber-200 hover:bg-amber-500/20">
                ↩ Undo
              </button>
              <span className="min-w-0 text-content-muted">{undo}</span>
            </div>
          )}
          {end && (
            <p role="status" className="mt-1.5 text-xs text-content-muted">{end}</p>
          )}
          <p className="mt-1.5 font-mono text-[0.6875rem] text-content-subtle">{BURST_HINT}</p>
        </>
      )}

      {helpOpen && (
        /* A region, not a dialog: it must never take the focus, because the
           keyboard IS the feature being documented. */
        <div id="video-burst-shortcuts" role="region" aria-label="Burst mode shortcuts"
          className="mt-2 rounded-md border border-border bg-surface-raised p-2">
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
            {BURST_SHORTCUTS.map((s) => (
              <div key={s.keys} className="contents">
                <dt className="whitespace-nowrap font-mono font-semibold text-content">{s.keys}</dt>
                <dd className="min-w-0 text-content-muted">{s.what}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-2 text-[0.6875rem] text-content-subtle">
            Shortcuts never fire while you are typing in a search or a threshold
            field. Turn burst mode off with Esc.
          </p>
        </div>
      )}
    </div>
  )
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch } from '../../api/fetchClient';
import { HelpBadge } from '../../help/HelpMode';
import {
  POLL_MS, machineLoadSummary, readMachineLoadPref, shouldPoll,
  systemStatsSegments, writeMachineLoadPref,
} from '../../utils/systemStats';

/* 📊 How hard the machine is working, on the board that is working it.
 *
 * The Canvas is where runs are launched and pictures are generated, and the
 * question it could not answer was the simplest one: "is anything HAPPENING?".
 * A run that is queued, a run that is training and a run that is wedged all
 * look identical from here. Four numbers settle it at a glance — and a glance
 * is exactly the budget: one 11-px line in the toolbar the board already has,
 * not a dashboard, not a graph, not a second page.
 *
 * WHAT IT DOES NOT DO, on purpose:
 *  • no history, no sparkline. A curve invites you to watch it; this is meant
 *    to be read in half a second and forgotten.
 *  • no per-process breakdown. "Which of these is ComfyUI?" is a Task Manager
 *    question and pretending to answer it here would need a far more expensive
 *    probe on every poll.
 *  • no alerting. Numbers are emerald below 50 %, amber 50-80 % and rose past
 *    80 % of a resource, and that is the entire vocabulary.
 *
 * COST. Three deliberate limits, because this is the only thing on the page
 * that polls forever:
 *  • only while the tab is VISIBLE (Page Visibility). A canvas left open
 *    overnight in a background tab would otherwise ask ~17 000 times.
 *  • only while it is unfolded — the ▾ toggle stops the timer, it does not
 *    just hide the line, and the choice is remembered.
 *  • `background: true`, so a poll that fails while the server restarts never
 *    raises the "connection lost" toast. A load readout is not worth an alarm;
 *    it just stops updating until the next poll succeeds.
 *
 * 📱 It used to be `hidden` below `sm`, and the reason given was the toolbar:
 * on a 400-px screen that row already wrapped twice, and every pixel spent
 * there was a pixel of board under the fold. That reason has gone — the readout
 * is not in the toolbar below `2xl` any more, it is in the board's ⋯ shelf,
 * where it costs the board nothing until it is opened. And the phone turns out
 * to be the device that wants it MOST: it is the screen you check the machine
 * from when you are not sitting at it, which is the whole reason the board is
 * opened over Tailscale in the first place.
 */

const TONE_CLASS = {
  calm: 'text-emerald-300/90',
  warm: 'text-amber-300/90',
  hot: 'text-rose-300',
};

export default function CanvasSystemStats() {
  const [enabled, setEnabled] = useState(readMachineLoadPref);
  const [stats, setStats] = useState(null);
  // Kept in a ref so the poll loop reads the CURRENT value without being torn
  // down and rebuilt (and re-fetching) on every visibility flicker.
  const enabledRef = useRef(enabled);
  enabledRef.current = enabled;

  const poll = useCallback(async () => {
    const visibility = typeof document !== 'undefined' ? document.visibilityState : undefined;
    if (!shouldPoll({ enabled: enabledRef.current, visibility })) return;
    try {
      const data = await apiFetch('/api/system/stats', { background: true });
      setStats(data && typeof data === 'object' ? data : null);
    } catch {
      // Leave the last reading on screen. A number that stopped moving is more
      // useful than a row of dashes, and the next tick will correct it.
    }
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    poll();
    const timer = setInterval(poll, POLL_MS);
    // Coming BACK to the tab refreshes immediately: the alternative is up to
    // five seconds of a stale number on the screen you just switched to.
    const onVisibility = () => { if (document.visibilityState === 'visible') poll(); };
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisibility);
    };
  }, [enabled, poll]);

  const toggle = () => {
    setEnabled((v) => {
      writeMachineLoadPref(!v);
      return !v;
    });
  };

  const segments = systemStatsSegments(stats);

  // Nothing measurable on this machine (a container with no card and no
  // psutil): draw NOTHING, not an empty frame with a toggle that reveals
  // nothing. The first poll has not answered yet in the same state, so the row
  // simply appears when it has something to say.
  if (enabled && !segments.length) return null;

  return (
    <span data-testid="canvas-system-stats"
      className="flex flex-wrap items-center gap-1.5">
      {enabled && (
        <span className="flex items-center gap-1.5 tabular-nums text-[0.625rem]"
          title={machineLoadSummary(segments)}>
          {segments.map((s) => (
            <span key={s.key} className="flex items-center gap-1 whitespace-nowrap"
              title={s.title}>
              <span className="text-content-muted">{s.label}</span>
              <span className={TONE_CLASS[s.tone] || TONE_CLASS.calm}>{s.text}</span>
            </span>
          ))}
        </span>
      )}
      <button type="button" onClick={toggle} aria-pressed={enabled}
        data-testid="canvas-system-stats-toggle"
        title={enabled
          ? 'Hide the machine load readout (stops polling the server)'
          : 'Show CPU, GPU, VRAM and RAM of the machine running LDS'}
        aria-label={enabled ? 'Hide the machine load readout' : 'Show the machine load readout'}
        className="flex h-6 items-center rounded border border-border bg-app/40 px-1 text-content-subtle/70 text-[0.625rem] hover:text-content">
        {enabled ? '▾' : '📊'}
      </button>
      <HelpBadge topic="canvas-machine-load" />
    </span>
  );
}

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiFetch, postJson } from '../../api/fetchClient';
import { useToast } from '../common/Toast';
import { HelpBadge } from '../../help/HelpMode';
import {
  POLL_MS, freeMemorySummary, machineLoadSummary, readMachineLoadPref, shouldPoll,
  systemStatsSegments, writeMachineLoadPref,
} from '../../utils/systemStats';

/* 📊 How hard the machine is working — one 11-px line, wherever it is mounted.
 *
 * Born on the Canvas, where the question it settles first came up: a run that
 * is queued, a run that is training and a run that is wedged all look identical
 * until you can see the machine move. The header mounts the same component so
 * every other page — the Test Studio above all, where LoRAs are exercised
 * while ComfyUI works — can answer "is anything HAPPENING?" without keeping
 * Task Manager or a ComfyUI tab open. One component, two mounts, so the two
 * lines cannot drift apart. (Asked for by Sam Exit on Discord, who was running
 * a ComfyUI resource monitor just to watch LDS work.)
 *
 * WHAT IT DOES NOT DO, on purpose:
 *  • no history, no sparkline. A curve invites you to watch it; this is meant
 *    to be read in half a second and forgotten.
 *  • no per-process breakdown. "Which of these is ComfyUI?" is a Task Manager
 *    question and pretending to answer it here would need a far more expensive
 *    probe on every poll.
 *  • no alerting. Numbers are emerald, amber or rose (load past 50 % / 80 % of
 *    a resource, GPU heat past 70° / 85°), and that is the entire vocabulary.
 *
 * COST. Three deliberate limits, because this is the only thing on the page
 * that polls forever:
 *  • only while the tab is VISIBLE (Page Visibility). A tab left open
 *    overnight in the background would otherwise ask ~17 000 times.
 *  • only while it is unfolded — the ▾ toggle stops the timer, it does not
 *    just hide the line, and the choice is remembered per mount (prefKey).
 *    The header mount starts FOLDED for the same reason: it is on every page,
 *    and a poll nobody asked for should not run on all of them.
 *  • `background: true`, so a poll that fails while the server restarts never
 *    raises the "connection lost" toast. A load readout is not worth an alarm;
 *    it just stops updating until the next poll succeeds.
 *
 * 📱 Never `hidden` at any width: whoever mounts it decides WHERE it lives at
 * each breakpoint (the Canvas moved it into the board's ⋯ shelf; the header
 * places it with useMediaQuery) — because the phone is the device that wants
 * it MOST, being the screen you check the machine from when you are not
 * sitting at it.
 */

const TONE_CLASS = {
  calm: 'text-emerald-300/90',
  warm: 'text-amber-300/90',
  hot: 'text-rose-300',
};

export default function SystemStatsReadout({
  prefKey, defaultEnabled, testId, helpTopic,
}) {
  const toast = useToast();
  const [enabled, setEnabled] = useState(
    () => readMachineLoadPref(undefined, prefKey, defaultEnabled));
  const [stats, setStats] = useState(null);
  // 🧹 One release at a time: the button is disabled while the server unloads
  // and re-reads (a couple of seconds), so a double press cannot queue two.
  const [freeing, setFreeing] = useState(false);
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
      writeMachineLoadPref(!v, undefined, prefKey);
      return !v;
    });
  };

  /* 🧹 Give the memory back. What holds it on a machine running LDS is
     ComfyUI's model cache (every model of the day, offloaded to RAM when it
     leaves the card, kept until asked) and the vision model kept warm for
     captioning — neither returns it by itself. The server pulls both levers,
     refuses while something renders or trains, and answers with what the OS
     measured; the toast repeats THAT, then the readout re-polls at once. */
  const freeMemory = async () => {
    if (freeing) return;
    setFreeing(true);
    try {
      const d = await postJson('/api/system/free-memory', {});
      toast.success(freeMemorySummary(d));
    } catch (err) {
      toast.error(err?.message || 'Could not free the memory');
    } finally {
      setFreeing(false);
      poll();
    }
  };

  const segments = systemStatsSegments(stats);

  // Nothing measurable on this machine (a container with no card and no
  // psutil): draw NOTHING, not an empty frame with a toggle that reveals
  // nothing. The first poll has not answered yet in the same state, so the row
  // simply appears when it has something to say.
  if (enabled && !segments.length) return null;

  return (
    <span data-testid={testId}
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
      {/* 🧹 Only while the numbers are on screen — it belongs to the reading
          it acts on, and a folded readout must not widen the header for a
          button nobody asked for. Same 40 px target rule as ▾. */}
      {enabled && (
        <button type="button" onClick={freeMemory} disabled={freeing}
          data-testid={`${testId}-free`} aria-busy={freeing}
          title="Free memory: unload the models ComfyUI keeps cached in RAM and VRAM, and the vision model LDS loaded. They reload on the next job. Refused while something is rendering or training."
          aria-label="Free memory"
          className="flex h-10 items-center rounded border border-border bg-app/40 px-1.5 text-content-subtle/70 text-[0.625rem] hover:text-content disabled:cursor-wait disabled:opacity-60 lg:h-6 lg:px-1">
          {freeing ? '…' : '🧹'}
        </button>
      )}
      <button type="button" onClick={toggle} aria-pressed={enabled}
        data-testid={`${testId}-toggle`}
        title={enabled
          ? 'Hide the machine load readout (stops polling the server)'
          : 'Show CPU, GPU, VRAM, RAM and temperature of the machine running LDS'}
        aria-label={enabled ? 'Hide the machine load readout' : 'Show the machine load readout'}
        /* 📱 40 px below `lg`, like every small control (the responsive probe
           measures targets, not intentions): a 24-px glyph next to 11-px text
           looks perfectly deliberate in a screenshot and is still a miss for
           a finger. */
        className="flex h-10 items-center rounded border border-border bg-app/40 px-1.5 text-content-subtle/70 text-[0.625rem] hover:text-content lg:h-6 lg:px-1">
        {enabled ? '▾' : '📊'}
      </button>
      <HelpBadge topic={helpTopic} />
    </span>
  );
}

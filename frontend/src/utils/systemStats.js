/* 📊 The Canvas load readout — turning /api/system/stats into one small line.
 *
 * Everything that decides WHAT is drawn lives here, in plain JS, so it is
 * covered by `node --test` (which cannot parse JSX). The component is left with
 * markup and a timer.
 *
 * Two rules the widget rests on:
 *
 *  • A field the server did not send is a field the machine cannot measure —
 *    no NVIDIA card, no psutil. It is DROPPED, never drawn as 0. "GPU 0%" on a
 *    machine with no GPU is a lie that reads as good news.
 *  • Colour reads the state at a glance: every number carries a tone (emerald
 *    below 50%, amber 50-80%, rose past 80%), not just the ones in trouble —
 *    a line where nothing is coloured until it is already a problem is a line
 *    you have to read the digits of to trust.
 */

/** Fractions of a resource at which the tone changes. */
const WARM_AT = 0.50;
const HOT_AT = 0.80;

/** localStorage key for "the user folded this away". NEVER rename it — a new
 *  key would silently re-open the widget for everyone who had closed it. */
export const MACHINE_LOAD_PREF_KEY = 'lds.canvas.machineLoad';

/** 'calm' | 'warm' | 'hot' for a 0..1 fraction. Anything unmeasurable is calm:
 *  an unknown load must never paint a warning — it draws as the resting tone. */
export function loadTone(fraction) {
  if (typeof fraction !== 'number' || !Number.isFinite(fraction)) return 'calm';
  if (fraction >= HOT_AT) return 'hot';
  if (fraction >= WARM_AT) return 'warm';
  return 'calm';
}

/** Gigabytes as the eye wants them: 21.3 → "21", 4.7 → "4.7". Above 10 GB the
 *  decimal is noise in a 11-px line; below it, it is the difference between
 *  "nearly empty" and "nearly full" on an 8 GB card. */
export function formatGb(gb) {
  if (typeof gb !== 'number' || !Number.isFinite(gb)) return '';
  return gb >= 10 ? String(Math.round(gb)) : String(Math.round(gb * 10) / 10);
}

function pctSegment(key, label, value, title) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const pct = Math.max(0, Math.min(100, Math.round(value)));
  return { key, label, text: `${pct}%`, tone: loadTone(pct / 100), title };
}

function memSegment(key, label, used, total, title) {
  if (typeof used !== 'number' || typeof total !== 'number') return null;
  if (!Number.isFinite(used) || !Number.isFinite(total) || total <= 0) return null;
  return {
    key,
    label,
    text: `${formatGb(used)}/${formatGb(total)}G`,
    tone: loadTone(used / total),
    title,
  };
}

/**
 * The line, as an ordered list of segments. `[]` means "this machine could not
 * answer anything" — the widget then draws nothing at all rather than an empty
 * frame, which is the correct outcome inside a container with no card and no
 * psutil.
 *
 * Order is deliberate: CPU and GPU (what is WORKING) before RAM and VRAM (what
 * is FULL), because during a run the first question is always "is it moving?".
 */
export function systemStatsSegments(stats) {
  const s = stats && typeof stats === 'object' ? stats : {};
  return [
    pctSegment('cpu', 'CPU', s.cpu_percent, 'Processor load on the machine running LDS'),
    pctSegment('gpu', 'GPU', s.gpu_percent, 'GPU 0 utilisation'),
    memSegment('vram', 'VRAM', s.vram_used_gb, s.vram_total_gb,
      'GPU 0 memory in use / installed'),
    memSegment('ram', 'RAM', s.ram_used_gb, s.ram_total_gb,
      'System memory in use / installed'),
  ].filter(Boolean);
}

/** One sentence for the toggle's title, built from what is actually available
 *  so the tooltip never promises a GPU the machine does not have. */
export function machineLoadSummary(segments) {
  if (!segments || !segments.length) return 'Machine load — nothing measurable here';
  return `Machine load: ${segments.map((x) => `${x.label} ${x.text}`).join(' · ')}`;
}

/* --- polling policy ---------------------------------------------------------
 * A readout is worth ~5 s of staleness; a BACKGROUND tab is worth none at all.
 * A canvas left open overnight in a background tab would otherwise fork
 * nvidia-smi ~17 000 times for nobody to look at it.
 */
export const POLL_MS = 5000;

/** Should the widget fetch right now? Pure so the rule is testable without a
 *  browser: hidden tab → no; folded away → no. */
export function shouldPoll({ enabled, visibility }) {
  if (!enabled) return false;
  // `undefined` = an environment with no Page Visibility API (old WebView,
  // jsdom). Default to visible: a readout that never refreshes is worse than
  // one that refreshes when nobody is watching.
  return visibility !== 'hidden';
}

/** Read the folded/unfolded preference. Anything unreadable (private mode,
 *  storage disabled) means "shown", the default. */
export function readMachineLoadPref(storage) {
  try {
    const store = storage
      || (typeof localStorage !== 'undefined' ? localStorage : null);
    if (!store) return true;
    return store.getItem(MACHINE_LOAD_PREF_KEY) !== 'off';
  } catch {
    return true;
  }
}

export function writeMachineLoadPref(enabled, storage) {
  try {
    const store = storage
      || (typeof localStorage !== 'undefined' ? localStorage : null);
    if (!store) return;
    store.setItem(MACHINE_LOAD_PREF_KEY, enabled ? 'on' : 'off');
  } catch {
    /* storage disabled — the widget simply forgets between reloads */
  }
}

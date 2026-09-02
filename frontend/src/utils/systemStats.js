/* 📊 The machine-load readout — turning /api/system/stats into one small line.
 *
 * Drawn in two places, by one component (SystemStatsReadout): the Canvas
 * toolbar, where it has always lived, and the app header, where it answers the
 * same question from every other page. Everything that decides WHAT is drawn
 * lives here, in plain JS, so it is covered by `node --test` (which cannot
 * parse JSX). The component is left with markup and a timer.
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

/** °C at which the GPU temperature tone changes. Heat has its own scale — 50%
 *  of a resource is routine while 50 °C is a card at rest — so the fractions
 *  above cannot serve. NVIDIA cards start defending themselves (throttling)
 *  in the 83-90 °C band: amber warns on the approach, rose means the card is
 *  already there. */
const TEMP_WARM_AT_C = 70;
const TEMP_HOT_AT_C = 85;

/** localStorage key for "the user folded this away" on the Canvas. NEVER
 *  rename it — a new key would silently re-open the widget for everyone who
 *  had closed it. */
export const MACHINE_LOAD_PREF_KEY = 'lds.canvas.machineLoad';

/** Same choice for the header readout, remembered separately: hiding the
 *  numbers on a crowded header must not hide them from the Canvas toolbar,
 *  where they were asked for first. Same renaming rule as above. */
export const HEADER_MACHINE_LOAD_PREF_KEY = 'lds.header.machineLoad';

/** 'calm' | 'warm' | 'hot' for a 0..1 fraction. Anything unmeasurable is calm:
 *  an unknown load must never paint a warning — it draws as the resting tone. */
export function loadTone(fraction) {
  if (typeof fraction !== 'number' || !Number.isFinite(fraction)) return 'calm';
  if (fraction >= HOT_AT) return 'hot';
  if (fraction >= WARM_AT) return 'warm';
  return 'calm';
}

/** 'calm' | 'warm' | 'hot' for a GPU temperature in °C, on the heat scale. */
export function tempTone(tempC) {
  if (typeof tempC !== 'number' || !Number.isFinite(tempC)) return 'calm';
  if (tempC >= TEMP_HOT_AT_C) return 'hot';
  if (tempC >= TEMP_WARM_AT_C) return 'warm';
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

function tempSegment(key, label, value, title) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return { key, label, text: `${Math.round(value)}°`, tone: tempTone(value), title };
}

/**
 * The line, as an ordered list of segments. `[]` means "this machine could not
 * answer anything" — the widget then draws nothing at all rather than an empty
 * frame, which is the correct outcome inside a container with no card and no
 * psutil.
 *
 * Order is deliberate: CPU and GPU (what is WORKING) before RAM and VRAM (what
 * is FULL), because during a run the first question is always "is it moving?".
 * Temperature trails: it is neither, it is the health note.
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
    tempSegment('temp', 'Temp', s.gpu_temp_c, 'GPU 0 temperature'),
  ].filter(Boolean);
}

/** One sentence for the toggle's title, built from what is actually available
 *  so the tooltip never promises a GPU the machine does not have. */
export function machineLoadSummary(segments) {
  if (!segments || !segments.length) return 'Machine load — nothing measurable here';
  return `Machine load: ${segments.map((x) => `${x.label} ${x.text}`).join(' · ')}`;
}

/** 🧹 What the free-memory button says when it is done — from what the server
 *  MEASURED (a before/after of the OS's own numbers), never from what it asked
 *  for. Three facts, in the order a person checks them: how much came back,
 *  where RAM stands now, and which lever did (or did not) act. */
export function freeMemorySummary(result) {
  const r = result || {};
  const has = (v) => typeof v === 'number' && Number.isFinite(v);
  const parts = [];
  if (has(r.freed_gb) && r.freed_gb >= 0.1) {
    parts.push(`Freed ${formatGb(r.freed_gb)} GB of RAM`);
  } else if (has(r.freed_gb)) {
    parts.push('No RAM came back');
  } else {
    parts.push('Memory release asked');
  }
  if (has(r.ram_after_gb) && has(r.ram_total_gb)) {
    parts.push(`RAM now ${formatGb(r.ram_after_gb)}/${formatGb(r.ram_total_gb)} GB`);
  }
  if (has(r.vram_before_gb) && has(r.vram_after_gb)) {
    parts.push(`VRAM ${formatGb(r.vram_before_gb)} → ${formatGb(r.vram_after_gb)} GB`);
  }
  const levers = [];
  if (r.comfyui === 'freed') levers.push('ComfyUI unloaded its models');
  else if (r.comfyui === 'offline') levers.push('ComfyUI is not running');
  else levers.push('ComfyUI did not confirm');
  if (r.vision_released === true) levers.push('vision model released');
  else if (r.vision_released === false) levers.push('no vision model of LDS to release');
  return `${parts.join(' · ')} (${levers.join(', ')}).`;
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

/** Read a folded/unfolded preference. The two mounts default apart — the
 *  Canvas readout is shown until folded (it always was), the header one stays
 *  a quiet 📊 button until asked (a header serves every page, and a poll
 *  nobody opted into should not run on all of them) — so the fallback is the
 *  mount's own, and anything unreadable (private mode, storage disabled)
 *  falls back the same way. */
export function readMachineLoadPref(storage, key = MACHINE_LOAD_PREF_KEY,
  fallback = true) {
  try {
    const store = storage
      || (typeof localStorage !== 'undefined' ? localStorage : null);
    if (!store) return fallback;
    const value = store.getItem(key);
    if (value === 'on') return true;
    if (value === 'off') return false;
    return fallback;
  } catch {
    return fallback;
  }
}

export function writeMachineLoadPref(enabled, storage, key = MACHINE_LOAD_PREF_KEY) {
  try {
    const store = storage
      || (typeof localStorage !== 'undefined' ? localStorage : null);
    if (!store) return;
    store.setItem(key, enabled ? 'on' : 'off');
  } catch {
    /* storage disabled — the widget simply forgets between reloads */
  }
}

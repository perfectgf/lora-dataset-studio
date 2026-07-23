/* Multi-engine generation: which engines a batch runs on, how the selected
   shots are shared between them, and what that costs.
   PURE JS (no JSX) so node --test can import and exercise it directly.

   WHY THIS FILE EXISTS
   --------------------
   The workspace used to generate with ONE engine, persisted as a plain string
   in localStorage `datasetGenerator`. Users want several at once — either to
   VARY the dataset (each shot goes to one engine) or to COMPARE engines on the
   same shots (every engine renders every shot).

   The storage rule of this repo forbids renaming or re-typing a persisted key:
   `datasetGenerator` is read by the regenerate path (useDataset.js) and by the
   ✎ identity-prompt modal, which both want ONE engine. So the string key is
   KEPT, unchanged, as a mirror of the PRIMARY engine, and the list lives in a
   new key next to it. A profile that only ever knew the old key reads back as a
   one-engine selection — i.e. exactly today's behaviour. */

/** Canonical engine order — drives the card order, the primary pick and the
 *  round-robin. Stable: it is also the order batches are BUILT in, and Klein
 *  must come last at DISPATCH time (see engineBatches). */
export const ENGINES = ['klein', 'nanobanana', 'chatgpt'];

export const API_ENGINES = ['nanobanana', 'chatgpt'];

export const ENGINE_LABELS = {
  klein: 'Klein',
  nanobanana: 'Nano Banana Pro',
  chatgpt: 'ChatGPT',
};

/* Per-engine accent colour. Deliberately NOT green: green already means
   "kept / already in the dataset / free" everywhere else in the app, so using
   it for "selected" made two different messages share one colour. Indigo /
   amber / sky stay distinguishable in a dark theme AND in deuteranopia, which a
   green+amber pair does not. Class strings are spelled out in full because
   Tailwind scans source text — never build them by concatenation. */
export const ENGINE_ACCENTS = {
  klein: {
    card: 'border-indigo-400/60 bg-indigo-500/15 ring-1 ring-indigo-400/40',
    title: 'text-indigo-200',
    text: 'text-indigo-300',
    icon: 'text-indigo-300',
    pill: 'bg-indigo-500/25 text-indigo-200',
    dot: 'bg-indigo-400',
  },
  nanobanana: {
    card: 'border-amber-400/60 bg-amber-500/15 ring-1 ring-amber-400/40',
    title: 'text-amber-200',
    text: 'text-amber-300',
    icon: 'text-amber-300',
    pill: 'bg-amber-500/25 text-amber-200',
    dot: 'bg-amber-400',
  },
  chatgpt: {
    card: 'border-sky-400/60 bg-sky-500/15 ring-1 ring-sky-400/40',
    title: 'text-sky-200',
    text: 'text-sky-300',
    icon: 'text-sky-300',
    pill: 'bg-sky-500/25 text-sky-200',
    dot: 'bg-sky-400',
  },
};

/** Pay-per-image rate. Klein is local GPU time, hence free; the ChatGPT
 *  subscription lane spends plan quota, not dollars (handled by estimateCost). */
export const ENGINE_RATES = { klein: 0, nanobanana: 0.15, chatgpt: 0.17 };

export const STORAGE_ENGINES = 'datasetGenerators';     // JSON list (new)
export const STORAGE_PRIMARY = 'datasetGenerator';      // legacy string mirror — NEVER renamed
export const STORAGE_MODE = 'datasetGeneratorMode';     // 'split' | 'all'

/** The engine a profile with no stored preference generates with — the historic
 *  useState default of the workspace. */
export const DEFAULT_ENGINE = 'nanobanana';
export const MODES = ['split', 'all'];
/** Sharing the N selected shots between the engines (total = N, today's cost)
 *  is the default: nobody should multiply their bill without asking. */
export const DEFAULT_MODE = 'split';

/** Keep only real engine ids, de-duplicated, in canonical order. Anything else
 *  (a typo, a removed engine, a non-string) is dropped rather than trusted. */
export function canonicalEngines(list) {
  const wanted = new Set(Array.isArray(list)
    ? list.filter((e) => typeof e === 'string').map((e) => e.toLowerCase())
    : []);
  return ENGINES.filter((e) => wanted.has(e));
}

/** The stored selection, with the legacy single-string key as fallback.
 *  Order of trust: the list key → the legacy string → the historic default.
 *  An EMPTY stored list is a real state (the user unchecked everything) and is
 *  returned as such; only a missing/unusable key falls through. */
export function readEngines(storage) {
  let raw = null;
  try { raw = storage?.getItem(STORAGE_ENGINES) ?? null; } catch { raw = null; }
  if (raw != null) {
    try {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return canonicalEngines(parsed);
    } catch { /* corrupt JSON: fall through to the legacy key */ }
  }
  let legacy = null;
  try { legacy = storage?.getItem(STORAGE_PRIMARY) ?? null; } catch { legacy = null; }
  const fromLegacy = canonicalEngines([legacy]);
  if (fromLegacy.length) return fromLegacy;
  return [DEFAULT_ENGINE];
}

/** Persist the selection AND refresh the legacy mirror, so every existing
 *  single-engine reader (regenerate, the ✎ modal) keeps seeing a valid engine.
 *  The mirror is left untouched when nothing is selected — an empty selection
 *  generates nothing, and blanking it would make regenerate lose its engine. */
export function writeEngines(storage, engines) {
  const list = canonicalEngines(engines);
  try {
    storage?.setItem(STORAGE_ENGINES, JSON.stringify(list));
    if (list.length) storage?.setItem(STORAGE_PRIMARY, list[0]);
  } catch { /* private browsing / full storage: the in-memory state still works */ }
  return list;
}

export function readMode(storage) {
  let raw = null;
  try { raw = storage?.getItem(STORAGE_MODE); } catch { raw = null; }
  return MODES.includes(raw) ? raw : DEFAULT_MODE;
}

export function writeMode(storage, mode) {
  const value = MODES.includes(mode) ? mode : DEFAULT_MODE;
  try { storage?.setItem(STORAGE_MODE, value); } catch { /* ignore */ }
  return value;
}

/** The one engine single-engine consumers should use: first in canonical order.
 *  null when nothing is selected (callers keep their own fallback). */
export function primaryEngine(engines) {
  return canonicalEngines(engines)[0] || null;
}

/** Share `variations` between `engines`.
 *  - 'all'   : every engine renders EVERY shot (comparison — total = N × engines)
 *  - 'split' : round-robin, every shot goes to exactly ONE engine (variety —
 *              total = N, unchanged cost). 25 shots over 3 engines → 9/8/8.
 *  Returns [{ generator, variations }] in canonical order, with empty entries
 *  dropped (more engines than shots in split mode). One engine → a single entry
 *  holding all the shots, i.e. strictly the pre-existing behaviour. */
export function distributeVariations(variations, engines, mode) {
  const shots = Array.isArray(variations) ? variations : [];
  const list = canonicalEngines(engines);
  if (!list.length || !shots.length) return [];
  if (mode === 'all') return list.map((generator) => ({ generator, variations: [...shots] }));
  const buckets = list.map((generator) => ({ generator, variations: [] }));
  shots.forEach((shot, i) => { buckets[i % list.length].variations.push(shot); });
  return buckets.filter((b) => b.variations.length);
}

/** Dispatch order for the server: API engines first, local Klein LAST.
 *  The API batches are background threads that start returning images right
 *  away; Klein holds the single GPU and runs its shots in series, so putting it
 *  first would make the whole batch look frozen. */
export function engineBatches(variations, engines, mode) {
  const batches = distributeVariations(variations, engines, mode);
  return [...batches].sort((a, b) => (a.generator === 'klein' ? 1 : 0) - (b.generator === 'klein' ? 1 : 0));
}

/** True when the run mixes the local GPU engine with at least one API engine —
 *  the case where Klein's shots visibly queue behind the API ones. */
export function kleinQueuesBehindApi(engines) {
  const list = canonicalEngines(engines);
  return list.includes('klein') && list.some((e) => API_ENGINES.includes(e));
}

/** How many images the batch will produce: shots × multiplier, per engine. */
export function totalImages(shotCount, engines, mode, multiplier = 1) {
  const n = Math.max(0, Number(shotCount) || 0);
  const mult = Math.max(1, Number(multiplier) || 1);
  const list = canonicalEngines(engines);
  if (!list.length || !n) return 0;
  return (mode === 'all' ? n * list.length : n) * mult;
}

/** Dollar estimate for the batch. Klein contributes 0 (local GPU) and so does
 *  ChatGPT when it runs on the subscription lane (plan quota, not dollars).
 *  In split mode each engine only pays for ITS share — which is why the split
 *  is computed here rather than averaged. */
export function estimateCost(shotCount, engines, mode, { multiplier = 1, gptViaSub = false } = {}) {
  const n = Math.max(0, Number(shotCount) || 0);
  const mult = Math.max(1, Number(multiplier) || 1);
  const list = canonicalEngines(engines);
  if (!list.length || !n) return 0;
  const rate = (engine) => (engine === 'chatgpt' && gptViaSub ? 0 : ENGINE_RATES[engine] || 0);
  if (mode === 'all') return list.reduce((sum, e) => sum + n * mult * rate(e), 0);
  // split: round-robin share, same arithmetic as distributeVariations.
  return list.reduce((sum, e, i) => {
    const share = Math.floor(n / list.length) + (i < n % list.length ? 1 : 0);
    return sum + share * mult * rate(e);
  }, 0);
}

/** The engines that actually BILL for this run — names the guard-rail confirm
 *  ("this will cost $X on …") without listing free lanes. */
export function billingEngines(engines, { gptViaSub = false } = {}) {
  return canonicalEngines(engines).filter(
    (e) => (ENGINE_RATES[e] || 0) > 0 && !(e === 'chatgpt' && gptViaSub));
}

/** Why Generate is unavailable, or null when it can run. The empty selection is
 *  a real, reachable state (every card unchecked), and it must SAY so instead of
 *  queueing an empty batch. `maxFanout` mirrors the server cap; it is read from
 *  /api/capabilities, never hardcoded here, and 0/undefined disables the check
 *  (the server stays the authority and refuses with its own message). */
export function generateBlockedReason({ engines, shotCount, mode, multiplier = 1, maxFanout = 0 }) {
  const list = canonicalEngines(engines);
  if (!list.length) return 'Pick at least one engine above';
  if (!Number(shotCount)) return 'Select at least one shot';
  const total = totalImages(shotCount, list, mode, multiplier);
  if (maxFanout > 0 && total > maxFanout) {
    return `${total} images is over the ${maxFanout}-per-batch limit — `
      + (mode === 'all' ? 'switch to Split, ' : '') + 'uncheck an engine or select fewer shots';
  }
  return null;
}

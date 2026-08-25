/* 🎚 Bank triage thresholds — the DESCRIPTION of the twelve knobs, and the pure
   logic the Bank's threshold panel runs on.

   WHY THIS FILE EXISTS
   --------------------
   The twelve thresholds that decide every Bank flag lived in exactly one place:
   Settings ▸ Captioning & quality. So tightening duplicate detection meant
   leaving the bank you were triaging, finding a card three screens away in a
   different area of the app, editing a number, saving, and coming back to see
   whether it did anything. The panel this file backs puts all twelve where the
   work happens.

   ONE SOURCE OF TRUTH, NOT TWO. This module describes the fields — their group,
   their unit, which direction catches more, when they take effect. It holds NO
   default values: those come from the server (`config_defaults.bank`, derived
   from config.DEFAULTS), exactly like Settings ▸ Captioning, and the shared
   ResetToDefault button reads them. A default typed here would drift the day
   config.py moves and the reset button would then restore a stale number while
   claiming it is the default — bankThresholds.test.js fails the build if one
   appears. Both screens edit `config.bank.<key>` through the same PUT
   /api/settings, so there is no third source of truth and no sync problem: they
   are the same value seen twice.

   KEYS ARE PERSISTED. `field` is the config key inside the `bank` section and
   lives in every user's config.json. Never rename one without an alias.

   THE DIRECTION TRAP (this is why `catchesMoreWhen` exists)
   --------------------------------------------------------
   "Stricter" is a meaningless word here, because it points opposite ways for
   different knobs, and two of them are exact mirrors of each other:

     dup_distance          is a DISTANCE  (dHash bits) — RAISE it to catch more
     semantic_dup_threshold is a SIMILARITY (cosine)   — LOWER it to catch more

   Both are "make duplicate detection catch more", and they move in opposite
   directions. A UI that does not say so out loud will get set backwards, and
   backwards looks exactly like "it did nothing". So every field carries the ONE
   unambiguous fact — which way makes this filter catch MORE images — and the
   panel prints it as a sentence next to the input, not as a hidden tooltip.

   Pure functions and data, no JSX, so `node --test` can execute this file
   (it parses .js, not .jsx). The panel itself is BankThresholdsPanel.jsx. */

/** The config section every field below belongs to. Also the PUT payload key. */
export const BANK_SECTION = 'bank';

/* Groups, ordered as they render. Grouped by the QUESTION the user is asking —
   not by the historical order of config.py — because that is how someone
   arrives at these controls: "why is this junk still here?" has a different
   answer from "why are these two copies not grouped?".

   `defaultOpen` decides what is visible without a click. Every group starts
   folded when the parent panel opens, so the user expands only the question
   they need and the panel stays readable at 400 px. */
export const THRESHOLD_GROUPS = [
  { id: 'quality', label: 'Image quality', defaultOpen: false,
    blurb: 'Is the picture technically good enough to train on?' },
  { id: 'duplicates', label: 'Duplicates', defaultOpen: false,
    blurb: 'Have I already got this shot?' },
  { id: 'resolution', label: 'Size & framing', defaultOpen: false,
    blurb: 'Is there enough real picture in the file?' },
  { id: 'content', label: 'Content', defaultOpen: false,
    blurb: 'Is it the right person, and is it worth keeping?' },
  { id: 'style', label: 'Style', defaultOpen: false,
    blurb: 'How close must two images look to count as the same style?' },
];

/* When a change takes effect. The Bank recomputes verdicts from PERSISTED raw
   scores at read time, so most of these re-sort the grid the instant they are
   saved — that is the whole point of storing scores instead of verdicts. The
   four grouping thresholds are baked into stored group ids by their pass, so
   they cannot show a live preview and must say so instead of faking one. */
export const APPLIES = {
  instant: 'Applies immediately — the bank re-sorts, no rescan.',
  scan: 'Applies at the next quality scan.',
  faces: 'Applies at the next face pass.',
  score: 'Applies at the next score pass.',
  dedup: 'Applies at the next ✂ semantic dedup (instant — cached embeddings).',
};

/* The pass that has to re-run for a non-instant threshold to mean anything —
   offered as a BUTTON next to the field, because "applies at the next pass" is
   only half an answer if the next pass is somewhere else on the page.

   The duplicate one is the cheap surprise: a scan with rescan off has an empty
   pool on an already-scanned bank, so re-grouping thousands of images costs a
   walk over stored hashes, no decode. Worth saying, or the button reads as
   "rescan 36 000 files" and nobody presses it.

   `body` is what makes that true AND safe. The quality scan only regroups when
   the hashes it stored actually moved — regrouping 50 000 unchanged hashes at
   the tail of a scan that had 2 images to look at is what froze the app for two
   minutes. This button asks for the grouping on purpose, so it keeps working on
   a bank where the scan itself has nothing to do. */
export const PASS_RERUN = {
  scan: { endpoint: 'scan', body: { regroup: true }, label: '↻ Re-group duplicates',
    note: 'Regroups from the stored hashes — nothing is decoded again.' },
  faces: { endpoint: 'faces', label: '↻ Re-run the face pass',
    note: 'Re-clusters from the cached face embeddings.' },
  score: { endpoint: 'score', label: '↻ Re-run the score pass',
    note: 'Needs the score extra; re-clusters styles.' },
  dedup: { endpoint: 'semantic-dedup', label: '↻ Re-find the same shots',
    note: 'Instant — runs off the cached embeddings.' },
}

/** The pass to offer re-running for this threshold, or null when the change is
 *  already live and there is nothing to run. */
export function rerunFor(t) {
  return t.applies === 'instant' ? null : (PASS_RERUN[t.applies] || null)
}

/* The twelve. `flag` names the Bank flag this threshold decides, and is what
   makes a live "how many images would this catch" count possible: it is the
   same flag key the /bank/<id> payload already counts. Only the read-time
   thresholds have one — the grouping thresholds produce groups, not flags. */
export const BANK_THRESHOLDS = [
  {
    field: 'sharpness_min', group: 'quality', flag: 'blur',
    label: 'Sharpness minimum', unit: 'Laplacian variance',
    step: 10, min: 0, integer: false,
    catchesMoreWhen: 'raised', applies: 'instant',
    hint: 'Laplacian variance of the image\'s SHARPEST regions under this is flagged blurry — a bokeh subject on a soft background stays sharp. ~100 is the classic rule of thumb.',
  },
  {
    field: 'noise_max', group: 'quality', flag: 'noise',
    label: 'Noise maximum', unit: 'residual grain',
    step: 1, min: 0, integer: false,
    catchesMoreWhen: 'lowered', applies: 'instant',
    hint: 'Residual grain over this is flagged noisy.',
  },
  {
    field: 'uniformity_min', group: 'quality', flag: 'uniform',
    label: 'Uniformity minimum', unit: 'grayscale spread',
    step: 1, min: 0, integer: false,
    catchesMoreWhen: 'raised', applies: 'instant',
    hint: 'Grayscale spread under this is flagged flat — solid colours, empty screenshots.',
  },
  {
    field: 'min_side', group: 'resolution', flag: 'small',
    label: 'Minimum side', unit: 'px',
    step: 64, min: 0, integer: true,
    catchesMoreWhen: 'raised', applies: 'instant',
    hint: 'Smaller side under this is flagged small. Trainers only ever downscale, so an image below your training size is lost detail.',
  },
  {
    field: 'detail_min', group: 'resolution', flag: 'soft_detail',
    label: 'Real-detail minimum', unit: '0–1 of the stored size',
    step: 0.02, min: 0, max: 1, integer: false,
    catchesMoreWhen: 'raised', applies: 'instant',
    hint: 'Share of the stored size that still carries real picture; under this is flagged soft detail. The usual cause is an enlargement — but a soft or out-of-focus photo reads the same, so treat it as a score, not proof.',
  },
  {
    field: 'bars_max', group: 'resolution', flag: 'bars',
    label: 'Black-bar maximum', unit: '0–1 of the frame',
    step: 0.01, min: 0, max: 1, integer: false,
    catchesMoreWhen: 'lowered', applies: 'instant',
    hint: 'Share of the frame allowed to be flat black letterbox before black bars — video screenshots, padded stills.',
  },
  {
    field: 'dup_distance', group: 'duplicates', flag: null,
    label: 'Duplicate distance', unit: 'dHash bits (of 64)',
    step: 1, min: 0, max: 16, integer: true,
    catchesMoreWhen: 'raised', applies: 'scan',
    hint: 'How many hash bits two images may differ by and still group as ≈ duplicates. This is a DISTANCE: bigger means more tolerant.',
  },
  {
    field: 'semantic_dup_threshold', group: 'duplicates', flag: null,
    label: 'Semantic duplicate similarity', unit: 'cosine similarity',
    step: 0.01, min: 0, max: 1, integer: false,
    catchesMoreWhen: 'lowered', applies: 'dedup',
    // Spelled out because it is the exact mirror of the field above it.
    hint: 'How alike two scored images must look to count as ✂ the same shot — a crop or a re-compression a hash misses. This is a SIMILARITY, so it moves the OPPOSITE way to the distance above.',
  },
  {
    field: 'face_threshold', group: 'content', flag: null,
    label: 'Same-person similarity', unit: 'cosine similarity',
    step: 0.01, min: 0, max: 1, integer: false,
    catchesMoreWhen: 'lowered', applies: 'faces',
    hint: 'How alike two faces must be to land in the same person cluster. Lower merges more people together; higher splits one person into several clusters.',
  },
  {
    field: 'aesthetic_min', group: 'content', flag: 'low_aesthetic',
    label: 'Aesthetic minimum', unit: 'LAION score, ~1–10',
    step: 0.5, min: 0, max: 10, integer: false,
    catchesMoreWhen: 'raised', applies: 'instant',
    hint: 'LAION aesthetic rating under which an image is flagged low aesthetic. Needs the score pass.',
  },
  {
    field: 'nsfw_max', group: 'content', flag: 'nsfw',
    label: 'NSFW maximum', unit: 'probability, 0–1',
    step: 0.05, min: 0, max: 1, integer: false,
    catchesMoreWhen: 'lowered', applies: 'instant',
    hint: 'NSFW probability over which an image is flagged NSFW, to split a mixed dump. Needs the score pass.',
  },
  {
    field: 'style_threshold', group: 'style', flag: null,
    label: 'Same-style similarity', unit: 'cosine similarity',
    step: 0.01, min: 0, max: 1, integer: false,
    catchesMoreWhen: 'lowered', applies: 'score',
    hint: 'How alike two images must look to share a style cluster. Deliberately looser than the semantic-duplicate bar — a crop is far closer than merely "same style".',
  },
];

/** The fields of one group, in declaration order. */
export function thresholdsInGroup(groupId) {
  return BANK_THRESHOLDS.filter((t) => t.group === groupId);
}

/** Look one up by its config key. */
export function thresholdByField(field) {
  return BANK_THRESHOLDS.find((t) => t.field === field);
}

/** The sentence that kills the direction trap. Deliberately phrased around the
 *  OUTCOME ("catches more images"), never around "stricter" — strict points one
 *  way for a distance and the other way for a similarity, which is exactly how
 *  a duplicate threshold ends up set backwards. */
export function directionHint(t) {
  return t.catchesMoreWhen === 'raised'
    ? '▲ Raise it to catch more images · ▼ lower it to catch fewer'
    : '▼ Lower it to catch more images · ▲ raise it to catch fewer';
}

/** Same fact, compressed for a screen reader / the aria description. */
export function directionSummary(t) {
  return t.catchesMoreWhen === 'raised'
    ? 'Higher values catch more images.'
    : 'Lower values catch more images.';
}

/** Coerce what an <input type="number"> gives back into what the config stores.
 *  Blank stays blank so a half-typed field does not snap to 0 under the cursor;
 *  the caller decides that a blank is not savable. */
export function coerceValue(t, raw) {
  if (raw === '' || raw === null || raw === undefined) return '';
  const n = t.integer ? parseInt(raw, 10) : parseFloat(raw);
  if (!Number.isFinite(n)) return '';
  return n;
}

/** Is this a value we can send to the server? Blank and out-of-range are not:
 *  a bars_max of 4 would flag nothing forever and read as a broken feature. */
export function isValidValue(t, value) {
  if (value === '' || value === null || value === undefined) return false;
  const n = Number(value);
  if (!Number.isFinite(n)) return false;
  if (t.min !== undefined && n < t.min) return false;
  if (t.max !== undefined && n > t.max) return false;
  return true;
}

/** The `bank` section as edited, over the saved one. Values only — never the
 *  descriptions above. */
export function mergedBank(savedBank, edits) {
  return { ...(savedBank || {}), ...(edits || {}) };
}

/** Which fields the user has actually changed and could save. Compares against
 *  the SAVED config, so re-typing the saved number is not a change. */
export function dirtyFields(savedBank, edits) {
  const saved = savedBank || {};
  return Object.keys(edits || {}).filter((k) => {
    const t = thresholdByField(k);
    if (!t || !isValidValue(t, edits[k])) return false;
    return Number(edits[k]) !== Number(saved[k]);
  });
}

/** Fields whose saved value differs from the shipped default — what "Reset all"
 *  would actually touch, and what the group's "customised" badge counts.
 *  `configDefaults` is the SERVER's defaults payload; a section it does not
 *  carry (older backend) yields nothing rather than a guess. */
export function customisedFields(savedBank, configDefaults) {
  const defaults = (configDefaults || {})[BANK_SECTION] || {};
  const saved = savedBank || {};
  return BANK_THRESHOLDS
    .filter((t) => Object.prototype.hasOwnProperty.call(defaults, t.field))
    .filter((t) => Number(saved[t.field]) !== Number(defaults[t.field]))
    .map((t) => t.field);
}

/** The edit map that "Reset all to defaults" applies: every field the server
 *  knows a default for, set back to it. Returns {} when the server sent no
 *  defaults — offering a reset we cannot honour is worse than offering none. */
export function resetAllEdits(configDefaults) {
  const defaults = (configDefaults || {})[BANK_SECTION] || {};
  const out = {};
  for (const t of BANK_THRESHOLDS) {
    if (Object.prototype.hasOwnProperty.call(defaults, t.field)) {
      out[t.field] = defaults[t.field];
    }
  }
  return out;
}

/** How many customised fields sit in one group — the badge that tells you a
 *  folded group is hiding a change you made. */
export function customisedCountInGroup(groupId, savedBank, configDefaults) {
  const custom = new Set(customisedFields(savedBank, configDefaults));
  return thresholdsInGroup(groupId).filter((t) => custom.has(t.field)).length;
}

/** The read-time thresholds, i.e. the ones a live count can be previewed for.
 *  Everything else is baked into stored groups by a pass and would need that
 *  pass re-run to mean anything. */
export function previewableFields() {
  return BANK_THRESHOLDS.filter((t) => t.applies === 'instant' && t.flag)
    .map((t) => t.field);
}

/** The effect line under one input: how many images the CANDIDATE value would
 *  flag, and how that compares to what is flagged today. `counts` is the
 *  preview payload's flag map, `current` the saved payload's.
 *
 *  Returns null when there is nothing honest to say — no flag for this field,
 *  no preview yet, or a bank that has not been scanned. A made-up zero here
 *  would read as "your threshold catches nothing", which is a different and
 *  wrong statement. */
export function effectLine(t, counts, current) {
  if (!t.flag || !counts || !Object.prototype.hasOwnProperty.call(counts, t.flag)) {
    return null;
  }
  const next = counts[t.flag];
  const now = current && Object.prototype.hasOwnProperty.call(current, t.flag)
    ? current[t.flag] : null;
  const n = (v) => Number(v).toLocaleString();
  if (now === null || now === next) return `${n(next)} images flagged`;
  const arrow = next > now ? '↑' : '↓';
  return `${n(now)} → ${n(next)} images flagged ${arrow}`;
}

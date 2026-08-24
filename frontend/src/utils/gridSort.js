/**
 * Grid ORDERING — the companion of `gridStatusFilter.js` / `tagFilter.js`, which
 * only ever narrow a list. Asked for by nofaceman (Discord): both the bank and a
 * dataset already MEASURE things (aesthetic rating, sharpness, face similarity)
 * and let you filter on them, but nothing could put the best — or the worst —
 * in front of your eyes first, which is what makes a review fast.
 *
 * The bank menu now covers EVERY quantity its passes persist, both ways (asked
 * again on Discord after the first three shipped). The rationale for going wide
 * is in image_bank_service._SORT_KEYS: a chip ranks only the rows that cross its
 * threshold, so it can never answer "the noisiest of the ones I am keeping" nor
 * anything asked the other way round.
 *
 * Two surfaces, two mechanics, ONE contract:
 *  • Bank — the grid is paginated over thousands of rows, so the order is a SQL
 *    `sort=` parameter (see image_bank_service._sort_order). This file only owns
 *    the MENU: which entries exist and which ones are honestly usable right now.
 *  • Dataset — the workspace already holds every image of the dataset in memory
 *    (dataset_payload ships them all, and both existing grid filters are
 *    client-side), so sorting here is a pure array transform. Sending it to SQL
 *    would mean a refetch per click for no gain and a second, divergent order.
 *
 * The contract both share:
 *  • A value the matching pass never produced (NULL / undefined) SINKS TO THE END
 *    in BOTH directions. "Worst first" must open on the worst MEASURED image, not
 *    on the pile nobody measured — that is the same decision the resolution sort
 *    already took ("Unscanned images sink to the end").
 *  • An entry whose data does not exist yet is OFFERED BUT DISABLED, with the
 *    reason in its own label — same precedent as the 👥 Group by person button
 *    ("Install the Quality tools (Setup) to sort by person"). A sort that
 *    silently reorders nothing reads as a broken app.
 *  • Sorting NEVER changes membership: it composes with every filter, and the
 *    filtered list keeps exactly the same images.
 *
 * `id`s are stored (bank query string, dataset localStorage) — never rename one
 * without an alias, per the repo rule on stored identifiers.
 */

/** Bank sort entries, in menu order. `needs` names the payload count that must
 *  be > 0 for the entry to do anything; `missing` is the reason shown when it is
 *  0 (appended to the label, because a disabled <option>'s title is invisible on
 *  most platforms). */
const SCAN = { needs: 'scanned', missing: 'run 🔎 Scan quality', group: '🔎 Scan quality' };
const SCORE = { needs: 'scored', missing: 'run ✨ Score', group: '✨ Score' };
const FACE = { needs: 'faces', missing: 'run 🎭 Faces', group: '🎭 Faces' };
const FILE = { needs: 'scanned', missing: 'run 🔎 Scan quality', group: '📁 File' };
// The head angle comes from the 🎭 Faces pass but has its OWN readiness count:
// a bank scanned before the pass kept the yaw has thousands of faces and zero
// measured angles, so gating this entry on `faces` would offer a sort that
// reorders nothing. `angle_measured` is the only honest gate.
const ANGLE = { needs: 'angle_measured', missing: 'measure head angles',
  group: '🎭 Faces' };
const MEDIUM = { needs: 'medium_classified', missing: 'run 🎨 Medium',
  group: '🎨 Medium' };

/** One measure = two entries (↓ then ↑). `hi`/`lo` are what each direction puts
 *  in front of you — written as the REASON to pick it, not as a restatement of
 *  the arrow, because "why would I click this" is the only question the menu has
 *  to answer. */
const measure = (key, label, pass, hi, lo) => [
  { id: `${key}_desc`, label: `${label} ↓`, title: hi, ...pass },
  { id: `${key}_asc`, label: `${label} ↑`, title: lo, ...pass },
];

export const BANK_SORTS = [
  { id: 'default', label: 'Default', group: '',
    title: 'The bank\'s own order (or the worst-first order of the active flag chip)' },
  ...measure('res', 'Resolution', FILE,
    'Biggest first, in megapixels (width × height)',
    'Smallest first — surfaces the thumbnails hiding in a big dump'),
  ...measure('size', 'File size', FILE,
    'Heaviest files first — what a 9 000-image folder is actually made of',
    'Lightest files first — re-saved and re-compressed copies come up together'),
  ...measure('aesthetic', 'Aesthetic', SCORE,
    'Best-looking first — the keepers, from the ✨ Score pass',
    'Worst-looking first — the fastest way to prune, from the ✨ Score pass'),
  ...measure('nsfw', 'NSFW likelihood', SCORE,
    'Most explicit first — from the ✨ Score pass',
    'Safest first — the chip only shows what crosses the threshold, this shows the rest'),
  ...measure('sharp', 'Sharpness', SCAN,
    'Sharpest first, from the 🔎 Scan quality pass',
    'Blurriest first — the misses come to you, from the 🔎 Scan quality pass'),
  ...measure('noise', 'Noise', SCAN,
    'Grainiest first — high-ISO and heavily upscaled shots',
    'Cleanest first'),
  ...measure('flat', 'Contrast', SCAN,
    'Most contrasted first',
    'Flattest first — blank, washed-out and near-empty frames'),
  ...measure('detail', 'Detail', SCAN,
    'Most real detail first — full effective resolution',
    'Softest first — the enlargements pretending to be big images'),
  ...measure('bars', 'Letterbox bars', SCAN,
    'Most black bars first — the video screenshots',
    'No bars first'),
  ...measure('jpeg', 'JPEG quality', SCAN,
    'Best-saved first',
    'Most compressed first — the re-shared, re-saved copies'),
  ...measure('face', 'Face confidence', FACE,
    'Clearest faces first — from the 🎭 Faces pass',
    'Weakest faces first — tiny, turned or half-hidden faces'),
  // The head angle is ordered on its ABSOLUTE value: "turned left" and "turned
  // right" are one shot type, and splitting them would halve every answer for a
  // distinction no training set makes.
  ...measure('yaw', 'Head angle', ANGLE,
    'Most turned away first — the profiles and three-quarters, from the 🎭 Faces pass',
    'Most face-on first'),
  // The one sort whose USEFUL direction is ↑: it opens on the images the medium
  // classifier nearly could not call, which is exactly the pile worth a human
  // glance. A chip can only show you what crossed the bar; this shows you what
  // sat on it.
  ...measure('medium_conf', 'Medium confidence', MEDIUM,
    'Surest medium verdicts first — from the 🎨 Medium pass',
    'Least sure first — the images the classifier nearly could not call'),
];

const DEFAULT_BANK_SORT = 'default';

const BANK_IDS = new Set(BANK_SORTS.map((s) => s.id));

/** Normalise a stored/URL bank sort id (stale localStorage, a value from a build
 *  that offered a sort this one dropped, undefined…) to a usable one. */
export function normalizeBankSort(value) {
  return (typeof value === 'string' && BANK_IDS.has(value))
    ? value : DEFAULT_BANK_SORT;
}

/** localStorage key for a bank's remembered order. Per BANK, not global: the
 *  order that suits a 9 000-image scrape dump is not the one that suits a
 *  hand-picked set, and re-choosing it on every visit was the friction. */
export function bankSortStorageKey(bankId) {
  return `lds.bank.${bankId}.sort`;
}

/** localStorage, or nothing at all: this module is imported by node tests and by
 *  any non-browser consumer, and a private-mode browser throws on ACCESS to
 *  localStorage, not only on read. `store` is injectable so the behaviour is
 *  testable without a DOM. */
function defaultStore() {
  try {
    return globalThis.localStorage || null;
  } catch { return null; }
}

/** This bank's remembered order, or 'default'. Anything unreadable, absent or no
 *  longer offered degrades to 'default' — a stored preference must never be able
 *  to leave the grid in an order the server would reject. */
export function loadBankSort(bankId, store = defaultStore()) {
  try {
    return normalizeBankSort(store?.getItem(bankSortStorageKey(bankId)));
  } catch { return DEFAULT_BANK_SORT; }
}

/** Remember (or, for 'default', forget) this bank's order. Storage failures are
 *  swallowed: a full quota must not break a click that otherwise worked. */
export function saveBankSort(bankId, sortId, store = defaultStore()) {
  const id = normalizeBankSort(sortId);
  try {
    if (id === DEFAULT_BANK_SORT) store?.removeItem(bankSortStorageKey(bankId));
    else store?.setItem(bankSortStorageKey(bankId), id);
  } catch { /* private mode / quota — the session still has the right order */ }
  return id;
}

/**
 * The bank's Sort menu for a given payload `counts` ({scanned, scored, …}).
 * An entry is disabled ONLY when we positively know its pass reached 0 images:
 * while the payload is still loading (`counts` absent) nothing is greyed out,
 * because "I don't know yet" must never be shown as "you can't".
 */
export function bankSortOptions(counts) {
  return BANK_SORTS.map((s) => {
    const known = counts && typeof counts[s.needs] === 'number';
    const empty = Boolean(s.needs && known && counts[s.needs] <= 0);
    return {
      id: s.id,
      group: s.group ?? '',
      label: empty ? `${s.label} — ${s.missing} first` : s.label,
      title: empty ? `Nothing measured yet — ${s.missing} to sort by this.` : s.title,
      disabled: empty,
    };
  });
}

/**
 * The same menu, cut into the passes that produce it ([{group, options}], groups
 * in first-appearance order). Eleven measures × two directions is a long flat
 * list; grouped by the pass that MEASURED them, it is a short list of familiar
 * names — and it tells the user which pass to run to unlock a whole section.
 * The ungrouped head (Default) comes back with group ''.
 */
export function bankSortGroups(counts) {
  const groups = [];
  const byName = new Map();
  for (const o of bankSortOptions(counts)) {
    if (!byName.has(o.group)) {
      const entry = { group: o.group, options: [] };
      byName.set(o.group, entry);
      groups.push(entry);
    }
    byName.get(o.group).options.push(o);
  }
  return groups;
}

/* -------------------------------------------------------------------------- */

/** The four shot types, in the order the app already speaks them everywhere
 *  else: the composition target (12/6/6/1), the variation catalog groups, the
 *  framing column's own enum. Ordering the grid on this INDEX rather than
 *  alphabetically is what makes the sort a grouping — equal keys land next to
 *  each other, so "all the bust shots" become one run you can read across. */
const FRAMING_ORDER = ['face', 'bust', 'body', 'back'];

const framingRank = (img) => {
  const i = FRAMING_ORDER.indexOf(img?.framing);
  return i < 0 ? null : i;    // 'unknown', NULL, anything unclassified: no rank
};

/** What a dataset row has to CARRY for an entry to do anything, and the pass
 *  that puts it there. Same doctrine as the bank menu above: an entry whose data
 *  does not exist is offered but DISABLED, naming the pass to run — a sort that
 *  silently reorders nothing reads as a broken app. */
const NEEDS = {
  face: {
    has: (i) => faceScore(i) !== null,
    missing: 'run 🎭 Analyze faces first',
    reason: 'No image has a face score yet — run 🎭 Analyze faces (Curation) to sort by this.',
  },
  framing: {
    has: (i) => framingRank(i) !== null,
    missing: 'run 📐 Classify framing first',
    reason: 'No image has a shot type yet — run 📐 Classify framing (under the Composition bar) to group by this.',
  },
};

/** Dataset sort entries.
 *
 *  Two families, and they answer different questions. The MEASURED one (face
 *  similarity) ranks the whole grid best-to-worst; the CATEGORICAL one (shot
 *  type) does not rank anything, it puts like next to like. Asked for by
 *  .samexit (Discord) three times over four days: the generated list mixes face,
 *  bust, body and back shots together, which is exactly the wrong arrangement
 *  for the only question being asked at that moment — "do I have too many of
 *  these, not enough of those, and which of these near-identical ones do I
 *  keep?". Nothing else a dataset row carries is offered here: aesthetics and
 *  sharpness are BANK columns, so a menu entry for them could only ever be
 *  greyed out. */
export const DATASET_SORTS = [
  { id: 'default', label: 'Newest first',
    title: 'The order images were added — most recent first' },
  { id: 'face_desc', label: 'Face similarity ↓', needs: ['face'],
    title: 'Closest to your reference photo first — from 🎭 Analyze faces' },
  { id: 'face_asc', label: 'Face similarity ↑', needs: ['face'],
    title: 'Least like your reference first — the ones to cut, from 🎭 Analyze faces' },
  { id: 'shot', label: 'Shot type', needs: ['framing'],
    title: 'Face, then bust, then body, then back — every shot of one kind in a single run, newest first inside each' },
  { id: 'shot_face', label: 'Shot type, then face similarity ↓', needs: ['framing', 'face'],
    title: 'The same grouping, with the closest to your reference at the head of each kind — the order for deciding what to keep' },
];

export const DEFAULT_DATASET_SORT = 'default';

const DATASET_IDS = new Set(DATASET_SORTS.map((s) => s.id));

/** Normalise anything (stale localStorage, undefined…) to a usable sort id. */
export function normalizeDatasetSort(value) {
  return (typeof value === 'string' && DATASET_IDS.has(value))
    ? value : DEFAULT_DATASET_SORT;
}

const faceScore = (img) => {
  const v = img?.face_score;
  return (typeof v === 'number' && Number.isFinite(v)) ? v : null;
};

/** Each entry is a LIST of ordering steps, applied in order, the next one
 *  breaking the ties the previous left. One step is the historical case; two is
 *  what turns a grouping into something you can also act on. */
const DATASET_SORT_SPECS = {
  face_desc: [{ value: faceScore, direction: 'desc' }],
  face_asc: [{ value: faceScore, direction: 'asc' }],
  shot: [{ value: framingRank, direction: 'asc' }],
  shot_face: [
    { value: framingRank, direction: 'asc' },
    { value: faceScore, direction: 'desc' },
  ],
};

/**
 * Reorder the dataset grid. Returns the SAME array reference for 'default' and
 * for an unknown id (cheap no-op, like filterImagesByStatus) — the payload
 * already arrives newest-first. Never adds or drops an image.
 *
 * A value the matching pass never produced (no face_score, no shot type) SINKS
 * TO THE END of the step that reads it, in both directions — the same rule the
 * bank sorts follow, and the reason "worst first" opens on the worst MEASURED
 * image rather than on the pile nobody measured. On a two-step order that sink
 * is per step: the unclassified images gather at the end of the grid, and inside
 * each shot type the unscored ones gather at the end of their own run.
 *
 * Ties left by every step fall back to the default newest-first order, so the
 * result is deterministic.
 */
export function sortDatasetImages(images, sortId) {
  const list = images || [];
  const steps = DATASET_SORT_SPECS[normalizeDatasetSort(sortId)];
  if (!steps || list.length < 2) return list;
  return [...list].sort((a, b) => {
    for (const step of steps) {
      const dir = step.direction === 'asc' ? 1 : -1;
      const va = step.value(a);
      const vb = step.value(b);
      if ((va === null) !== (vb === null)) return va === null ? 1 : -1;
      if (va !== null && va !== vb) return (va - vb) * dir;
    }
    return (b.id || 0) - (a.id || 0);
  });
}

/** The dataset's Sort menu. An entry is disabled — with the reason, and naming
 *  the pass to run — while NO image of the set carries what it orders on, so
 *  neither "sort by similarity" nor "group by shot type" can ever look broken.
 *  An entry needing two things names the FIRST one still missing, because a
 *  sentence listing two passes is a sentence nobody finishes reading. */
export function datasetSortOptions(images) {
  const list = images || [];
  const met = Object.fromEntries(
    Object.entries(NEEDS).map(([key, need]) => [key, list.some(need.has)]));
  return DATASET_SORTS.map((s) => {
    const lacking = (s.needs || []).find((key) => !met[key]);
    const need = lacking ? NEEDS[lacking] : null;
    return {
      id: s.id,
      label: need ? `${s.label} — ${need.missing}` : s.label,
      title: need ? need.reason : s.title,
      disabled: !!need,
    };
  });
}

/* Editable "built-in default" prompts — the ONE rule both surfaces obey.
   PURE JS (no JSX) so node --test can import and exercise it directly.

   WHY THIS FILE EXISTS
   --------------------
   These prompts (config `identity_prompts.*`) are stored as a GLOBAL override
   whose blank value means "follow the shipped default". The backend enforces
   that literally: `face_variations.get_identity_prompt` only honours an override
   when it holds non-blank text, otherwise it returns the hardcoded constant
   byte-for-byte, and `config.DEFAULTS` ships each string as '' on purpose.

   The UI used to expose that as TWO boxes — an empty field plus a read-only
   "Built-in default (currently in use)" block with a "Load default to edit"
   button. One box is better, but pre-filling a single box with the default text
   would quietly PERSIST A COPY of it: the user follows the default today, saves
   an identical string, and every future improvement to the shipped prompt stops
   reaching them, silently and forever.

   So the single box SHOWS the default and is editable, and this module is what
   keeps the storage honest: text equal to the shipped default (ignoring only
   surrounding whitespace) normalises back to '' — "following the default" —
   and only genuinely different text is stored as an override. Every onChange
   goes through `normalizePromptOverride`, so the config never holds a frozen
   copy at any point, not just at save time. */

/** The value to STORE for a prompt box holding `value`, given the shipped
 *  `defaultText`. '' means "follow the built-in default" (the backend contract).
 *  Blank input, or input equal to the default up to surrounding whitespace,
 *  collapses to '' — anything else is a real override, kept verbatim. */
export function normalizePromptOverride(value, defaultText) {
  const raw = typeof value === 'string' ? value : '';
  if (!raw.trim()) return '';
  const def = typeof defaultText === 'string' ? defaultText : '';
  if (raw === def || raw.trim() === def.trim()) return '';
  return raw;
}

/** True when the stored value means "use the shipped default". */
export function isFollowingDefault(value, defaultText) {
  return normalizePromptOverride(value, defaultText) === '';
}

/** What the single box DISPLAYS: the override when there is one, otherwise the
 *  real shipped default (never an empty box behind a placeholder). */
export function promptBoxText(value, defaultText) {
  const raw = typeof value === 'string' ? value : '';
  if (raw) return raw;
  return typeof defaultText === 'string' ? defaultText : '';
}

/* Shared metadata for the editable identity prompts — imported by BOTH the
   Settings card and the workspace modal so the two surfaces can never drift
   apart on labels, keys or which engine a prompt actually drives.
   `key` mirrors config identity_prompts.* — NEVER renamed (persisted globally).
   `engines` says which engine family really consumes the prompt, verified in
   face_variations.py: wrap_variation picks face_multi/face_single for the API
   engines, and BOTH local engines share `_compose_edit_prompt`, which always
   reads klein_identity — wrap_variation_klein and wrap_variation_krea alike.
   Hence "Local engines", not "Klein": the box was named after one of its two
   consumers, and a user asked on Discord whether it applied to Krea 2 at all.
   A prompt this file names after one engine is a prompt the other engine's
   users will leave alone. */

/** The engines whose prompts go through wrap_variation, i.e. every API engine.
 *  Listed once: an engine missing from here would silently be treated as Klein
 *  by activeExtraRefPromptKey and badge the wrong prompt box. */
export const API_PROMPT_ENGINES = ['nanobanana', 'chatgpt', 'openrouter'];

export const IDENTITY_PROMPT_FIELDS = [
  {
    key: 'face_single',
    id: 'identity-prompt-face-single',
    label: 'API engine — identity lock (single reference)',
    engines: API_PROMPT_ENGINES,
    desc: 'Prepended to every Nano Banana / ChatGPT / OpenRouter variation made from ONE reference photo. Tells the model to keep the exact face and take outfit + expression from the description, not the reference.',
  },
  {
    key: 'face_multi',
    id: 'identity-prompt-face-multi',
    label: 'API engine — identity lock (multiple references)',
    engines: API_PROMPT_ENGINES,
    desc: 'Same, but for variations generated from SEVERAL reference photos of the person — tells the model all references are the same person and to use them together.',
  },
  {
    key: 'klein_identity',
    id: 'identity-prompt-klein-identity',
    label: 'Local engines — restage & face-identity block',
    engines: ['klein', 'krea'],
    desc: 'The instruction block BOTH local engines — Klein and Krea 2 Edit — use to restage the shot while keeping the face identical. Steers pose/framing/outfit changes without altering the person.',
  },
];

/* The descriptions above name a FACE and a person — true of the human set only.
   A box shown for an Animal dataset that says "keep the exact face" is what
   invites a user to rewrite it for animals; the wording has to follow the
   subject. Only the human strings above stay byte-identical (they are the ones
   users already know); the rest are derived per subject. */
const SUBJECT_NOUNS = {
  animal: { one: 'the animal', kind: 'animal', trait: 'coat, markings and build' },
  creature: { one: 'the creature', kind: 'creature', trait: 'body form, texture and features' },
  object: { one: 'the object', kind: 'object', trait: 'shape, colour and materials' },
  other: { one: 'the subject', kind: 'subject', trait: 'shape, colours and details' },
  // A drawn character's "traits" are design choices, not physical ones — and the
  // art style is one of them, which is why this lock also has to forbid the
  // photorealism every other lock asks for.
  anime: { one: 'the character', kind: 'character',
    trait: 'hair, eyes, signature outfit, accessories and drawn art style' },
};

/** The three editable identity fields, worded for `subjectType`. Keys, ids and
 *  engine mapping never change — only the human-readable text does. */
export function identityPromptFields(subjectType) {
  const st = normalizeSubject(subjectType);
  if (st === 'human') return IDENTITY_PROMPT_FIELDS;
  const n = SUBJECT_NOUNS[st];
  const descs = {
    face_single: `Prepended to every Nano Banana / ChatGPT variation made from ONE reference photo of ${n.one}. Tells the model to preserve its ${n.trait}, and to take the pose and setting from the description, not the reference.`,
    face_multi: `Same, but for variations generated from SEVERAL reference photos of the same ${n.kind} — tells the model they all show one ${n.kind} and to use them together.`,
    klein_identity: `The instruction block BOTH local engines — Klein and Krea 2 Edit — use to restage the shot while keeping ${n.one} identical. Steers pose/framing/setting changes without altering its ${n.trait}.`,
  };
  const labels = {
    face_single: 'API engine — identity lock (single reference)',
    face_multi: 'API engine — identity lock (multiple references)',
    klein_identity: `Local engines — restage & ${n.kind}-identity block`,
  };
  return IDENTITY_PROMPT_FIELDS.map((f) => ({
    ...f, label: labels[f.key] || f.label, desc: descs[f.key] || f.desc,
  }));
}

/* --- Per-subject-type storage ------------------------------------------------
   An identity lock is written FOR A SUBJECT. Storing one text for all of them is
   what let a prompt tuned on an Animal dataset ride on human generations —
   tails, extra limbs, odd footwear (reported by ashish.sinha on Discord).

   Layout, mirroring backend face_variations.identity_prompt_config_key:
     human      -> identity_prompts.<kind>                    (the ORIGINAL key)
     non-human  -> identity_prompts.by_subject.<type>.<kind>
   Human deliberately keeps the flat key: every override written before this fix
   landed there (the editor showed human text and had no subject selector), so
   reading it as the human override preserves it with no migration. A non-human
   subject NEVER falls back to it — that fallback is the bug.
   `klein_improve` (+ its toggle) stays flat and global: it is a subject-agnostic
   quality instruction, identical in all five default tables. */

/** Which subject types own their own copy of the identity locks. */
export const PROMPT_SUBJECT_TYPES = ['human', 'animal', 'creature', 'object', 'other', 'anime'];

/** The kinds scoped per subject — mirrors backend PER_SUBJECT_PROMPT_KINDS. */
export const PER_SUBJECT_PROMPT_KINDS = [
  'face_single', 'face_multi', 'klein_identity',
  // The rendering tail and the framing detail joined them: anime's tail asks for
  // an illustration where every photographic type asks for a photograph, and the
  // framing blocks are six different tables already.
  'render_tail_sfw', 'render_tail_nsfw',
  'framing_face', 'framing_bust', 'framing_body', 'framing_back',
];

/* --- The five parts that used to be hardcoded --------------------------------
   The identity locks were only ONE of the six sources a local-edit prompt is
   assembled from. These are the others: they shipped in every prompt with no way
   to see or change them, and one of them (the markings hold order) caused a live
   incident. Same storage contract as the locks — blank means "shipped default" —
   so a field left alone keeps receiving improvements.

   `id` is the DOM id (also the help-registry focus target when a topic points at
   a single field); `key` mirrors the config key and is NEVER renamed. */

/** Global parts — one text for every subject type. `_augment_prompt` only ever
 *  runs on the human catalog, so the two directives and the garment list have no
 *  per-subject meaning to split. */
export const GLOBAL_PROMPT_PART_FIELDS = [
  {
    key: 'markings_lock',
    id: 'prompt-part-markings-lock',
    label: 'Local engines — hold the skin (Krea)',
    engines: ['krea'],
    rows: 3,
    desc: 'Sent with every Krea prompt. It stops the model from inventing marks on the skin, or redrawing the ones the reference already has, which is what made tattoos come back different on every shot.',
    // The incident, in one sentence, at the point of edit. Naming a feature in
    // this box is what summons it — the earlier wording enumerated "tattoos…"
    // and the model painted them on subjects who had none.
    warn: 'Careful with this one. An earlier version listed what to preserve — “tattoos, scars, moles…” — and the model started painting tattoos on people who have none: naming a feature is enough to summon it. Describe what NOT to do (add, redraw, move, remove) without naming a single body feature.',
  },
  {
    key: 'outfit_vary',
    id: 'prompt-part-outfit-vary',
    label: 'Every shot — outfit directive',
    engines: [...API_PROMPT_ENGINES, 'klein'],
    rows: 3,
    desc: 'Added to every human shot that does not already name a garment, so the model dresses the subject from the description instead of copying the reference outfit. Krea replaces it with a concrete garment from the palette below, so editing this text does not change what Krea sends.',
  },
  {
    key: 'expression_neutral',
    id: 'prompt-part-expression-neutral',
    label: 'Every shot — expression directive',
    engines: [...API_PROMPT_ENGINES, 'klein'],
    rows: 3,
    desc: 'Added to every human shot that does not already name an expression, so the reference’s smile or grimace does not ride on all 40 variations.',
  },
  {
    key: 'outfit_palette',
    id: 'prompt-part-outfit-palette',
    label: 'Krea — concrete garments (one per line)',
    engines: ['krea'],
    rows: 8,
    desc: 'Krea keeps whatever it is not positively told to change, so “a different outfit” does nothing on it: each shot is given a real garment from this list instead.',
    warn: 'The garment is picked from the shot’s name, by position in this list — so ADDING OR REMOVING A LINE reshuffles which garment every shot gets. Same shots, different clothes. Editing the wording of a line only changes that one. Leave the box empty to go back to the shipped list.',
  },
];

/** Parts scoped per subject type — shown under the subject chips. */
export const SUBJECT_PROMPT_PART_FIELDS = [
  {
    key: 'render_tail_sfw',
    id: 'prompt-part-render-tail-sfw',
    label: 'Local engines — rendering tail (SFW)',
    engines: ['klein', 'krea'],
    rows: 2,
    desc: 'The last thing Klein and Krea read on a safe-for-work shot — the medium and the clamp. This is where “Professional realistic photograph” lives (and, for Anime, the instruction to stay a drawing).',
  },
  {
    key: 'render_tail_nsfw',
    id: 'prompt-part-render-tail-nsfw',
    label: 'Local engines — rendering tail (uncensored)',
    engines: ['klein', 'krea'],
    rows: 2,
    desc: 'The same tail on an uncensored shot: it drops the SFW clamp and asks for anatomically correct forms. Only the local engines ever see it — the API engines refuse this content.',
  },
];

/** The per-framing detail block, one field per framing. Klein and Krea under-fill
 *  a terse tag prompt; this is the block that says what the shot should look
 *  like. The four framings are the internal enum — never renamed. */
export const FRAMING_PROMPT_PART_FIELDS = [
  { key: 'framing_face', id: 'prompt-part-framing-face', label: 'Face / close-up' },
  { key: 'framing_bust', id: 'prompt-part-framing-bust', label: 'Bust / half-length' },
  { key: 'framing_body', id: 'prompt-part-framing-body', label: 'Full body' },
  { key: 'framing_back', id: 'prompt-part-framing-back', label: 'From behind' },
].map((f) => ({ ...f, engines: ['klein', 'krea'], rows: 3 }));

/** Every editable prompt-part key, in the order the card shows them — the list
 *  the preview panel and the "customised" dot both read, so a field added above
 *  is never forgotten by one of them. */
export const PROMPT_PART_KEYS = [
  ...GLOBAL_PROMPT_PART_FIELDS, ...SUBJECT_PROMPT_PART_FIELDS, ...FRAMING_PROMPT_PART_FIELDS,
].map((f) => f.key);

function normalizeSubject(subjectType) {
  const s = String(subjectType || 'human').toLowerCase();
  return PROMPT_SUBJECT_TYPES.includes(s) ? s : 'human';
}

function isFlat(subjectType, kind) {
  return normalizeSubject(subjectType) === 'human'
    || !PER_SUBJECT_PROMPT_KINDS.includes(kind);
}

/** The stored override for (subjectType, kind) inside a config `identity_prompts`
 *  object — '' / undefined meaning "follow the shipped default". */
export function readIdentityPrompt(identityPrompts, subjectType, kind) {
  const ip = identityPrompts || {};
  if (isFlat(subjectType, kind)) return ip[kind];
  return ((ip.by_subject || {})[normalizeSubject(subjectType)] || {})[kind];
}

/** A NEW `identity_prompts` object with (subjectType, kind) set to `value`.
 *  Immutable: React state stays replaced, never mutated in place. */
export function writeIdentityPrompt(identityPrompts, subjectType, kind, value) {
  const ip = { ...(identityPrompts || {}) };
  if (isFlat(subjectType, kind)) {
    ip[kind] = value;
    return ip;
  }
  const st = normalizeSubject(subjectType);
  const by = { ...(ip.by_subject || {}) };
  by[st] = { ...(by[st] || {}), [kind]: value };
  ip.by_subject = by;
  return ip;
}

/** The PARTIAL `identity_prompts` patch that saves `kinds` for ONE subject type.
 *  /api/settings deep-merges, so a save from the workspace touches only the
 *  prompts of the dataset's own subject — never another subject's texts. */
export function identityPromptPatch(subjectType, kinds, identityPrompts) {
  let patch = {};
  for (const k of kinds) {
    patch = writeIdentityPrompt(
      patch, subjectType, k, readIdentityPrompt(identityPrompts, subjectType, k) ?? '');
  }
  return patch;
}

/** The shipped defaults for one subject type, from a /api/settings payload.
 *  Falls back to the flat human map so an older payload (or a failed load) still
 *  shows real text instead of an empty box. */
export function identityDefaultsFor(payload, subjectType) {
  const p = payload || {};
  const st = normalizeSubject(subjectType);
  const bySubject = p.identity_prompt_defaults_by_subject || {};
  return bySubject[st] || p.identity_prompt_defaults || {};
}

/** True when this subject type carries at least one real override — drives the
 *  "edited" dot on the Settings subject selector, so a user can SEE that another
 *  subject holds a custom prompt instead of discovering it through a bad render. */
export function subjectHasOverride(identityPrompts, subjectType) {
  return PER_SUBJECT_PROMPT_KINDS.some((k) => {
    const v = readIdentityPrompt(identityPrompts, subjectType, k);
    return typeof v === 'string' && v.trim() !== '';
  });
}

/** The multi-reference identity prompts, in the order the Extra-refs modal
 *  shows them. Extra references mean ref_count > 1, so the API engines take
 *  `face_multi`; Klein takes `klein_identity` whatever the reference count.
 *  Editing only one of the two would let a Klein user rewrite a text with NO
 *  effect on their generations — hence both, each labelled with its engine. */
export const EXTRA_REF_PROMPT_KEYS = ['face_multi', 'klein_identity'];

/** Which of the two the currently selected engine actually uses.
 *  `generator` is the workspace's persisted engine id (localStorage
 *  `datasetGenerator`, the same source VariationCatalog reads). The mapping
 *  MIRRORS VariationCatalog exactly, or the badge would point at the wrong box:
 *  nothing stored yet falls back to 'nanobanana' (its useState default), and any
 *  other value — including a legacy/unknown one — is Klein (its `isKlein` is
 *  "neither API engine"). */
export function activeExtraRefPromptKey(generator) {
  const g = String(generator || 'nanobanana').toLowerCase();
  return API_PROMPT_ENGINES.includes(g) ? 'face_multi' : 'klein_identity';
}

/** Same question for a MULTI-engine selection: every key at least one selected
 *  engine consumes. A run on Klein + ChatGPT really does use BOTH texts, so
 *  badging only the primary engine's box would tell the user the other one
 *  doesn't matter — the exact misunderstanding this badge exists to prevent.
 *  An empty/unknown selection falls back to the single-engine answer, so the
 *  badge never disappears on a legacy profile. */
export function activeExtraRefPromptKeys(generators) {
  const list = Array.isArray(generators) ? generators : [generators];
  const keys = new Set(list.filter(Boolean).map(activeExtraRefPromptKey));
  if (!keys.size) keys.add(activeExtraRefPromptKey(null));
  return EXTRA_REF_PROMPT_KEYS.filter((k) => keys.has(k));
}

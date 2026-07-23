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
   engines, wrap_variation_klein always uses klein_identity. */
export const IDENTITY_PROMPT_FIELDS = [
  {
    key: 'face_single',
    id: 'identity-prompt-face-single',
    label: 'API engine — identity lock (single reference)',
    engines: ['nanobanana', 'chatgpt'],
    desc: 'Prepended to every Nano Banana / ChatGPT variation made from ONE reference photo. Tells the model to keep the exact face and take outfit + expression from the description, not the reference.',
  },
  {
    key: 'face_multi',
    id: 'identity-prompt-face-multi',
    label: 'API engine — identity lock (multiple references)',
    engines: ['nanobanana', 'chatgpt'],
    desc: 'Same, but for variations generated from SEVERAL reference photos of the person — tells the model all references are the same person and to use them together.',
  },
  {
    key: 'klein_identity',
    id: 'identity-prompt-klein-identity',
    label: 'Klein — restage & face-identity block',
    engines: ['klein'],
    desc: 'The instruction block Klein (local) uses to restage the shot while keeping the face identical. Steers pose/framing/outfit changes without altering the person.',
  },
];

/** The multi-reference identity prompts, in the order the Extra-refs modal
 *  shows them. Extra references mean ref_count > 1, so the API engines take
 *  `face_multi`; Klein takes `klein_identity` whatever the reference count.
 *  Editing only one of the two would let a Klein user rewrite a text with NO
 *  effect on their generations — hence both, each labelled with its engine. */
export const EXTRA_REF_PROMPT_KEYS = ['face_multi', 'klein_identity'];

/** Which of the two the currently selected engine actually uses.
 *  `generator` is the workspace's persisted engine id (localStorage
 *  `datasetGenerator`, same source VariationCatalog reads); anything that is
 *  not one of the two API engines is Klein, exactly like VariationCatalog. */
export function activeExtraRefPromptKey(generator) {
  const g = String(generator || '').toLowerCase();
  return g === 'nanobanana' || g === 'chatgpt' ? 'face_multi' : 'klein_identity';
}

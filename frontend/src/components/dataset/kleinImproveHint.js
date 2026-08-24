/* What "✨ Upscale & improve" is about to ASK Klein, said where the button is.
   PURE JS (JSX-free) so node --test can import it.

   THE REPORTED PROBLEM (Qeeyana, Reddit). "Using the Klein quality inpaint often
   harms the image by making textures realistic despite images being anime […]
   would help to be able to tweak the settings. Maybe there is but I can't find
   it." Both halves were true at once: the levers all existed (the instruction
   itself, its on/off toggle, four strength knobs) and NONE of them was named on
   the screen where the damage happens. The lightbox carried a single "Adjust
   improve strength →" link, which points at the knobs — not at the sentence that
   was actually doing it.

   And the sentence is the whole story. The shipped default reads "add detailed
   texture, add sharp details, add candid shot, add soft focus effect": a recipe
   for a photograph, applied identically to every dataset, anime included. A user
   who READS those eight words needs no documentation to understand why their
   drawing came back with skin pores. So the fix is not another link — it is
   quoting the instruction, live, next to the button.

   LIVE, not the shipped text: the box is editable and switchable off, and a
   hint that describes the default to someone who has already changed it is worse
   than no hint. Hence the {loaded, enabled, prompt} shape — everything here is a
   pure function of what the server actually holds. */

/**
 * The EFFECTIVE improve instruction, read from a /api/settings payload.
 * Mirrors the backend contract exactly (face_variations.get_identity_prompt +
 * face_dataset_service._improve_prompt): `klein_improve` is a FLAT key (it is
 * subject-agnostic, unlike its neighbours), blank means "use the shipped
 * default", and `klein_improve_enabled === false` means no prompt at all.
 * A payload that has not arrived (or that failed) yields loaded:false, never a
 * guess — see IMPROVE_LINE_UNKNOWN.
 */
export function readImproveInstruction(payload) {
  if (!payload || typeof payload !== 'object') {
    return { loaded: false, enabled: true, prompt: '' };
  }
  const ip = (payload.config && payload.config.identity_prompts) || {};
  const override = typeof ip.klein_improve === 'string' ? ip.klein_improve : '';
  const shipped = (payload.identity_prompt_defaults || {}).klein_improve || '';
  return {
    loaded: true,
    enabled: ip.klein_improve_enabled !== false,
    prompt: override.trim() ? override : shipped,
  };
}

/** Shown before /api/settings answers (and if it never does): honest about the
    existence of the lever without inventing its content. */
export const IMPROVE_LINE_UNKNOWN =
  'Improve sends a fixed instruction to Klein. It is editable, and can be turned off.';

/** The toggle is off: the pass adds resolution and nothing else. */
export const IMPROVE_LINE_OFF =
  'Improve currently sends NO instruction — it only upscales.';

/** Anime/drawn datasets, when an instruction IS being sent. Amber, because on
    those datasets the default is actively working against the user.

    It CITES its source instead of asserting a fact about the images. The old
    wording ("This dataset is drawn.") read as a verdict the app had reached by
    looking at the pictures — it had not: the only thing it knows is the subject
    type someone picked. On a photoreal dataset left marked Anime, that verdict
    is simply false, and the reader has no way to guess which setting produced
    it. Naming the setting makes the note both checkable and actionable: if the
    sentence is wrong, the fix is one field away. */
const IMPROVE_ANIME_CAUTION =
  'This dataset’s subject type is set to anime. Words like “detailed texture” and '
  + '“sharp details” describe a photograph — they are what pushes anime skin and fabric '
  + 'towards realism. Edit the instruction, or turn it off and upscale only.';

/** Longest quote kept inline; the full text always rides in the title attribute. */
const QUOTE_MAX = 120;

/** Collapse whitespace so a multi-line prompt stays one readable line. */
function flatten(text) {
  return String(text || '').replace(/\s+/g, ' ').trim();
}

/** The quoted instruction, shortened for display. Never cuts mid-word when it
    can avoid it — a truncation that ends on "add sh" reads like a bug. */
export function shortenPrompt(text, max = QUOTE_MAX) {
  const flat = flatten(text);
  if (flat.length <= max) return flat;
  const cut = flat.slice(0, max);
  const lastSpace = cut.lastIndexOf(' ');
  return `${(lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut).replace(/[\s,;.]+$/, '')}…`;
}

/**
 * The one line to render next to the improve action.
 * @returns {{ text: string, quote: string|null, full: string|null, tone: 'muted'|'quote' }}
 *   `quote` is the (shortened) instruction to render between quotes, `full` the
 *   untruncated text for a title attribute. Both null when there is nothing to
 *   quote (still loading, or the toggle is off).
 */
export function improveInstructionLine({ loaded = false, enabled = true, prompt = '' } = {}) {
  const full = flatten(prompt);
  if (!loaded) return { text: IMPROVE_LINE_UNKNOWN, quote: null, full: null, tone: 'muted' };
  // An empty prompt with the toggle ON is the same OUTCOME as the toggle off —
  // no instruction is sent — so it must not read as "asks Klein for nothing".
  if (!enabled || !full) return { text: IMPROVE_LINE_OFF, quote: null, full: null, tone: 'muted' };
  return { text: 'Improve asks Klein to:', quote: shortenPrompt(full), full, tone: 'quote' };
}

/** The amber caution, or null. Only for drawn datasets, and only when an
    instruction is actually being sent — telling an anime user to beware of a
    prompt they already turned off would be noise. */
export function improveAnimeCaution({ loaded = false, enabled = true, prompt = '',
                                      subjectType = '' } = {}) {
  if (!loaded || !enabled || !flatten(prompt)) return null;
  return String(subjectType).toLowerCase() === 'anime' ? IMPROVE_ANIME_CAUTION : null;
}

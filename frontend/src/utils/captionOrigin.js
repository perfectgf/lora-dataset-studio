/* WHO wrote THIS caption — the per-image reading of `caption_origin`.

   PURE JS (no JSX, no imports) so `node --test` drives it directly.

   WHY THIS FILE EXISTS
   --------------------
   captionEngines.js reports the PASS that just ran ("12 by JoyCaption, 3 by the
   Ollama vision model") and its own header names the gap it left open, in as many
   words: "It reports the pass that just ran, not the provenance of a caption you
   are looking at weeks later." The column it asked for exists now — `caption_origin`
   on both image tables, written by every caption writer through
   services/caption_origin.stamp — and it was being used for exactly one decision
   (which rows a forced re-caption spares) while the screen where you READ the
   sentence still showed an anonymous string.

   That gap is not cosmetic. The default backend is 'auto', which is a CHAIN:
   JoyCaption writes what it can, the Ollama vision model covers the rest. The two
   halves land in the same column, in the same font, and until this module nothing
   on screen told them apart — so a caption that describes something the image does
   not contain looked exactly like one that reads it correctly.

   THE FOUR STATES, and the fourth is why this is a module and not a ternary:

     'asserted'   — a human typed, corrected or imported the text. Never folded
                    into the two engines: it is the one a forced pass SKIPS, and
                    calling it "machine-written" would be wrong about the work the
                    user did.
     'joycaption' — written by JoyCaption (local, via the ai-toolkit venv).
     'ollama'     — written by the local LLM vision model. The stored VALUE keeps
                    its spelling: it sits in caption_origin on BOTH surfaces and a
                    rename would need an alias table. It means "the configured
                    local provider", which may be LM Studio.
     NULL / ''    — the author was NEVER RECORDED. This is what every row that
                    predates the column carries, and it is NOT "machine": reading
                    the absence as an engine would put an attribution on screen
                    that nothing measured. It gets its own wording, and callers
                    that would rather stay silent on it can (`known` is false).

   An origin this build does not know is shown UNDER ITS OWN NAME rather than
   dropped — the same rule captionEngines.js follows for an unknown writer, and for
   the same reason: swallowing it rebuilds the blind spot the module exists to close.

   THE KEYS ARE STORED IN USER DATABASES (services/caption_origin.py says so in its
   header). Read them; never rename them here. */

/* Canonical order and the three wordings each state needs: `chip` for a badge
   beside the text, `short` for a sentence fragment, `title` for the tooltip that
   says what the state MEANS rather than repeating the label. */
export const CAPTION_ORIGINS = [
  {
    key: 'asserted',
    chip: '✍ you',
    short: 'You wrote this',
    title: 'You wrote or corrected this caption. A forced 🔄 Re-caption skips it '
      + 'unless you tick the box that says otherwise.',
  },
  {
    key: 'joycaption',
    chip: 'JoyCaption',
    short: 'Written by JoyCaption',
    title: 'Written by JoyCaption, which runs locally through the ai-toolkit folder.',
  },
  {
    key: 'ollama',
    chip: 'Local LLM',
    short: 'Written by the local LLM vision model',
    title: 'Written by the local LLM vision model (Ollama or LM Studio, whichever is '
      + 'configured). With the Auto backend this is the '
      + 'second half of the run — the images JoyCaption did not caption.',
  },
];

/* The absence, spelled out. Its `known` flag is false so a caller can choose
   silence (a grid of legacy rows would otherwise carry the same chip on every
   tile, which is noise), while a caller that has room says the true thing. */
export const CAPTION_ORIGIN_UNRECORDED = {
  key: '',
  chip: 'author not recorded',
  short: 'Author not recorded',
  title: 'This caption was written before the app recorded who writes captions. '
    + 'That is not the same as "written by a model" — nobody knows, and the app '
    + 'does not guess.',
  known: false,
};

/** The entry for a stored value. Never null: an empty/unknown origin resolves to
 *  CAPTION_ORIGIN_UNRECORDED, and an origin this build does not know resolves to
 *  an entry named after itself. Callers branch on `.known`, not on null. */
export function captionOriginInfo(origin) {
  const key = typeof origin === 'string' ? origin.trim().toLowerCase() : '';
  if (!key) return CAPTION_ORIGIN_UNRECORDED;
  const hit = CAPTION_ORIGINS.find((o) => o.key === key);
  if (hit) return { ...hit, known: true };
  return {
    key,
    chip: key,
    short: `Written by ${key}`,
    title: `Written by ${key}, an origin this version of the app does not know. `
      + 'Shown under its own name rather than dropped.',
    known: true,
  };
}

/** Is this caption the user's own word? The one question that changes what a
 *  destructive button is allowed to do, asked in the UI's vocabulary. */
export function captionIsAsserted(origin) {
  return captionOriginInfo(origin).key === 'asserted';
}

/** One line for a tooltip that already describes an image, or '' when there is no
 *  caption to attribute. An empty caption has no author, whatever the column says. */
export function captionOriginTooltipLine(caption, origin) {
  if (!(caption || '').trim()) return '';
  return captionOriginInfo(origin).short;
}

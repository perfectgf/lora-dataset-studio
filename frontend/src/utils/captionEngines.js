/* WHO wrote the captions of the pass that just ran.

   PURE JS (no JSX, no imports) so `node --test` drives it directly.

   WHY THIS FILE EXISTS
   --------------------
   The captioning backend defaults to 'auto', described in Settings as "prefers
   JoyCaption (via ai-toolkit) and falls back to the Ollama vision model". The
   fallback is real and it is SILENT: JoyCaption captions the images it can in one
   batch, Ollama covers the rest, and on a Concept dataset Ollama also rewrites
   JoyCaption's drafts. Those engines do not write alike — one is blunt and literal,
   the other is a rewrite in its own register — so a user whose captions suddenly
   read differently had no way at all to learn why. The setting says what the app
   INTENDS; nothing said what the run actually DID.

   The backend now counts each stored caption against the engine that wrote it
   (face_dataset_service.CAPTION_WRITER_*), the caption route returns those counts,
   and this turns them into the one line shown where the result appears.

   The keys are a contract with the backend — read them, never rename them here.

   WHAT THIS DELIBERATELY DOES NOT DO
   ----------------------------------
   It reports the pass that just ran, not the provenance of a caption you are looking
   at weeks later: nothing is stored per image. Persisting the writer would mean a new
   column on face_dataset_image, which every existing database would have to gain by
   migration — a heavier change than the defect warrants, and one nobody asked for.
   Follow-up, if it is ever wanted: a `caption_writer` column filled by the same
   counters, which would let a tile say who wrote ITS caption. */

/* Canonical order, and the two wordings each writer needs: `solo` when it wrote the
   whole pass (a sentence), `short` when several shared it (a label in a count).
   `joycaption_refined` is its own writer, not a blend to be split in two: on the
   Concept path the stored text IS Ollama's rewrite of a JoyCaption draft, and
   calling it either one alone would be false. */
export const CAPTION_WRITERS = [
  { key: 'joycaption', short: 'JoyCaption', solo: 'Written by JoyCaption.' },
  { key: 'joycaption_refined', short: 'JoyCaption + Ollama',
    solo: 'Drafted by JoyCaption, rewritten by the Ollama vision model.' },
  { key: 'ollama', short: 'Ollama', solo: 'Written by the Ollama vision model.' },
];

// Why this line is worth a glance, for the control's title/tooltip. States the
// mechanism, then the lever — never "contact support".
export const CAPTION_ENGINE_WHY =
  'The Auto backend uses JoyCaption first and falls back to the Ollama vision model, '
  + 'and the two write in different styles. Pick one engine in ⚙️ Options to keep a '
  + 'single voice across a dataset.';

/** [{key, short, solo, n}] for the engines that actually wrote something, in
 *  canonical order — then any key this build doesn't know, which is listed under its
 *  own name rather than dropped. Silently swallowing an unknown writer would rebuild
 *  the very blind spot this module exists to close. */
export function captionEngineBreakdown(engines) {
  const e = (engines && typeof engines === 'object') ? engines : {};
  const known = CAPTION_WRITERS
    .filter((w) => Number(e[w.key]) > 0)
    .map((w) => ({ ...w, n: Number(e[w.key]) }));
  const extra = Object.keys(e)
    .filter((k) => Number(e[k]) > 0 && !CAPTION_WRITERS.some((w) => w.key === k))
    .map((k) => ({ key: k, short: k, solo: `Written by ${k}.`, n: Number(e[k]) }));
  return [...known, ...extra];
}

/** The ONE line to show after a pass. '' when the run wrote nothing (or an older
 *  backend sent no counts at all) — an empty string is how a caller knows to show
 *  nothing, rather than a confident "Written by nobody". */
export function captionEnginesSummary(engines) {
  const parts = captionEngineBreakdown(engines);
  if (!parts.length) return '';
  if (parts.length === 1) return parts[0].solo;
  return parts.map((p) => `${p.n} by ${p.short}`).join(' · ');
}

/** The same information appended to a result sentence, e.g. the toast that already
 *  reports "12 captioned". Empty when there is nothing to add, so the caller can
 *  concatenate unconditionally. */
export function captionResultSuffix(engines) {
  const s = captionEnginesSummary(engines);
  return s ? ` · ${s}` : '';
}

/* The other half of "what the run actually did": the images the engine REFUSED.

   A refusal never stops the batch, so a pass could hand back fewer captions than
   images and say nothing about the difference — "37 captioned" on an 89-image
   dataset, with the reason living only in the server log. The count comes back on
   the caption response as {skipped, skipped_reason}; the reason is the engine's own
   sentence, passed through verbatim rather than re-worded here.

   '' when nothing was skipped (or an older backend sent no count), so a caller can
   concatenate it unconditionally. */
export function captionSkippedSuffix(result) {
  const n = Number(result && result.skipped);
  if (!Number.isFinite(n) || n <= 0) return '';
  const reason = (result.skipped_reason || '').trim();
  return reason ? ` — ${n} skipped: ${reason}` : ` — ${n} skipped`;
}

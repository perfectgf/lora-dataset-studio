/* Picking real captions out of a dataset, for the Preview-prompts textarea.
 *
 * The preview prompts are the only images a user can judge a run by while it is
 * still costing money, and the shipped defaults describe NOBODY — a generic
 * "a portrait of {trigger}" tells you nothing about whether this dataset's
 * subject is being learned. The captions the dataset already carries are the
 * one text in the app written about these exact images.
 *
 * Two rules that are not obvious from the call site:
 *   · Only KEPT images. Rejected and pending ones are not what the run trains
 *     on, so a prompt drawn from them would preview something the LoRA never
 *     saw.
 *   · Always the LONG caption. With dual captions on, an image carries a short
 *     one too, but the run itself trains on the long text — previewing the
 *     short one would compare the model against a prompt shape it never met.
 */

/** The distinct, non-empty long captions of the images this run trains on. */
export function keptCaptions(images) {
  const seen = new Set();
  const out = [];
  for (const img of images || []) {
    if (!img || img.status !== 'keep') continue;
    const caption = String(img.caption || '').trim();
    if (!caption || seen.has(caption)) continue;
    seen.add(caption);
    out.push(caption);
  }
  return out;
}

/** Up to `limit` distinct captions, drawn at random. Re-drawing is the point:
 *  the button replaces the textarea, so clicking again is a new sample rather
 *  than an ever-growing pile. `random` is injectable so a test can pin a draw. */
export function pickDatasetCaptions(images, limit = 5, random = Math.random) {
  const pool = keptCaptions(images);
  const take = Math.max(0, Math.min(Math.floor(limit) || 0, pool.length));
  // Partial Fisher-Yates on a copy: distinct by construction, no reject loop.
  for (let i = 0; i < take; i += 1) {
    const j = i + Math.floor(random() * (pool.length - i));
    [pool[i], pool[j]] = [pool[j], pool[i]];
  }
  return pool.slice(0, take);
}

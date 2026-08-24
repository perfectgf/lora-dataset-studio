/**
 * 🔄 Re-run of the ✨ Upscale & improve pass, for a tile that IS an improvement.
 *
 * The generic regenerate button is deliberately hidden on these tiles: that route
 * restarts from the dataset's reference photo and the catalog prompt, so it would
 * quietly produce an unrelated variation instead of a better version of THIS
 * image. The right gesture is to run the improve pass again, from the same parent
 * image, with the settings as they are today (klein.improve_steps, megapixels,
 * base/consistency strength and the improvement instruction are all editable —
 * tuning them is the only reason to re-run).
 *
 * Kept in plain .js (no JSX) so `node --test` can exercise the decision itself.
 */
const KLEIN_IMAGE_IMPROVE = 'klein_image_improve';

export const REIMPROVE_TITLE =
  'Re-run Upscale & improve on the source image, with your current improve settings';
// Wording mirrors the backend refusals (REIMPROVE_* in face_dataset_service.py):
// the tile says WHY before the click instead of surfacing a 400 after it.
export const REIMPROVE_NO_PARENT_TITLE =
  'Cannot re-improve: the source image this came from was deleted';

export const isImageImproveRow = (img) => img?.derivation_kind === KLEIN_IMAGE_IMPROVE;

/**
 * What the tile should offer for an improvement result.
 * Returns null when there is nothing to show (not an improvement, or the pass is
 * still running), otherwise `{ enabled, title }` for a real <button>.
 */
export function improveRerunAffordance(img) {
  if (!isImageImproveRow(img)) return null;
  // Still generating: the row has no file yet and the backend answers 409.
  if (img.status === 'pending' && !img.filename) return null;
  // Dangling parent (the source was deleted — there is no ForeignKey). A DISABLED
  // button that explains itself beats a dead button that fails on click.
  if (img.parent_image_id == null) {
    return { enabled: false, title: REIMPROVE_NO_PARENT_TITLE };
  }
  return { enabled: true, title: REIMPROVE_TITLE };
}

/**
 * The original guard, unchanged and now testable: the generic regenerate route is
 * never offered for a derived row (improvement or small-image rescue).
 */
export function canRegenerateGeneric(img, { isRescueDerived = false } = {}) {
  if (isRescueDerived || isImageImproveRow(img)) return false;
  return img?.source === 'generated' && !(img.status === 'pending' && !img.filename);
}

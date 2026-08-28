/* ✨ Upscale & improve, offered on a picture of the ◉ Canvas board.

   WHY THIS IS NOT THE DATASET IMPROVE, even though it is the same button.
   The board's pictures are `lora_test_image` rows (that is what
   `canvas_image_node.image_id` stores); the dataset lightbox's pictures are
   `face_dataset_image` rows. Two tables, two INDEPENDENT id spaces. Sending a
   board id to `/api/dataset/image/<id>/improve` does not fail — it finds a real,
   unrelated dataset image and improves THAT one. So the surfaces share the
   engine choice, the wording and the Klein note (all of them reused from
   utils/improveEngines.js and components/dataset/KleinImproveNote.jsx), and
   nothing else. The route is its own.

   Pure module, no JSX — `node --test` cannot parse JSX and this is the part with
   the cases worth pinning. */

/** The stored derivation kind of a canvas improvement, mirroring the backend
    (`lora_test_studio.CANVAS_IMAGE_IMPROVE`). ⚠️ It is written into user
    databases: renaming it would strand every row already there, so it never
    changes without an alias path. */
export const CANVAS_IMAGE_IMPROVE = 'canvas_image_improve'

/** True when this row IS an improvement produced by the pass. */
export const isCanvasImproveRow = (img) => img?.derivation_kind === CANVAS_IMAGE_IMPROVE

/**
 * Why ✨ cannot be offered for this picture, or null when it can.
 *
 * Improve results CHAIN (the server lifted its derivation guard): an improve
 * row is as valid a source as the render it came from — Klein detail then
 * SeedVR2 resolution on the same picture is the point. So being one is no
 * refusal. The only picture that cannot be improved is one the board holds
 * without a library row — there is no id to send. Returning a REASON rather
 * than a boolean is what lets the caller choose between hiding the action and
 * showing it disabled with the reason attached.
 */
export function canvasImproveRefusal(img) {
  if (!img || !Number.isInteger(Number(img.id))) {
    // A picture the board holds only as a URL — the lane's reference face, a
    // pill's preview. There is no row to improve, and no id to send.
    return 'This picture has no library entry to improve.'
  }
  return null
}

/** Whether the ✨ group should appear at all for this picture. */
export const canImproveCanvasImage = (img) => canvasImproveRefusal(img) === null

/** What the board says once the pass is queued.

    It names WHERE the result will appear, because that is the one thing this
    surface cannot show by itself: the improvement arrives as its own row in the
    checkpoint's gallery, and until the user opens that gallery (or pins it)
    nothing on the board changes. A toast that only said "started" would read as
    a no-op on a screen where nothing moves. */
export function canvasImproveLaunchMessage(engineLabel) {
  return `${engineLabel || 'Improve'} started — the result arrives in this `
    + 'checkpoint’s gallery, next to the original, and can be pinned from there. '
    + 'The picture on the board is left untouched.'
}

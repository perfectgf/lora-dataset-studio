/* 🗑 Deleting a picture FROM the board.
 *
 * ── Why this is not the ✕ ───────────────────────────────────────────────────
 * The board already has a ✕ and it is deliberately NOT a delete: it closes the
 * node and keeps its geometry, so re-pinning the same image from its gallery
 * puts it back where and how big you left it. That promise is the feature.
 *
 * What was missing is the other half. The board is where you actually LOOK at
 * renders side by side, so it is where you decide that one of them is a
 * failure — and until now the only way to act on that decision was: close the
 * node, open the run, find the checkpoint, open its gallery, enter Select mode,
 * find the same picture again among fifty thumbnails, tick it, confirm. Six
 * steps to delete the thing you were pointing at.
 *
 * ── It deletes for real, through the gallery's own route ────────────────────
 * Not a second delete: `galleryEndpoints` is the gallery's own pairing of
 * (read, delete) per scope, so the board and the gallery cannot disagree about
 * what "delete" means — including whether the install moves files to a
 * recoverable place or removes them outright, which is a SETTING and is
 * reported back by the same payload the gallery reads.
 *
 * ── Which scope ────────────────────────────────────────────────────────────
 * A pinned picture knows the checkpoint it came from (`record_id` + `step`), so
 * the delete is scoped to that checkpoint. A legacy row with no step is scoped
 * to the run instead — the same fallback `galleryScope` already makes. In both
 * cases the body names ONE image id, so a wider scope can never mean a wider
 * delete; the scope only decides which route validates the id.
 */
import { galleryEndpoints } from './runGallery.js';

/**
 * What deleting THIS pinned node would call: `{ endpoint, imageId, scope }`,
 * or `null` when the node cannot be traced back to a run (which is not a
 * failure to report — it is a node the board should not offer to delete).
 */
export function canvasImageDeleteTarget(node) {
  const img = node?.image;
  const imageId = Number(node?.imageId ?? img?.id);
  const recordId = img?.record_id;
  if (!Number.isFinite(imageId) || recordId == null) return null;
  const step = img?.step ?? null;
  const endpoints = galleryEndpoints({ recordId, step });
  if (!endpoints?.remove) return null;
  return { endpoint: endpoints.remove, imageId, scope: step == null ? 'run' : 'checkpoint' };
}

/**
 * The two-step guard, as data rather than as three booleans in a component.
 *
 * A delete one tap away from 🔍 and ✕ on a 28-px cluster is a delete that WILL
 * happen by accident — that is exactly why the gallery hides its own behind a
 * mode and a confirmation. The board cannot afford a mode, so the button arms
 * itself instead: first press arms, second press deletes, and it disarms on its
 * own after ARM_MS so a board left open does not keep a live delete under the
 * cursor.
 */
export const DELETE_ARM_MS = 4000;

/** What the button must SAY and LOOK like in each of its three states. */
export function canvasDeleteButtonState({ armed = false, busy = false, label = 'this image' } = {}) {
  if (busy) return { glyph: '…', disabled: true, tone: 'busy', title: 'Deleting…', aria: 'Deleting this image' };
  if (armed) {
    return { glyph: '🗑!', disabled: false, tone: 'armed',
      // The word "permanently" is deliberately absent: whether the file is
      // recoverable is an install SETTING, and promising either way here would
      // be a lie on half the installs. The result sentence, which comes from
      // the server, is where that is stated.
      title: `Press again to delete ${label} — this removes the image itself, not just the node`,
      aria: `Confirm deleting ${label}` };
  }
  return { glyph: '🗑', disabled: false, tone: 'idle',
    title: `Delete ${label} — removes the image itself. Press once to arm, again to confirm. `
      + '✕ only takes it off the board.',
    aria: `Delete ${label}` };
}

/** The sentence a failed delete leaves on the node. Short: it is drawn inside a
 *  thumbnail, and the same strip the download refusal uses. */
export function canvasDeleteError(err) {
  return err?.message || 'Could not delete this image';
}

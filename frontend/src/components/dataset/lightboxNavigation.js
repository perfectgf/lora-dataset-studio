/**
 * Moving from one inspected image to the next, and the state that must NOT come
 * with it.
 *
 * ── Which list ──────────────────────────────────────────────────────────────
 * The lightbox walks the list the grid SHOWS — filtered, sorted, in the grid's
 * own order — not the raw dataset payload. Anything else would send ⟩ to an
 * image the user cannot see behind the overlay, and "where am I" becomes
 * unanswerable. Paging is a rendering window over that same list (the pager
 * says "1–500 of 6211", and "select all" already means every page), so
 * navigation crosses page boundaries and the grid follows underneath — closing
 * the lightbox never lands you on a page that does not contain the image you
 * were just looking at.
 *
 * Small-image rescue rows are traversed like anything else, and deliberately:
 * the grid only ever renders the RESOLVED winner of a rescue pair (the unresolved
 * pairs live in Curation and never reach this list), so a rescue winner is an
 * ordinary tile on screen. Skipping it would make ⟩ jump over a picture the user
 * is looking at. What the grid withholds from them is the tick box — a bulk
 * action on a paired provenance — which is a different question from "show me
 * the next picture".
 *
 * ── The state that must not travel ──────────────────────────────────────────
 * Zoom, the comparison pane and "an improvement is running" are properties of
 * ONE image. Carried onto the next one they range from merely wrong (a 100 %
 * zoom on a picture you have not looked at yet) to dangerous (a pane captioned
 * "original" showing the parent of the PREVIOUS image).
 *
 * The guarantee here is structural rather than a reset effect: the state is
 * stamped with the id it was computed for, and a render that finds a foreign
 * stamp ignores it and uses a fresh one. There is therefore no frame in which
 * stale state is painted, and no ordering to get right.
 *
 * READING a foreign stamp is not enough on its own, because there is ONE slot.
 * A late writer — the `finally` of an improve that started on the previous
 * image — would stamp that slot with its own id, and the read would then find a
 * foreign stamp for the image actually on screen and hand it a FRESH state:
 * your zoom dropped and your comparison pane closed, on an image that improve
 * never touched. Measured, not feared. So writing checks the stamp too, and a
 * writer that no longer owns the screen is dropped: see `stampedPatch`.
 */

/**
 * Everything an image starts from, whatever the previous one ended on.
 *
 * `compareMode` is a MODE, not a flag: the lightbox offers two comparisons —
 * against the original an improve pass came from, and against the dataset's
 * reference photo — and they are mutually exclusive, so 'none' | 'derived' |
 * 'reference' is the honest shape. It replaced a boolean `comparing`; keeping
 * the state in one slot is what makes "moving image closes whichever pane was
 * open" true for both without a second reset path.
 *
 * `repairOpen` — the ✦ Repair dialog — is here for exactly that reason too: it
 * is a full-screen overlay showing ONE image, so ⟩ pressed behind it would
 * leave you repairing a picture that is no longer the one on screen.
 *
 * `actionsOpen` — the narrow-screen actions panel — lives here for the same
 * reason, not because a panel belongs to a picture: it is a full-screen overlay
 * on a phone, and ⟩ pressed behind it would otherwise land you on an image you
 * cannot see, under a panel you did not reopen. Sharing the stamped slot makes
 * "moving image gives you the picture back" structural, like the two panes.
 *
 * `deciding` — a ✓ Keep / ✕ Reject verdict is in flight for THIS image — lives
 * here for the plainest reason of the lot: the verdict advances to the next
 * picture the moment it lands, so the flag must die with the image it belonged
 * to. Held in a useState of its own, a slow POST would have left the NEXT image
 * with its buttons greyed out.
 */
export function freshLightboxImageState(imageId) {
  return {
    imageId: imageId ?? null,
    full: false,
    compareMode: 'none',
    improving: false,
    actionsOpen: false,
    repairOpen: false,
    // 📷 In the id-stamped slot like every other overlay flag: an open picker
    // is a moment on THIS image, and ⟨ / ⟩ must leave it behind with the image
    // it belonged to rather than carry it onto the next one.
    cameraOpen: false,
    improveOpen: false,
    deciding: false,
  };
}

/**
 * The state to RENDER for `imageId`: the stored one when it belongs to this
 * image, a fresh one otherwise. Pure — call it during render.
 */
export function lightboxImageState(stored, imageId) {
  const id = imageId ?? null;
  if (!stored || stored.imageId !== id) return freshLightboxImageState(id);
  return stored;
}

/**
 * The next stored state when the holder of `stampId` asks for `patch`, given
 * that `currentId` is the image on screen right now.
 *
 * A writer that no longer owns the screen is DROPPED rather than allowed to
 * stamp the single slot — see the header. Pure, so the sequence that exposed it
 * (improve on A → ⟩ → zoom B → A resolves) is testable without a DOM.
 */
export function stampedPatch(stored, patch, stampId, currentId) {
  const stamp = stampId ?? null;
  if (stamp !== (currentId ?? null)) return stored;
  return { ...lightboxImageState(stored, stamp), ...patch, imageId: stamp };
}

const FIRST = (total) => `You are on the first of the ${total} images shown here.`;
const LAST = (total) => `You are on the last of the ${total} images shown here.`;
const ONLY = 'The current filters show only this image.';

/**
 * Where the inspected image sits in the shown list, and what ⟨ / ⟩ do from
 * there.
 *
 * `available: false` — the image is not in the list at all — is a real case, not
 * a guard: the rescue-review preview opens the lightbox on a candidate that
 * lives in Curation, and a poll can retire the image under an open lightbox. No
 * list, no arrows; nothing is invented.
 *
 * The ends do NOT wrap. Wrapping makes "have I seen everything?" unanswerable on
 * a wall of near-identical shots, which is exactly the question curation is: the
 * button goes dead and SAYS which end you reached, rather than silently doing
 * nothing or quietly restarting the loop.
 */
export function lightboxNeighbours(images, currentId) {
  const list = (Array.isArray(images) ? images : []).filter((i) => i && i.id != null);
  const total = list.length;
  const index = currentId == null ? -1 : list.findIndex((i) => i.id === currentId);
  if (index < 0) {
    return {
      available: false, index: -1, total, position: '',
      prev: null, next: null, prevReason: null, nextReason: null,
    };
  }
  const atFirst = index === 0;
  const atLast = index === total - 1;
  const endReason = total <= 1 ? ONLY : null;
  return {
    available: true,
    index,
    total,
    position: `${index + 1} / ${total}`,
    prev: atFirst ? null : list[index - 1],
    next: atLast ? null : list[index + 1],
    prevReason: atFirst ? (endReason || FIRST(total)) : null,
    nextReason: atLast ? (endReason || LAST(total)) : null,
  };
}

/* The "does this field own the arrow keys?" guard used to live here, in a
   private copy that only knew about ← and →. It moved to
   components/shared/reviewShortcuts.js together with the rest of the review
   grammar, because the lightbox now answers K/R/S as well and the Bank answers
   the same letters: one guard, one place, and the checkbox exemption the Bank
   paid for in a real bug applies here too. */

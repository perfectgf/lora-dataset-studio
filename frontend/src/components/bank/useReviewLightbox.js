// react-frontend/src/components/bank/useReviewLightbox.js
// The bank's ▶ Review fast-triage lightbox cluster — the id snapshot it
// walks, and the open / decided / rotated / edited / close handlers —
// moved VERBATIM from BankWorkspace.jsx (2026-08-24, hook series wave 5).
// fetchAllIds and filterParams stay panel-owned and ride as params: the
// grid, "Select all in filter" and ▶ Review must keep walking the SAME
// filter translation.
import { useState } from 'react';

export function useReviewLightbox({
  bankId, filter, filterParams, fetchAllIds, showSelected, selected,
  selectedOrder, setPage, toast, refreshPayload, refreshImages,
}) {
  // ▶ Review — the fast-triage lightbox. `review` holds the SNAPSHOT of ids it
  // walks ({ids, startId}); null when closed. Snapshotting at open is the whole
  // point: a decision drops the image out of the current filter, so a live list
  // would reorder under the cursor and make the run skip or loop.
  const [review, setReview] = useState(null)
  const [reviewLoading, setReviewLoading] = useState(false)

  // Open ▶ Review over what the user is actually looking at: the whole current
  // filter (all pages, current sort), or the selection when the "Show selected"
  // view is on. `startId` (the ▶ on a tile) opens on that image.
  const openReview = async (startId = null) => {
    setReviewLoading(true)
    try {
      const ids = showSelected
        ? ((selectedOrder && selectedOrder.length) ? selectedOrder : [...selected])
        : await fetchAllIds(bankId, filterParams(filter))
      if (!ids.length) {
        toast.info('Nothing to review — no image matches the current filter.')
        return
      }
      setReview({ ids, startId })
    } catch (e) {
      toast.error(e?.message || 'Could not build the review list.')
    } finally {
      setReviewLoading(false)
    }
  }

  // One decision landed in the lightbox — refresh the header counters so
  // kept/rejected/undecided track the run live. The grid is refreshed once, on
  // close, so its tiles don't shuffle around behind the lightbox.
  const onReviewDecided = () => { refreshPayload() }
  // A turn made in ▶ Review must already be right on the tile behind it: the
  // grid is only refetched on close, and a tile still lying sideways would read
  // as "it didn't take".
  const onReviewRotated = (imageId, rotation) => setPage((prev) => ({
    ...prev,
    images: prev.images.map((im) => (im.id === imageId
      ? { ...im, rotation, width: im.height, height: im.width }
      : im)),
  }))
  /* Same rule for a ✂ crop / ↩ revert made in ▶ Review, and it matters MORE here:
     the tile's thumbnail URL carries the edit generation, so a tile left with the
     old generation would keep serving the pre-crop image from the browser cache
     for an hour behind the lightbox. The state comes from the route's own reply
     rather than being guessed. */
  const onReviewEdited = (imageId, state) => setPage((prev) => ({
    ...prev,
    images: prev.images.map((im) => (im.id === imageId
      ? { ...im,
          edit_method: state?.edit_method ?? null,
          edit_generation: state?.edit_generation ?? 0,
          rotation: state?.rotation ?? 0,
          width: state?.width ?? im.width,
          height: state?.height ?? im.height }
      : im)),
  }))
  const closeReview = () => { setReview(null); refreshPayload(); refreshImages() }
  return {
    review, reviewLoading, openReview, onReviewDecided, onReviewRotated,
    onReviewEdited, closeReview,
  };
}

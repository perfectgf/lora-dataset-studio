/** When a failed load of a video dataset may take the workspace away.
 *
 * The rule the page owes, written as a VALUE so it can be tested without a DOM:
 * a transient failure of the refresh that runs after every write (a caption
 * save, a removal, a bulk rewrite) must never replace the screen — that costs
 * every unsaved caption draft, the selection, the filter, the sort, the open
 * section, and it unmounts the training block while its run carries on
 * server-side. The image lane settled this and wrote it down (useDataset.js:
 * only a definitive 404 ejects; transient errors keep the workspace).
 *
 * Two things eject, and only two: the dataset is GONE (a 404 — apiFetch puts
 * the status on the error), or there is nothing on screen yet to keep (the very
 * first load). Everything else keeps the last good payload.
 */
export function shouldEjectOnLoadError(err, hadPayload) {
  if (!hadPayload) return true
  // The status, never the message: a proxy's HTML error page or a network
  // layer's wording can contain "not found" about something that is not this
  // dataset, and matching text there would eject a user with drafts in hand.
  return Number(err?.status) === 404
}

/** What the page shows while it keeps a stale payload: nothing loud, but
 * SOMETHING. apiFetch stays silent in background mode by design, and the app's
 * offline banner only covers a network failure — a server that answered 500 is
 * "reachable" to it. Without this line, a caption saved and then not refreshed
 * would silently show its previous text. */
export function staleNote(err) {
  const status = Number(err?.status)
  if (status) return `Showing the last loaded state — the server answered ${status} on the last refresh.`
  return 'Showing the last loaded state — the last refresh did not reach the server.'
}

/** 🗃️ Bank folder sync — pure helpers (no JSX, so `node --test` can run them).
 *
 * A bank points at a LIVE folder: the backend re-walks it on the bank list and
 * on the workspace payload, and reports the outcome as `folder_sync`
 * ({added, missing, unavailable, error}). The mutation must never be silent —
 * the user has to know why their counters moved, and that a file listed in the
 * bank is no longer on disk. */

/** The one-shot notification for a walk that just happened: new files, or a
 * refusal. Returns {type, text} for the toast, or null when there is nothing
 * worth interrupting the user for (nothing added = nothing to say). */
export function folderSyncToast(sync) {
  if (!sync) return null
  if (sync.error) return { type: 'error', text: `Folder refresh — ${sync.error}` }
  const added = Number(sync.added) || 0
  const notAdded = Number(sync.not_added) || 0
  // The per-bank ceiling used to REFUSE the whole batch. It now takes what fits
  // and says what it left, so the user learns the number and the remedy instead
  // of a flat no. Zero images added is still worth a toast in this case — that
  // is precisely the case where silence would look like a bug.
  if (notAdded > 0) {
    const limit = Number(sync.limit) || 0
    return {
      type: 'warning',
      text: `${added} new image(s) added; ${notAdded} were not — this bank is at its `
        + `ceiling of ${limit.toLocaleString('en-US')} images. Move the rest into a `
        + 'second bank, or delete images from the folder to make room.',
    }
  }
  if (added > 0) {
    return {
      type: 'success',
      text: `${added} new image(s) found in the folder — added to the bank.`,
    }
  }
  return null
}

/** Same, for the bank LIST (one walk per bank): a single line instead of one
 * toast per bank. Null when nothing was added anywhere. */
export function bankListSyncToast(banks) {
  const rows = Array.isArray(banks) ? banks : []
  let added = 0
  let n = 0
  let notAdded = 0
  for (const b of rows) {
    const a = Number(b?.folder_sync?.added) || 0
    notAdded += Number(b?.folder_sync?.not_added) || 0
    if (a > 0) { added += a; n += 1 }
  }
  // A bank at its ceiling must say so from the LIST too — that is where a user
  // who just dumped a scrape into the folder looks first.
  if (notAdded > 0) {
    return {
      type: 'warning',
      text: `${added} new image(s) added; ${notAdded} were not — a bank is at its `
        + 'image ceiling. Open it to see which, and split the folder or make room.',
    }
  }
  if (added <= 0) return null
  return {
    type: 'success',
    text: n === 1
      ? `${added} new image(s) found in the folder — added to the bank.`
      : `${added} new image(s) found across ${n} banks — added to them.`,
  }
}

/** How long ago, in words. Deliberately coarse: the point is "recent enough" vs
 * "old", not a stopwatch. */
function agoText(seconds) {
  const s = Math.max(0, Number(seconds) || 0)
  if (s < 60) return 'just now'
  const m = Math.round(s / 60)
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  return h === 1 ? 'an hour ago' : `${h} hours ago`
}

/** The bank LIST's honesty line about how fresh its counts are.
 *
 * The list used to re-walk every bank's source folder before rendering — a full
 * disk inventory of the whole library paid on every navigation (690-1 190 ms on
 * a real 86 493-image library). It no longer does: a folder is walked when its
 * bank is OPENED, or when the user clicks 🔄 Rescan folders.
 *
 * That trade is only acceptable if the page SAYS it — a silently stale list
 * would be worse than a slow one. Returns {stale, text}, or null when there is
 * no bank to be honest about. `folder_sync.walked` / `.age` come from the
 * server (per bank, since this app run). */
export function folderCheckNote(banks) {
  const syncs = (Array.isArray(banks) ? banks : [])
    .map((b) => b?.folder_sync).filter(Boolean)
  if (!syncs.length) return null
  const unchecked = syncs.filter((s) => !s.walked).length
  if (unchecked > 0) {
    return {
      stale: true,
      text: unchecked === syncs.length
        ? 'Counts below are what the app knew last time — a folder is re-checked '
          + 'when you open its bank. Rescan folders to update them all now.'
        : `Counts below are what the app knew last time for ${unchecked} of these `
          + 'banks — a folder is re-checked when you open its bank. Rescan folders '
          + 'to update them all now.',
    }
  }
  const oldest = Math.max(...syncs.map((s) => Number(s.age) || 0))
  return { stale: false, text: `Source folders checked ${agoText(oldest)}.` }
}

/** The PERSISTENT note (a line in the workspace header, not a toast): the state
 * of the source folder as of the last walk. Files that disappeared are reported
 * and kept — the bank never deletes rows behind the user's back, so a folder
 * that got moved or unplugged shows up as a warning instead of a vanished
 * triage. Returns {tone, text} or null when everything is in sync. */
export function folderSyncNote(sync) {
  if (!sync) return null
  if (sync.unavailable) {
    return {
      tone: 'error',
      text: 'Source folder unavailable (moved, renamed or on a disconnected drive) — '
        + 'the bank keeps every image and decision; reconnect the drive, or point '
        + 'the bank at the folder\'s new location.',
      canRelocate: true,
    }
  }
  const missing = Number(sync.missing) || 0
  if (missing > 0) {
    return {
      tone: 'warn',
      text: `${missing} image(s) listed here are no longer in the folder — their rows `
        + 'are kept (nothing is deleted for you) but they will fail to load. If you '
        + 'moved the folder, point the bank at its new location. If the files are '
        + 'really gone (a downloader cleaned up, you tidied the folder), Forget '
        + 'missing drops their rows.',
      canRelocate: true,
      // The other honest cause, with its own remedy. Only offered on MISSING —
      // an unavailable folder means the walk itself failed, and forgetting on
      // that verdict would erase the whole triage (the server refuses it too).
      canForget: true,
      missing,
    }
  }
  return null
}

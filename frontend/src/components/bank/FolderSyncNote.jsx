import { folderSyncNote } from './bankSync'

/** 🗃️ The state of a bank's SOURCE FOLDER as of the last automatic walk.
 *
 * Shown on the bank card and in the workspace header. It only appears when
 * something is off — files listed in the bank that are no longer on disk, or a
 * folder that has gone away entirely. Both cases are reported, never acted on:
 * the bank keeps every row and every decision, because a disconnected drive
 * must not be able to erase a triage.
 *
 * The note carries the FIX and not only the diagnosis — the user is already
 * reading this line, which is where the buttons belong. There are TWO honest
 * fixes because there are two honest causes: the folder MOVED (``onRelocate``
 * repoints the bank, losing nothing) or the files are REALLY GONE — a
 * downloader that cleans up its own intermediates, a sync client, a by-hand
 * tidy — and ``onForget`` lets the bank drop those rows. */
export default function FolderSyncNote({ sync, onRelocate, onForget }) {
  const note = folderSyncNote(sync)
  if (!note) return null
  const tone = note.tone === 'error'
    ? 'border-rose-400/40 bg-rose-500/10 text-rose-200'
    : 'border-amber-400/40 bg-amber-500/10 text-amber-200'
  return (
    <div className={`rounded-md border px-2 py-1 text-xs ${tone}`}>
      <p>{note.tone === 'error' ? '⚠️ ' : 'ℹ️ '}{note.text}</p>
      <div className="mt-1 flex flex-wrap gap-2">
        {note.canRelocate && onRelocate && (
          <button type="button" onClick={onRelocate}
            className="rounded border border-current px-2 py-0.5 text-xs font-semibold hover:bg-white/10">
            📦 Move folder…
          </button>
        )}
        {note.canForget && onForget && (
          <button type="button" onClick={onForget}
            title="Drop the rows whose file is no longer in the folder. Rows only — nothing on disk is touched. Asks first, with a fresh count."
            className="rounded border border-current px-2 py-0.5 text-xs font-semibold hover:bg-white/10">
            🧹 Forget missing ({note.missing.toLocaleString('en-US')})…
          </button>
        )}
      </div>
    </div>
  )
}

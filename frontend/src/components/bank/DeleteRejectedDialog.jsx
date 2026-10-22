import { useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import {
  datasetConflictBlock, deleteDestination, deletePreviewState,
  deleteRejectedStart, isRecoverable, sharedFileCount, sharedFilesWarning,
} from './bankOverlap'

/** 🗑 Delete rejected from disk — the ONE bank action that writes to the source
 * folder. Unlike the reversible reject STATUS, this removes the actual files.
 *
 * The dialog asks the server what the run would really do BEFORE arming: where
 * the files would go (system Recycle Bin, the app's Trash, or nowhere), and
 * whether another bank points at the same folder — nested banks share files, so
 * a cleanup here can amputate a bank the user isn't even looking at. Gated
 * behind a type-DELETE confirmation; Cancel is the default focus. */
export default function DeleteRejectedDialog({ bankId, count, sourcePath, onClose, onDone }) {
  const toast = useToast()
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const [preview, setPreview] = useState(null)
  const armed = confirm.trim().toUpperCase() === 'DELETE'

  useEffect(() => {
    let alive = true
    apiFetch(`/api/bank/${bankId}/delete-rejected/preview`)
      .then((d) => { if (alive) setPreview(d) })
      .catch(() => { if (alive) setPreview({ failed: true }) })
    return () => { alive = false }
  }, [bankId])

  const shared = sharedFilesWarning(preview)
  const sharedCount = sharedFileCount(preview)
  const destination = deleteDestination(preview?.mode)
  // No preview = no ⚠ banner, no verified destination. The button must not arm
  // on evidence that never arrived (see deletePreviewState).
  const check = deletePreviewState(preview)
  // The one refusal that is not a warning: this bank's folder belongs to a
  // dataset, so these "rejects" are its training images. The server says no —
  // the button must never even arm.
  const block = preview?.source_write_block
    ? { blocked: true, title: 'Source files are protected', text: preview.source_write_block }
    : datasetConflictBlock(preview?.dataset_conflict)

  const run = async () => {
    if (busy || !armed || !check.ready || block.blocked) return
    setBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/delete-rejected`, {})
      if (d.skipped?.length) console.warn('Delete rejected — skipped files:', d.skipped)
      const note = deleteRejectedStart(d, {
        destination: deleteDestination(d.mode),
        recoverable: isRecoverable(d.mode),
      })
      toast[note.type](note.text)
      onDone?.()
    } catch (e) {
      toast.error(e?.message || 'Delete failed.')
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Delete rejected from disk"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="w-full max-w-md max-h-[90vh] overflow-y-auto rounded-xl border border-rose-500/60 bg-surface-overlay p-4 sm:p-5 shadow-2xl space-y-4">
        <h2 className="text-base font-bold text-rose-300">🗑 Delete rejected from disk</h2>
        <div className="rounded-md border border-rose-500/50 bg-rose-500/10 p-3 text-sm text-rose-200 space-y-2">
          <p className="font-semibold">
            This removes {count} rejected file{count === 1 ? '' : 's'} from your disk.
          </p>
          <p className="text-rose-200/90">
            Every image currently marked ✕ rejected leaves its source folder{' '}
            {check.ready
              ? <>for <span className="font-semibold">{destination}</span></>
              : <span className="font-semibold">— where to is still being checked</span>}
            . Kept and undecided images are left untouched.
          </p>
        </div>
        {!check.ready && (
          <div className={`rounded-md border p-3 text-sm space-y-1 ${check.state === 'failed'
            ? 'border-amber-500/60 bg-amber-500/10 text-amber-200'
            : 'border-border bg-surface-raised text-content-muted'}`}>
            <p className="font-semibold">{check.title}</p>
            <p className="text-xs">{check.text}</p>
          </div>
        )}
        {block.blocked && (
          <div className="rounded-md border border-rose-500/70 bg-rose-500/15 p-3 text-sm text-rose-100 space-y-1">
            <p className="font-semibold">{block.title}</p>
            <p className="text-rose-100/90">{block.text}</p>
          </div>
        )}
        {shared && (
          <div className="rounded-md border border-amber-500/60 bg-amber-500/10 p-3 text-sm text-amber-200 space-y-1">
            <p className="font-semibold">
              ⚠ Another bank uses these files
              {sharedCount ? ` (${sharedCount})` : ''}
            </p>
            <p className="text-amber-200/90">{shared}</p>
          </div>
        )}
        <p className="text-xs text-content-subtle">
          Source folder:{' '}
          <span className="font-mono text-content-muted break-all">{sourcePath}</span>
        </p>
        <div className={block.blocked ? 'opacity-40 pointer-events-none' : undefined}>
          <label htmlFor="delete-rejected-confirm" className="block text-sm text-content">
            Type <span className="font-mono font-bold text-rose-300">DELETE</span> to confirm
          </label>
          <input id="delete-rejected-confirm" type="text" autoComplete="off"
            value={confirm} onChange={(e) => setConfirm(e.target.value)}
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
        </div>
        <div className="flex flex-col-reverse sm:flex-row sm:justify-end gap-2">
          <button type="button" onClick={onClose} autoFocus
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="button" onClick={run}
            disabled={busy || !armed || !check.ready || block.blocked}
            title={block.blocked ? block.title : (check.ready ? undefined : check.title)}
            className="rounded-md bg-rose-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-rose-500 disabled:opacity-40">
            {busy ? 'Deleting…'
              : block.blocked ? 'Blocked'
                : check.state === 'checking' ? 'Checking…'
                  : `Delete ${count} file${count === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>
    </div>
  )
}

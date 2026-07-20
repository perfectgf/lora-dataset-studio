import { useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'

/** 🗑 Delete rejected from disk — the ONE bank action that writes to the source
 * folder. Unlike the reversible reject STATUS, this removes the actual files.
 * With send2trash installed they go to the OS trash; otherwise they are gone for
 * good — either way the app's own trash cannot recover them. Gated behind a
 * type-DELETE confirmation (an irreversible destructive action deserves a
 * deliberate keystroke, not a mis-click). Cancel is the default focus. */
export default function DeleteRejectedDialog({ bankId, count, sourcePath, onClose, onDone }) {
  const toast = useToast()
  const [confirm, setConfirm] = useState('')
  const [busy, setBusy] = useState(false)
  const armed = confirm.trim().toUpperCase() === 'DELETE'

  const run = async () => {
    if (busy || !armed) return
    setBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/delete-rejected`, {})
      const gone = (d.deleted || 0) + (d.trashed || 0) + (d.already_absent || 0)
      const where = d.mode === 'trash' ? 'moved to the OS trash' : 'permanently deleted'
      let msg = `${gone} rejected file(s) ${where}.`
      if (d.skipped?.length) msg += ` ${d.skipped.length} skipped (see console).`
      if (d.skipped?.length) console.warn('Delete rejected — skipped files:', d.skipped)
      toast.success(msg)
      onDone?.()
    } catch (e) {
      toast.error(e?.message || 'Delete failed.')
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Delete rejected from disk"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4">
      <div className="w-full max-w-md rounded-xl border border-rose-500/60 bg-surface-overlay p-5 shadow-2xl space-y-4">
        <h2 className="text-base font-bold text-rose-300">🗑 Delete rejected from disk</h2>
        <div className="rounded-md border border-rose-500/50 bg-rose-500/10 p-3 text-sm text-rose-200 space-y-2">
          <p className="font-semibold">
            This deletes {count} rejected file{count === 1 ? '' : 's'} from your disk. This cannot be undone.
          </p>
          <p className="text-rose-200/90">
            Every image currently marked ✕ rejected is removed from its source folder
            (sent to your OS trash when available, otherwise permanently deleted).
            The app&apos;s own trash cannot bring these back — they are files outside the app.
            Kept and undecided images are left untouched.
          </p>
        </div>
        <p className="text-xs text-content-subtle">
          Source folder:{' '}
          <span className="font-mono text-content-muted break-all">{sourcePath}</span>
        </p>
        <div>
          <label htmlFor="delete-rejected-confirm" className="block text-sm text-content">
            Type <span className="font-mono font-bold text-rose-300">DELETE</span> to confirm
          </label>
          <input id="delete-rejected-confirm" type="text" autoComplete="off"
            value={confirm} onChange={(e) => setConfirm(e.target.value)}
            className="mt-1 w-full rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content" />
        </div>
        <div className="flex justify-end gap-2">
          <button type="button" onClick={onClose} autoFocus
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="button" onClick={run} disabled={busy || !armed}
            className="rounded-md bg-rose-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-rose-500 disabled:opacity-40">
            {busy ? 'Deleting…' : `Delete ${count} file${count === 1 ? '' : 's'}`}
          </button>
        </div>
      </div>
    </div>
  )
}

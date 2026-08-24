import { useEffect, useState } from 'react'
import { postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'

/** 🧹 Forget the rows whose source file is no longer in the folder.
 *
 * The folder-sync warning has TWO honest causes and used to offer one remedy.
 * A folder that MOVED is 📦 Move folder…'s job — repoint the bank, lose
 * nothing. This dialog is for the other cause: files really deleted from the
 * folder (a downloader that cleans up its own intermediates, a sync client, a
 * by-hand tidy). Their rows failed to load for ever and kept counting against
 * the bank's ceiling, with no button that let go of them.
 *
 * The count shown is a FRESH server-side walk done when the dialog opens —
 * never the banner's possibly-stale number — and the same walk backs the
 * delete. Rows only: nothing on disk is ever touched (the premise is that the
 * files are already gone). The dropped rows take their keep/reject decisions
 * and analyses with them, so the dialog says so before the click. A folder
 * that cannot be walked at all is refused by the server (an unplugged drive
 * must never be able to erase a triage) and reported here as-is. */
export default function ForgetMissingDialog({ bankId, bankName, onClose, onDone }) {
  const toast = useToast()
  const [preview, setPreview] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let live = true
    postJson(`/api/bank/${bankId}/forget-missing`, {})
      .then((d) => { if (live) setPreview(d) })
      .catch((e) => { if (live) setError(e?.message || 'Could not check the folder.') })
    return () => { live = false }
  }, [bankId])

  const missing = Number(preview?.missing) || 0
  const present = Number(preview?.present) || 0

  const apply = async () => {
    if (busy || !missing) return
    setBusy(true)
    setError(null)
    try {
      const d = await postJson(`/api/bank/${bankId}/forget-missing`, { confirm: true })
      toast.success(`${d.removed} missing row(s) forgotten — this bank now lists `
        + `the ${Number(d.remaining).toLocaleString('en-US')} image(s) that are on disk.`)
      onDone?.(d)
      onClose?.()
    } catch (e) {
      setError(e?.message || 'Could not forget the missing rows.')
      setBusy(false)
    }
  }

  return (
    <div role="dialog" aria-modal="true" aria-label="Forget the missing images"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-3 sm:p-4">
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 sm:p-5 shadow-2xl space-y-4">
        <h2 className="text-base font-bold text-content">🧹 Forget the missing images</h2>
        <p className="text-sm text-content-muted">
          Files that were deleted from{' '}
          <span className="font-semibold text-content">{bankName}</span>&apos;s folder
          leave their rows behind: they fail to load, muddy the counters and still
          count against the bank&apos;s image ceiling. This drops exactly those rows.
          Nothing on disk is touched — if the folder simply moved, use 📦 Move
          folder… instead, which keeps every row.
        </p>

        {error && (
          <p className="rounded-md border border-rose-500/60 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {error}
          </p>
        )}
        {!preview && !error && (
          <p className="text-sm text-content-subtle">Checking the folder…</p>
        )}
        {preview && (missing > 0 ? (
          <div className="rounded-md border border-amber-500/60 bg-amber-500/10 px-3 py-2 text-sm text-amber-200 space-y-1">
            <p className="font-semibold">
              {missing.toLocaleString('en-US')} row(s) point at a file that is no
              longer in the folder — {present.toLocaleString('en-US')} image(s) are
              on disk and stay exactly as they are.
            </p>
            <p className="opacity-90">
              The forgotten rows take their keep/reject decisions, scores and
              captions with them. This cannot be undone.
            </p>
            {preview.missing_sample?.length > 0 && (
              <ul className="mt-1 space-y-0.5 font-mono text-xs opacity-80">
                {preview.missing_sample.map((rel) => (
                  <li key={rel} className="break-all">· {rel}</li>
                ))}
                {missing > preview.missing_sample.length && (
                  <li>· …and {(missing - preview.missing_sample.length).toLocaleString('en-US')} more</li>
                )}
              </ul>
            )}
          </div>
        ) : (
          <p className="rounded-md border border-emerald-500/50 bg-emerald-500/10 px-3 py-2 text-sm text-emerald-200">
            Every row&apos;s file is on disk — there is nothing to forget.
          </p>
        ))}

        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button type="button" onClick={onClose} autoFocus
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="button" onClick={apply} disabled={busy || !missing}
            className="rounded-md bg-amber-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-amber-500 disabled:opacity-40">
            {busy ? 'Forgetting…' : `🧹 Forget ${missing.toLocaleString('en-US')} row(s)`}
          </button>
        </div>
      </div>
    </div>
  )
}

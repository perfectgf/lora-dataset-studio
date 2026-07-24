import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'

/** Ask the SERVER to open its native "choose a folder" dialog (the folder lives
 * on the machine running the app, so a browser file-picker can't reach it).
 * Resolves to the endpoint's answer — {available, path?, cancelled?, reason?} —
 * and never throws for the expected "no desktop on this server" case: the
 * endpoint replies 200 with available:false and the caller falls back to the
 * in-app browser. A genuine network error also degrades to available:false. */
export async function pickNativeFolder(initial) {
  try {
    return await postJson('/api/system/pick-folder', { initial: initial || '' })
  } catch {
    return { available: false, reason: 'network' }
  }
}

/** Read-only in-app folder browser (drives → subfolders), the fallback when the
 * server has no native dialog — used from the LAN/tablet or a Linux/vast.ai box.
 * Nothing is written; only directories are listed. onPick(path) then onClose. */
export function FolderBrowserModal({ initial, onPick, onClose }) {
  const [path, setPath] = useState(initial || null)
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const load = useCallback(async (p) => {
    setLoading(true); setError('')
    try {
      const q = p ? `?path=${encodeURIComponent(p)}` : ''
      const d = await apiFetch(`/api/system/list-folders${q}`)
      setData(d)
      setPath(d.path)
    } catch (e) {
      // A bad starting path (e.g. a stale pasted value) shouldn't dead-end the
      // browser — surface it and drop back to the drive list.
      setError(e?.message || 'Could not open that folder.')
      if (p) load(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load(initial || null) }, [load, initial])

  const entries = data?.entries || []
  const atRoot = !data || data.is_root

  return (
    <div role="dialog" aria-modal="true" aria-label="Choose a folder"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <div className="flex w-full max-w-lg flex-col rounded-xl border border-border bg-surface-overlay p-5 shadow-2xl"
        style={{ maxHeight: '80vh' }}>
        <h2 className="text-base font-bold text-content">📁 Choose a folder</h2>
        <p className="mt-1 text-xs text-content-muted">
          Folders on the machine running the app. Nothing is opened or modified —
          you're only picking a location.
        </p>

        <div className="mt-3 flex items-center gap-2">
          <button type="button" onClick={() => load(atRoot ? null : (data?.parent ?? null))}
            disabled={loading || atRoot}
            className="rounded-md border border-border px-2 py-1 text-xs text-content hover:bg-surface-raised disabled:opacity-40">
            ⬆ Up
          </button>
          <span className="min-w-0 grow truncate font-mono text-xs text-content-subtle"
            title={data?.path || 'This computer'}>
            {data?.path || 'This computer'}
          </span>
        </div>

        {error && <p className="mt-2 text-xs text-amber-300">{error}</p>}

        <ul className="mt-2 grow overflow-y-auto rounded-md border border-border bg-surface-raised">
          {loading ? (
            <li className="px-3 py-2 text-xs text-content-muted">Loading…</li>
          ) : entries.length === 0 ? (
            <li className="px-3 py-2 text-xs text-content-muted">No subfolders here.</li>
          ) : entries.map((e) => (
            <li key={e.path}>
              <button type="button" onClick={() => load(e.path)}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm text-content hover:bg-surface">
                <span aria-hidden="true">📁</span>
                <span className="min-w-0 truncate">{e.name}</span>
              </button>
            </li>
          ))}
        </ul>

        <div className="mt-4 flex justify-end gap-2">
          <button type="button" onClick={onClose}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised">
            Cancel
          </button>
          <button type="button" disabled={atRoot || loading}
            onClick={() => { onPick(data.path); onClose() }}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-white disabled:opacity-50">
            Use this folder
          </button>
        </div>
      </div>
    </div>
  )
}

/** A path text field with a Browse… button. The field stays editable (pasting a
 * path still works); Browse tries the server's native dialog first and, if the
 * server has no desktop, opens the in-app folder browser. Reused by the Image
 * bank and dataset folder-import. */
export default function FolderPickerField({
  id, label, value, onChange, placeholder, required, hint,
}) {
  const [busy, setBusy] = useState(false)
  const [browsing, setBrowsing] = useState(false)

  const browse = async () => {
    if (busy) return
    setBusy(true)
    try {
      const r = await pickNativeFolder(value)
      if (r.available) {
        if (r.path) onChange(r.path)
        // r.cancelled → the user backed out of the native dialog; do nothing.
      } else {
        setBrowsing(true)  // no native dialog here → the in-app browser
      }
    } finally {
      setBusy(false)
    }
  }

  return (
    <div>
      {label && (
        <label htmlFor={id} className="block text-sm font-medium text-content">{label}</label>
      )}
      <div className="mt-1 flex items-stretch gap-2">
        <input id={id} value={value} onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder} required={required}
          className="w-full min-w-0 grow rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm text-content font-mono" />
        <button type="button" onClick={browse} disabled={busy}
          className="shrink-0 rounded-md border border-border bg-surface-raised px-3 py-1.5 text-sm font-semibold text-content hover:bg-surface disabled:opacity-50">
          {busy ? 'Opening…' : '📂 Browse…'}
        </button>
      </div>
      {hint && <p className="mt-1 text-xs text-content-muted">{hint}</p>}
      {browsing && (
        <FolderBrowserModal initial={value || null}
          onPick={onChange} onClose={() => setBrowsing(false)} />
      )}
    </div>
  )
}

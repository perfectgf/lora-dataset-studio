import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { attemptModalSubmit } from '../../utils/submitOutcome.js'

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
 * Nothing is written; only directories are listed.
 *
 * `onPick(path)` MUST answer {ok:true} or {ok:false, error} (or throw). It used
 * to be fired unawaited and followed by an unconditional onClose(), so a refused
 * folder import ("no images in that folder", another dataset job running) closed
 * the browser and threw away both the chosen path and the position in the tree —
 * on a machine with no native dialog, walking back down there is the whole cost.
 * A host that genuinely cannot fail says so explicitly (see FolderPickerField):
 * silence is not a success. */
export function FolderBrowserModal({ initial, onPick, onClose }) {
  // What the user is TYPING, kept apart from `data.path` (where the browser
  // actually is): a half-typed path must not be mistaken for the current folder,
  // and "Use this folder" must keep committing what is on screen, not the draft.
  const [typed, setTyped] = useState(initial || '')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  // Two different failures, two different boxes: `browseError` is "I could not
  // LIST that folder" (amber, informational, the list falls back to the drives);
  // `error` is "the host REFUSED the folder you picked" (red, blocking, keeps
  // you exactly where you are so you can pick another one).
  const [browseError, setBrowseError] = useState('')
  const [error, setError] = useState('')

  /* ONE way out, shut only while the pick is being posted. */
  const dismiss = () => { if (!busy) onClose() }
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') dismiss() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, onClose])  // eslint-disable-line react-hooks/exhaustive-deps

  const use = async () => {
    if (busy || !data?.path) return
    setBusy(true)
    setError('')
    let outcome
    try {
      outcome = await attemptModalSubmit(() => onPick(data.path),
        { fallback: 'That folder was refused' })
    } finally { setBusy(false) }
    if (outcome.close) onClose()
    else setError(outcome.error)
  }

  const load = useCallback(async (p) => {
    setLoading(true); setBrowseError('')
    try {
      const q = p ? `?path=${encodeURIComponent(p)}` : ''
      const d = await apiFetch(`/api/system/list-folders${q}`)
      setData(d)
      // Follow the browser: after a click, an Up, or a successful jump, the box
      // shows where you ARE, so the next paste replaces a real path.
      setTyped(d.path || '')
    } catch (e) {
      // A bad starting path (e.g. a stale pasted value) shouldn't dead-end the
      // browser — surface it and drop back to the drive list.
      setBrowseError(e?.message || 'Could not open that folder.')
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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget) dismiss() }}>
      <div className="flex w-full max-w-lg flex-col rounded-xl border border-border bg-surface-overlay p-5 shadow-2xl"
        style={{ maxHeight: '80vh' }}>
        <h2 className="text-base font-bold text-content">📁 Choose a folder</h2>
        <p className="mt-1 text-xs text-content-muted">
          Folders on the machine running the app. Nothing is opened or modified —
          you're only picking a location.
        </p>

        {/* An address bar, for the same reason the native dialog needed one: the
            path is very often already on the clipboard (someone sent it, or it
            was copied out of Explorer), and clicking down to it folder by folder
            is pure friction. This is also the ONLY way to paste a path on the
            lanes that never get a native dialog at all — LAN, tablet, Linux —
            where this browser is the whole picker. Enter jumps; a path that does
            not exist reports itself in the amber box and leaves you put. */}
        <form className="mt-3 flex items-center gap-2"
          onSubmit={(e) => { e.preventDefault(); const v = typed.trim(); if (v) load(v) }}>
          <button type="button" onClick={() => load(atRoot ? null : (data?.parent ?? null))}
            disabled={loading || atRoot}
            className="shrink-0 rounded-md border border-border px-2 py-1 text-xs text-content hover:bg-surface-raised disabled:opacity-40">
            ⬆ Up
          </button>
          <input
            aria-label="Folder path"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            placeholder={data?.path || 'Paste or type a path'}
            spellCheck={false}
            className="min-w-0 grow rounded-md border border-border bg-surface-raised px-2 py-1 font-mono text-xs text-content placeholder:text-content-subtle" />
          <button type="submit" disabled={loading || !typed.trim()}
            className="shrink-0 rounded-md border border-border px-2 py-1 text-xs text-content hover:bg-surface-raised disabled:opacity-40">
            Go
          </button>
        </form>

        {browseError && <p className="mt-2 text-xs text-amber-300">{browseError}</p>}

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

        {/* shrink-0 so the list above (grow + overflow) gives up the room instead
            of squashing this to a clipped sliver; max-h-24 so a long refusal
            cannot push "Use this folder" off a 400-px screen. */}
        {error && (
          <div role="alert"
            className="mt-2 shrink-0 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 max-h-24 overflow-y-auto">
            <span className="block whitespace-pre-wrap break-words text-xs leading-relaxed text-red-200">
              {error}
            </span>
            <span className="mt-1 block text-[0.625rem] text-content-subtle">
              You are still where you were — pick another folder and try again.
            </span>
          </div>
        )}

        <div className="mt-4 flex shrink-0 justify-end gap-2">
          <button type="button" onClick={dismiss} disabled={busy}
            className="rounded-md border border-border px-3 py-1.5 text-sm text-content hover:bg-surface-raised disabled:opacity-50">
            Cancel
          </button>
          <button type="button" disabled={busy || atRoot || loading} onClick={use}
            className="rounded-md bg-gradient-primary px-4 py-1.5 text-sm font-semibold text-gray-950 disabled:opacity-50">
            {busy ? 'Using…' : 'Use this folder'}
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
        /* This host only writes the path into the field above — nothing can
           refuse it — but it says {ok:true} out loud rather than returning
           nothing: the browser treats silence as "no answer", on purpose. */
        <FolderBrowserModal initial={value || null}
          onPick={(p) => { onChange(p); return { ok: true } }}
          onClose={() => setBrowsing(false)} />
      )}
    </div>
  )
}

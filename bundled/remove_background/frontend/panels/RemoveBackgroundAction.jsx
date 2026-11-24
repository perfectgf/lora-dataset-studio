import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { useFocusTrap } from '@lds/plugin-sdk/ui'
import { importFrom } from '../lib/api.js'
import Editor from './Editor.jsx'

const BUTTON = {
  dataset: 'min-h-10 lg:min-h-9 w-full sm:w-auto px-3 py-1.5 rounded-lg border border-indigo-400/50 bg-indigo-500/20 hover:bg-indigo-500/30 text-indigo-100 text-xs font-semibold disabled:cursor-not-allowed disabled:opacity-45',
  bank: 'min-h-10 lg:min-h-0 rounded-lg border border-white/25 px-4 py-2 text-sm text-white disabled:opacity-50 hover:bg-white/10',
  gallery: 'min-h-10 lg:min-h-0 inline-flex items-center gap-2 rounded-lg border border-indigo-400/50 bg-indigo-500/20 px-3 py-1.5 text-[0.75rem] font-semibold text-indigo-100 hover:bg-indigo-500/30 disabled:cursor-not-allowed disabled:opacity-45',
}

/** Where this picture can be read from, or null when there is no file to work on yet. */
function source(surface, img, bankId) {
  if (!img || img.id == null) return null
  if (surface === 'bank') return bankId == null ? null : { surface, url: `/api/bank/${bankId}/file/${img.id}`, name: img.filename || img.name }
  return img.filename || img.url ? { surface, imageId: img.id } : null
}

/**
 * The viewers' verb: the same editor, over the picture being looked at. The
 * original is copied into the plugin, so nothing here can change it.
 */
export default function RemoveBackgroundAction({ surface, img, bankId, hasRow = true, disabled = false, onLayer }) {
  const [open, setOpen] = useState(false)
  const [picture, setPicture] = useState(null)
  const [error, setError] = useState('')
  const dialog = useRef(null)
  useFocusTrap(dialog, open)
  useEffect(() => { setOpen(false); setPicture(null); setError('') }, [img?.id, surface])
  useEffect(() => {
    onLayer?.(open)
    return () => onLayer?.(false)
  }, [open, onLayer])
  useEffect(() => {
    if (!open) return undefined
    const onKey = event => { if (event.key === 'Escape') { event.stopPropagation(); setOpen(false) } }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open])
  const from = source(surface, img, bankId)
  const key = open && from ? `${from.surface}:${img.id}` : ''
  useEffect(() => {
    if (!key) return undefined
    let alive = true
    setError('')
    importFrom(from).then(found => { if (alive) setPicture(found) })
      .catch(problem => { if (alive) setError(problem.message || 'This picture could not be opened.') })
    return () => { alive = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key])

  if (!from || (surface === 'gallery' && !hasRow) || img._rescueReviewPreview) return null
  return <>
    <button type="button" data-testid="lightbox-remove-background" disabled={disabled} className={BUTTON[surface] || BUTTON.gallery}
      onClick={event => { event.stopPropagation(); setOpen(true) }}
      title="Keep one subject of this picture and remove the rest. The original stays untouched.">
      <span aria-hidden>✂️</span> Remove background
    </button>
    {open && createPortal(<div ref={dialog} role="dialog" aria-modal="true" aria-label="Remove background"
      onClick={event => event.stopPropagation()} onKeyDown={event => { if (event.key !== 'Tab') event.stopPropagation() }}
      className="fixed inset-0 z-[9998] flex flex-col gap-3 overflow-y-auto bg-app p-3 sm:p-4 lg:overflow-hidden">
      <div className="flex items-center gap-3">
        <h1 className="m-0 min-w-0 truncate text-base font-semibold text-content">Remove background</h1>
        <button type="button" onClick={() => setOpen(false)} aria-label="Close"
          className="ml-auto size-10 shrink-0 rounded-full bg-white/10 text-lg text-white hover:bg-white/20">×</button>
      </div>
      {error && <p role="alert" className="m-0 rounded-lg border border-red-400/30 bg-red-500/10 p-3 text-sm text-red-200">{error}</p>}
      {picture ? <Editor picture={picture} onChange={setPicture} className="flex-1" />
        : !error && <p role="status" className="m-auto text-sm text-content-muted">Opening the picture…</p>}
    </div>, document.body)}
  </>
}

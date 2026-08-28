import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import {
  buildModelOptions, filterModelOptions, emptyScanMessage,
  PINNED_MISSING_BADGE, PINNED_MISSING_TITLE,
} from '../../utils/modelFileOptions'

/**
 * Fetch the model files on disk for ONE picker slot (GET
 * /api/comfy/model-files?slot=…). Same shape and same failure policy as
 * useKleinGenerationLoras: on any failure it degrades to an empty list so the
 * field stays a plain free-text input, never a blocking empty dropdown.
 *
 * The scan is server-side and cached by the roots' mtime, so opening a panel
 * never waits on a cold walk of a slow/remote mount: the request is fired on
 * mount and the input is usable the whole time it is in flight.
 */
export function useModelFiles(slot) {
  const [state, setState] = useState({
    files: [], folder: '', loading: true, error: false, rescanning: false,
  })
  const load = useCallback(async (force = false) => {
    setState((s) => ({ ...s, loading: force ? s.loading : true, rescanning: force, error: false }))
    try {
      const qs = new URLSearchParams({ slot, ...(force ? { force: '1' } : {}) })
      const data = await apiFetch(`/api/comfy/model-files?${qs}`)
      setState({
        files: Array.isArray(data?.files) ? data.files : [],
        folder: data?.folder || '',
        loading: false, error: false, rescanning: false,
      })
    } catch {
      setState((s) => ({
        ...s, files: force ? s.files : [], loading: false, rescanning: false, error: true,
      }))
    }
  }, [slot])
  useEffect(() => { load(false) }, [load])
  return { ...state, rescan: () => load(true) }
}

/**
 * Searchable picker over the model files ComfyUI can actually load for one slot.
 *
 * The text input IS the value, exactly like KleinLoraCombobox — these fields
 * accept a full absolute path from outside every ComfyUI root (the Klein slots
 * hardlink it in), and no scan can enumerate that, so free text stays
 * first-class. What the dropdown adds is the list of what IS there.
 *
 * A value that names no scanned file is shown FIRST, flagged "not found", and
 * kept selected — see utils/modelFileOptions.js for why nothing is silently
 * dropped back to auto-detection here.
 */
export default function ModelFilePicker({
  id, value, onChange, ariaLabel, placeholder, folderHint,
  files, folder, loading, error, rescan, rescanning,
}) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState(null)   // null = "showing the value, not a search"
  const [highlight, setHighlight] = useState(0)
  const boxRef = useRef(null)
  const activeRef = useRef(null)

  useEffect(() => {
    if (!open) return undefined
    const onDown = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])
  useEffect(() => { if (open) activeRef.current?.scrollIntoView({ block: 'nearest' }) }, [highlight, open])

  const { options, pinnedMissing } = buildModelOptions(value, files, { loading })
  const shown = filterModelOptions(options, query, value)
  useEffect(() => { setHighlight((h) => Math.min(Math.max(0, h), Math.max(0, shown.length - 1))) },
    [shown.length])

  const where = folderHint || folder || 'ComfyUI’s model folders'
  const emptyMsg = emptyScanMessage({ loading, error, count: (files || []).length, folderHint: where })

  const pick = (name) => { onChange(name); setQuery(null); setOpen(false) }

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open) { setOpen(true); return }
      setHighlight((h) => Math.min(h + 1, shown.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlight((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter') {
      if (open && shown[highlight]) { e.preventDefault(); pick(shown[highlight].name) }
    } else if (e.key === 'Escape') {
      // stopPropagation, not just preventDefault: Escape peels ONE layer. This
      // combobox now also mounts inside window-keydown hosts (the 📷 picker in
      // the dataset lightbox), where an un-stopped Escape would close the open
      // dropdown AND the panel behind it in the same press.
      if (open) { e.preventDefault(); e.stopPropagation(); setOpen(false) }
    }
  }

  return (
    <div ref={boxRef} className="relative">
      <div className="flex items-start gap-1">
        <div className="relative flex-1 min-w-0">
          <input
            id={id}
            type="text"
            aria-label={ariaLabel}
            role="combobox"
            aria-expanded={open}
            aria-autocomplete="list"
            value={value || ''}
            onChange={(e) => { onChange(e.target.value); setQuery(e.target.value); setOpen(true); setHighlight(0) }}
            onFocus={() => setOpen(true)}
            onKeyDown={onKeyDown}
            placeholder={placeholder}
            /* pr-16 reserves the room the "not found" badge sits in. Same value
               as KleinLoraCombobox on purpose: at 400 px this input is already
               narrow, and a padding utility that no other component uses is one
               Tailwind may not have emitted — which renders as text running
               UNDER the badge, which is exactly what the 400 px capture caught. */
            className="min-h-10 lg:min-h-0 mt-0 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 pr-16 text-sm text-content placeholder:text-content-subtle focus:border-primary focus:outline-none"
          />
          {pinnedMissing && (
            <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center">
              <span title={PINNED_MISSING_TITLE}
                className="shrink-0 rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                {PINNED_MISSING_BADGE}
              </span>
            </div>
          )}
        </div>
        <button type="button" onClick={() => rescan?.()} disabled={rescanning}
          title={`Rescan ${where}`}
          aria-label={`Rescan model files for ${ariaLabel}`}
          className="grid h-10 w-10 lg:h-9 lg:w-9 shrink-0 place-items-center rounded-md border border-border text-content-muted hover:bg-surface-raised disabled:opacity-40">
          <span aria-hidden="true" className={rescanning ? 'animate-spin' : ''}>↻</span>
        </button>
      </div>

      {open && (
        <div className="absolute z-20 mt-1 max-h-64 w-full overflow-auto rounded-md border border-border bg-surface-overlay shadow-lg">
          {emptyMsg && <p className="px-2 py-2 text-xs text-content-muted">{emptyMsg}</p>}
          {!emptyMsg && shown.length === 0 && (
            <p className="px-2 py-2 text-xs text-content-muted">No file matches “{value}”.</p>
          )}
          {!emptyMsg && shown.length > 0 && (
            <ul role="listbox">
              {shown.map((o, i) => (
                <li key={o.name}>
                  <button type="button" role="option" aria-selected={i === highlight}
                    ref={i === highlight ? activeRef : null}
                    onMouseDown={(e) => { e.preventDefault(); pick(o.name) }}
                    className={`flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs text-content ${i === highlight ? 'bg-surface-raised' : 'hover:bg-surface-raised'}`}>
                    <span className="flex-1 truncate font-mono" title={o.name}>{o.name}</span>
                    {o.missing && (
                      <span className="shrink-0 rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                        {PINNED_MISSING_BADGE}
                      </span>
                    )}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}

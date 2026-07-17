import { useCallback, useEffect, useRef, useState } from 'react'
import { apiFetch } from '../../api/fetchClient'
import {
  buildVisibleOptions, compatBadge, findLora, isKnownLora,
} from '../../utils/kleinLoraOptions'

/**
 * Fetch the LoRAs on disk for the Klein generation-LoRA picker (GET
 * /api/loras/list). Fetched ONCE at the card level and shared across every preset
 * row's combobox — never per row. On any failure (ComfyUI unconfigured/unreachable,
 * network) it degrades to an empty list so the combobox falls back to a plain
 * free-text field, never a blocking empty dropdown.
 */
export function useKleinGenerationLoras() {
  const [state, setState] = useState({ loras: [], loading: true, error: false, rescanning: false })
  const load = useCallback(async (force = false) => {
    setState((s) => ({ ...s, loading: force ? s.loading : true, rescanning: force, error: false }))
    try {
      const data = await apiFetch(`/api/loras/list${force ? '?force=1' : ''}`)
      setState({ loras: Array.isArray(data?.loras) ? data.loras : [],
        loading: false, error: false, rescanning: false })
    } catch {
      setState((s) => ({ ...s, loras: force ? s.loras : [], loading: false, rescanning: false, error: true }))
    }
  }, [])
  useEffect(() => { load(false) }, [load])
  return { ...state, rescan: () => load(true) }
}

// tone -> badge classes (theme tokens; the graphite+amber palette, no neon green).
const BADGE_TONE = {
  compatible: 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300',
  incompatible: 'border-amber-500/40 bg-amber-500/10 text-amber-300',
  unknown: 'border-border-strong bg-surface-raised text-content-muted',
}

function ArchBadge({ compatible, label }) {
  const b = compatBadge(compatible, label)
  return (
    <span title={b.title}
      className={`shrink-0 rounded border px-1.5 py-0.5 text-[10px] font-medium ${BADGE_TONE[b.tone]}`}>
      {b.text}
    </span>
  )
}

function OptionRow({ entry, active, onPick, refCb }) {
  return (
    <li>
      <button type="button" role="option" aria-selected={active} ref={refCb}
        onMouseDown={(e) => { e.preventDefault(); onPick(entry.name) }}
        className={`flex w-full items-center gap-2 px-2 py-1.5 text-left text-xs text-content ${active ? 'bg-surface-raised' : 'hover:bg-surface-raised'}`}>
        <span className="flex-1 truncate font-mono" title={entry.name}>{entry.name}</span>
        <ArchBadge compatible={entry.compatible} label={entry.label} />
      </button>
    </li>
  )
}

/**
 * Searchable combobox for one preset LoRA row, backed by the on-disk scan.
 *
 * The text input IS the row value, so free-text stays a first-class fallback:
 * exotic configs and files not yet present can always be typed, and typing
 * SUBSTRING-filters the dropdown of scanned LoRAs (grouped Klein-compatible / other
 * arch, each with an arch badge; ≤20 shown). Arrow keys move a highlight, Enter
 * picks it, Escape closes; a click picks too. A ↻ button rescans. When the current
 * value names no scanned file (a renamed/absent LoRA, or ComfyUI down) it is shown
 * with a "not on disk" badge but kept editable — an existing preset is never
 * silently dropped.
 */
export default function KleinLoraCombobox({
  value, onChange, ariaLabel, loras, loading, error, rescan, rescanning,
}) {
  const [open, setOpen] = useState(false)
  const [highlight, setHighlight] = useState(0)
  const boxRef = useRef(null)
  const activeRef = useRef(null)

  // Close on an outside click.
  useEffect(() => {
    if (!open) return undefined
    const onDown = (e) => { if (boxRef.current && !boxRef.current.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const selected = findLora(value, loras)
  // "not on disk" only when we actually have a scan to check against — never cry
  // wolf while loading or when the scan is empty (ComfyUI down => free-text mode).
  const notFound = !!value && !loading && (loras || []).length > 0 && !isKnownLora(value, loras)
  const { options, compatible, other, hiddenCount } = buildVisibleOptions(loras, value)
  const hasSuggestions = options.length > 0

  // Keep the highlight in range as the filtered set shrinks/grows, and scroll it
  // into view during keyboard navigation.
  useEffect(() => { setHighlight((h) => Math.min(Math.max(0, h), Math.max(0, options.length - 1))) }, [options.length])
  useEffect(() => { if (open) activeRef.current?.scrollIntoView({ block: 'nearest' }) }, [highlight, open])

  const pick = (name) => { onChange(name); setOpen(false) }

  const onKeyDown = (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      if (!open) { setOpen(true); return }
      setHighlight((h) => Math.min(h + 1, options.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlight((h) => Math.max(h - 1, 0))
    } else if (e.key === 'Enter') {
      // Pick the highlighted suggestion; never submit the surrounding form. With no
      // open dropdown the typed free text just stays as-is.
      if (open && hasSuggestions && options[highlight]) { e.preventDefault(); pick(options[highlight].name) }
    } else if (e.key === 'Escape') {
      if (open) { e.preventDefault(); setOpen(false) }
    }
  }

  // Flat render index -> so each group's rows know their position in `options`.
  let flat = -1
  const renderRow = (e) => {
    flat += 1
    const i = flat
    return (
      <OptionRow key={e.name} entry={e} active={i === highlight}
        refCb={i === highlight ? activeRef : null} onPick={pick} />
    )
  }

  return (
    <div ref={boxRef} className="relative flex-1 min-w-[200px]">
      <div className="flex items-center gap-1">
        <div className="relative flex-1">
          <input
            type="text" aria-label={ariaLabel}
            role="combobox" aria-expanded={open} aria-autocomplete="list"
            value={value || ''}
            onChange={(e) => { onChange(e.target.value); setOpen(true); setHighlight(0) }}
            onFocus={() => setOpen(true)}
            onKeyDown={onKeyDown}
            placeholder="klein/my-lora.safetensors"
            className="mt-0 w-full rounded-md border border-border-strong bg-surface-raised px-3 py-2 pr-16 text-sm text-content placeholder:text-content-subtle focus:border-primary focus:outline-none"
          />
          <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center gap-1">
            {selected && <ArchBadge compatible={selected.compatible} label={selected.label} />}
            {notFound && (
              <span title="No file with this name was found under ComfyUI's models/loras — it may be renamed, not downloaded yet, or in a config the scan can't see. It's kept as typed."
                className="shrink-0 rounded border border-amber-500/40 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-300">
                not on disk
              </span>
            )}
          </div>
        </div>
        <button type="button" onClick={() => rescan?.()} disabled={rescanning}
          title="Rescan ComfyUI's loras folder"
          aria-label="Rescan LoRAs"
          className="grid h-9 w-9 shrink-0 place-items-center rounded-md border border-border text-content-muted hover:bg-surface-raised disabled:opacity-40">
          <span aria-hidden="true" className={rescanning ? 'animate-spin' : ''}>↻</span>
        </button>
      </div>

      {open && (
        <div className="absolute z-20 mt-1 max-h-64 w-full overflow-auto rounded-md border border-border bg-surface-overlay shadow-lg">
          {loading && <p className="px-2 py-2 text-xs text-content-muted">Scanning LoRAs…</p>}
          {!loading && error && (
            <p className="px-2 py-2 text-xs text-content-muted">
              Couldn&apos;t reach ComfyUI — type the LoRA path by hand.
            </p>
          )}
          {!loading && !error && (loras || []).length === 0 && (
            <p className="px-2 py-2 text-xs text-content-muted">
              No LoRAs found under ComfyUI&apos;s models/loras — type a path, then ↻ to rescan.
            </p>
          )}
          {!loading && hasSuggestions && (
            <ul role="listbox">
              {compatible.length > 0 && (
                <>
                  <li className="sticky top-0 bg-surface-overlay px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
                    Klein-compatible
                  </li>
                  {compatible.map(renderRow)}
                </>
              )}
              {other.length > 0 && (
                <>
                  <li className="sticky top-0 bg-surface-overlay px-2 py-1 text-[10px] font-semibold uppercase tracking-wide text-content-muted">
                    Other arch (loads as a no-op in the Klein graph)
                  </li>
                  {other.map(renderRow)}
                </>
              )}
              {hiddenCount > 0 && (
                <li className="px-2 py-1.5 text-[11px] text-content-subtle">
                  +{hiddenCount} more — keep typing to narrow the list.
                </li>
              )}
            </ul>
          )}
          {!loading && !error && (loras || []).length > 0 && !hasSuggestions && (
            <p className="px-2 py-2 text-xs text-content-muted">No LoRA matches “{value}”.</p>
          )}
        </div>
      )}
    </div>
  )
}

import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'

const GROUPS_PAGE = 25

/** Near-duplicate resolution: one card per unresolved group. "Keep best"
 * keeps the highest-resolution/sharpest member, "Keep first" the oldest by
 * import order; clicking a member keeps THAT one. Losers are rejected (a
 * reversible status) — nothing is ever deleted from disk. */
export default function DupGroupsPanel({ bankId, live, onChanged }) {
  const toast = useToast()
  const [data, setData] = useState(null)
  const [offset, setOffset] = useState(0)
  const [busy, setBusy] = useState(false)

  const refresh = useCallback(async (off = offset) => {
    try {
      const d = await apiFetch(`/api/bank/${bankId}/dup-groups?offset=${off}&limit=${GROUPS_PAGE}`)
      setData(d); setOffset(off)
    } catch (e) {
      toast.error(e?.message || 'Could not load the duplicate groups.')
    }
  }, [bankId, offset, toast])

  useEffect(() => { refresh(0) // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankId])

  const resolve = async (body, okMsg) => {
    if (busy) return
    setBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/dups/resolve`, body)
      toast.success(okMsg || `Resolved ${d.resolved} group(s) — ${d.rejected} duplicate(s) rejected.`)
      await refresh(0)
      onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Resolution failed.')
    } finally {
      setBusy(false)
    }
  }

  if (data == null) return <p className="text-sm text-content-muted">Loading duplicate groups…</p>
  if (data.total === 0) {
    return (
      <p className="text-sm text-content-muted">
        No unresolved duplicate group — either the bank is clean, or every group has been resolved.
        (Groups appear after a 🔎 quality scan.)
      </p>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
        <span className="text-sm font-semibold text-content">
          ≈ {data.total} unresolved group{data.total > 1 ? 's' : ''}
        </span>
        <span className="text-xs text-content-subtle">
          Losers are rejected, never deleted — undo any of it from the ✕ Rejected filter.
        </span>
        <span className="ml-auto" />
        <button type="button" disabled={busy || live}
          onClick={() => resolve({ strategy: 'best' })}
          title="In every group: keep the highest-resolution (then sharpest) member, reject the rest"
          className="rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-white disabled:opacity-50">
          Resolve ALL — keep best
        </button>
        <button type="button" disabled={busy || live}
          onClick={() => resolve({ strategy: 'first' })}
          title="In every group: keep the first member (import order), reject the rest"
          className="rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content disabled:opacity-50 hover:bg-surface">
          Resolve ALL — keep first
        </button>
      </div>

      <ul className="space-y-3">
        {data.groups.map((g) => (
          <li key={g.group} className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex items-center gap-2 text-xs text-content-muted">
              <span className="font-semibold text-content">Group #{g.group}</span>
              <span>{g.images.length} images — click the one to KEEP</span>
              <span className="ml-auto" />
              <button type="button" disabled={busy || live}
                onClick={() => resolve({ strategy: 'best', group: g.group })}
                className="rounded-md border border-border px-2 py-0.5 text-content hover:bg-surface-raised disabled:opacity-50">
                Keep best
              </button>
              <button type="button" disabled={busy || live}
                onClick={() => resolve({ strategy: 'first', group: g.group })}
                className="rounded-md border border-border px-2 py-0.5 text-content hover:bg-surface-raised disabled:opacity-50">
                Keep first
              </button>
            </div>
            <ul className="flex flex-wrap gap-2">
              {g.images.map((img) => (
                <li key={img.id} className="w-32">
                  <button type="button" disabled={busy || live}
                    onClick={() => resolve({ keep_ids: [img.id] },
                      `Kept “${img.name}” — ${g.images.length - 1} duplicate(s) rejected.`)}
                    title={`Keep this one (${img.width || '?'}×${img.height || '?'}, sharpness ${img.blur_score != null ? Math.round(img.blur_score) : '?'})`}
                    className={`relative block w-full overflow-hidden rounded-lg border ${img.id === g.best_id
                      ? 'border-emerald-400 ring-1 ring-emerald-400' : 'border-border'} ${img.status === 'reject' ? 'opacity-50' : ''}`}>
                    <img src={`/api/bank/${bankId}/thumb/${img.id}`} alt={img.name}
                      loading="lazy" className="h-24 w-full object-cover" />
                    {img.id === g.best_id && (
                      <span className="absolute left-1 top-1 rounded bg-emerald-500/90 px-1 text-[10px] font-bold text-white">BEST</span>
                    )}
                  </button>
                  <p className="mt-0.5 truncate text-[10px] text-content-subtle" title={img.name}>
                    {img.width || '?'}×{img.height || '?'} · {img.name}
                  </p>
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>

      {data.total > GROUPS_PAGE && (
        <nav className="flex items-center gap-3 text-sm" aria-label="Duplicate group pages">
          <button type="button" disabled={offset === 0}
            onClick={() => refresh(Math.max(0, offset - GROUPS_PAGE))}
            className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">← Prev</button>
          <span className="text-content-muted">
            groups {offset + 1}–{Math.min(offset + GROUPS_PAGE, data.total)} of {data.total}
          </span>
          <button type="button" disabled={offset + GROUPS_PAGE >= data.total}
            onClick={() => refresh(offset + GROUPS_PAGE)}
            className="rounded-md border border-border px-2 py-1 text-content disabled:opacity-40">Next →</button>
        </nav>
      )}
    </div>
  )
}

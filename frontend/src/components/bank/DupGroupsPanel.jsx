import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import DupCompareLightbox from './DupCompareLightbox'

const GROUPS_PAGE = 25

// Two stages share this panel — exact/resized copies (dHash, stage 1) and
// semantic near-duplicates (crops/variants of the same shot, stage 2). They
// differ only in the endpoints and the wording; the resolution UX is identical.
const KINDS = {
  exact: {
    groupsPath: 'dup-groups',
    resolvePath: 'dups/resolve',
    distinctPath: 'dups/distinct',
    header: (n) => `≈ ${n} unresolved group${n > 1 ? 's' : ''}`,
    lead: 'Losers are rejected, never deleted — undo any of it from the ✕ Rejected filter.',
    cardHint: 'exact or resized copy — ⤢ to compare them full screen, or click the one to KEEP',
    compareTitle: 'Compare the duplicates',
    empty: 'No unresolved duplicate group — either the bank is clean, or every group has been resolved. (Groups appear after a 🔎 quality scan.)',
    pagesLabel: 'Duplicate group pages',
  },
  semantic: {
    groupsPath: 'semantic-dup-groups',
    resolvePath: 'semantic-dups/resolve',
    distinctPath: 'semantic-dups/distinct',
    header: (n) => `✂ ${n} “same shot” group${n > 1 ? 's' : ''}`,
    lead: 'Same shot, different crop/compression — the dHash never linked these. Losers are rejected (reversible), never deleted.',
    cardHint: 'same shot, different crop — ⤢ to compare them full screen, or click the one to KEEP',
    compareTitle: 'Compare the same shots',
    empty: (engineLabel) => `No unresolved semantic near-duplicate group. (Groups appear after the ${engineLabel} semantic index is ready, then ✂ Find crops & variants.)`,
    pagesLabel: 'Semantic duplicate group pages',
  },
}

/** Near-duplicate resolution: one card per unresolved group. "Keep best"
 * keeps the highest-resolution/sharpest (or best-scored) member, "Keep first"
 * the oldest by import order; clicking a member keeps THAT one. Losers are
 * rejected (a reversible status) — nothing is ever deleted from disk. ``kind``
 * selects stage 1 (exact/resized) or stage 2 (semantic crops/variants).
 *
 * ⤢ **Compare** opens the same groups full screen (`DupCompareLightbox`), which
 * is where the choice is actually MADE: the three verdicts on this card are
 * either bulk ones you take on trust or a click on a 96-pixel thumbnail, and a
 * stamp is not a size at which two copies of one shot can be told apart. */
export default function DupGroupsPanel({ bankId, live, onChanged, kind = 'exact',
  semanticLabel = 'CLIP', notDuplicates = 0 }) {
  const k = KINDS[kind] || KINDS.exact
  const toast = useToast()
  const [data, setData] = useState(null)
  const [offset, setOffset] = useState(0)
  const [busy, setBusy] = useState(false)
  // Which group ⤢ Compare opened on; null when the lightbox is closed. `false`
  // is not a group id and `0` could be one, so the open/closed state is its own
  // flag rather than a truthiness test on the id.
  const [comparing, setComparing] = useState(null)

  const refresh = useCallback(async (off = offset) => {
    try {
      const d = await apiFetch(`/api/bank/${bankId}/${k.groupsPath}?offset=${off}&limit=${GROUPS_PAGE}`)
      setData(d); setOffset(off)
    } catch (e) {
      toast.error(e?.message || 'Could not load the duplicate groups.')
    }
  }, [bankId, offset, toast, k.groupsPath])

  useEffect(() => { refresh(0) // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bankId, kind])

  const resolve = async (body, okMsg) => {
    if (busy) return
    setBusy(true)
    try {
      const d = await postJson(`/api/bank/${bankId}/${k.resolvePath}`, body)
      toast.success(okMsg || `Resolved ${d.resolved} group(s) — ${d.rejected} duplicate(s) rejected.`)
      await refresh(0)
      await onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Resolution failed.')
    } finally {
      setBusy(false)
    }
  }

  /* ≠ and its undo. Deliberately NOT routed through `resolve`: that helper's
     success message counts rejections, and the whole point here is that there
     are none. */
  const distinct = async (body, okMsg) => {
    if (busy) return
    setBusy(true)
    try {
      await postJson(`/api/bank/${bankId}/${k.distinctPath}`, body)
      toast.success(okMsg)
      await refresh(0)
      await onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Could not record that.')
    } finally {
      setBusy(false)
    }
  }

  /* The way back from ≠, and it has to be reachable from the EMPTY state too:
     marking the last group not-duplicates empties this panel, and an undo that
     disappears with the thing it undoes is not an undo. */
  const restoreLine = notDuplicates > 0 && (
    <p className="text-xs text-content-subtle">
      ≠ {notDuplicates} group{notDuplicates === 1 ? '' : 's'} marked <em>not duplicates</em> —
      kept whole, never proposed again.{' '}
      <button type="button" disabled={busy || live}
        onClick={() => distinct({ restore: true },
          `${notDuplicates} group(s) put back — they are proposed again.`)}
        className="underline decoration-dotted underline-offset-2 hover:text-content disabled:opacity-50">
        Put them back
      </button>
    </p>
  )

  if (data == null) return <p className="text-sm text-content-muted">Loading duplicate groups…</p>
  if (data.total === 0) {
    return (
      <div className="space-y-2">
        <p className="text-sm text-content-muted">
          {typeof k.empty === 'function' ? k.empty(semanticLabel) : k.empty}
        </p>
        {restoreLine}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
        <span className="text-sm font-semibold text-content">{k.header(data.total)}</span>
        <span className="text-xs text-content-subtle">{k.lead}</span>
        <span className="ml-auto" />
        {/* The careful path leads, and the two bulk ones stay exactly where they
            were: "keep best" is a verdict taken on trust, and this is the button
            that lets you check it before you take it. */}
        <button type="button" onClick={() => setComparing(data.groups[0]?.group ?? null)}
          disabled={!data.groups.length}
          title="Open these groups full screen — side by side or one at a time — and pick the copy to keep with K / R, seeing what you are choosing"
          className="min-h-10 lg:min-h-0 rounded-md bg-gradient-primary px-3 py-1 text-xs font-semibold text-gray-950 disabled:opacity-50">
          ⤢ Compare &amp; pick
        </button>
        <button type="button" disabled={busy || live}
          onClick={() => resolve({ strategy: 'best' })}
          title="In every group: keep the best (aesthetic score, then resolution/sharpness) member, reject the rest"
          className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content disabled:opacity-50 hover:bg-surface">
          Resolve ALL — keep best
        </button>
        <button type="button" disabled={busy || live}
          onClick={() => resolve({ strategy: 'first' })}
          title="In every group: keep the first member (import order), reject the rest"
          className="min-h-10 lg:min-h-0 rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content disabled:opacity-50 hover:bg-surface">
          Resolve ALL — keep first
        </button>
        {restoreLine && <div className="w-full">{restoreLine}</div>}
      </div>

      <ul className="space-y-3">
        {data.groups.map((g) => (
          <li key={g.group} className="rounded-lg border border-border bg-surface p-3">
            <div className="mb-2 flex flex-wrap items-center gap-2 text-xs text-content-muted">
              <span className="font-semibold text-content">Group #{g.group}</span>
              <span>{g.images.length} images — {k.cardHint}</span>
              <span className="ml-auto" />
              <button type="button" onClick={() => setComparing(g.group)}
                title="Compare these copies full screen — side by side or one at a time — then keep one with K"
                className="min-h-10 lg:min-h-0 rounded-md border border-primary/70 bg-primary/15 px-2 py-0.5 font-semibold text-content hover:bg-primary/25">
                ⤢ Compare
              </button>
              <button type="button" disabled={busy || live}
                onClick={() => resolve({ strategy: 'best', group: g.group })}
                className="min-h-10 lg:min-h-0 rounded-md border border-border px-2 py-0.5 text-content hover:bg-surface-raised disabled:opacity-50">
                Keep best
              </button>
              <button type="button" disabled={busy || live}
                onClick={() => resolve({ strategy: 'first', group: g.group })}
                className="min-h-10 lg:min-h-0 rounded-md border border-border px-2 py-0.5 text-content hover:bg-surface-raised disabled:opacity-50">
                Keep first
              </button>
              {/* The only verb on this card that rejects NOTHING. */}
              <button type="button" disabled={busy || live}
                onClick={() => distinct({ group: g.group },
                  `Group #${g.group} kept whole — it will not be proposed again.`)}
                title="These are not duplicates: keep every copy and stop proposing this group. Nothing is rejected, and the line above puts it back."
                className="min-h-10 lg:min-h-0 rounded-md border border-sky-400/60 bg-sky-500/10 px-2 py-0.5 font-semibold text-content hover:bg-sky-500/20 disabled:opacity-50">
                ≠ Not duplicates
              </button>
            </div>
            {/* The tiles are a little bigger than they were: still stamps, but
                stamps you can tell apart at a glance — enough to decide whether
                a group is worth ⤢ opening, which is where the choice is made. */}
            <ul className="flex flex-wrap gap-2">
              {g.images.map((img) => (
                <li key={img.id} className="w-40">
                  <button type="button" disabled={busy || live}
                    onClick={() => resolve({ keep_ids: [img.id] },
                      `Kept “${img.name}” — ${g.images.length - 1} duplicate(s) rejected.`)}
                    title={`Keep this one (${img.width || '?'}×${img.height || '?'}, sharpness ${img.blur_score != null ? Math.round(img.blur_score) : '?'})`}
                    className={`relative block w-full overflow-hidden rounded-lg border ${img.id === g.best_id
                      ? 'border-emerald-400 ring-1 ring-emerald-400' : 'border-border'} ${img.status === 'reject' ? 'opacity-50' : ''}`}>
                    <img src={`/api/bank/${bankId}/thumb/${img.id}`} alt={img.name}
                      loading="lazy" className="h-28 w-full object-cover" />
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
        <nav className="flex items-center gap-3 text-sm" aria-label={k.pagesLabel}>
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

      {/* The lightbox is seeded with the page on screen and refills itself from
          offset 0 — see DupCompareLightbox. The panel is refreshed ONCE, on
          close: refetching after every verdict would rebuild the list behind a
          full-screen overlay nobody can see, at one request per keystroke. */}
      {comparing !== null && (
        <DupCompareLightbox bankId={bankId} groupsPath={k.groupsPath}
          resolvePath={k.resolvePath} distinctPath={k.distinctPath}
          title={k.compareTitle}
          seedGroups={data.groups} startGroup={comparing} live={live}
          onClose={async () => {
            setComparing(null)
            await refresh(0)
            await onChanged?.()
          }} />
      )}
    </div>
  )
}

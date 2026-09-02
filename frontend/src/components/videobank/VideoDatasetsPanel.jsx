import { useCallback, useEffect, useState } from 'react'
import { Clapperboard } from 'lucide-react';
import { useNavigate } from 'react-router'
import { apiFetch, del, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'

/** 🎬 Video training sets, in the library, next to the image datasets.
 *
 * A LIST, and only a list. Each card opens the set's own workspace at
 * `/video-dataset/<id>` — the same relationship the image cards have with theirs.
 * It did not always: the card used to expand an accordion of clips and carry the
 * whole training block, which made the library the place you worked a video set
 * from, and made the set itself unaddressable (no link, no reload, no back
 * button). The workspace owns those verbs now; what stays here is what a library
 * is for — finding the right set and telling the two apart.
 *
 * TWO FIELDS RIDE ON EVERY CARD and they are not decoration: `training_verified`
 * and `licence_note`. They are shown in the workspace as well, but a dataset is
 * something you come BACK to weeks later, and "which of these can I actually
 * train, and which one am I not allowed to publish from where I live" is exactly
 * the question you have while looking at the list.
 */
// Same lazy localStorage read as the image sections above this panel — the
// fold must survive a reload for the same reason theirs does.
const FOLD_KEY = 'ldsVideoSetsCollapsed'

export default function VideoDatasetsPanel() {
  const toast = useToast()
  const navigate = useNavigate()
  const [datasets, setDatasets] = useState(null)
  // 📁 Collapsible like TRAINED / NOT TRAINED YET above it — it was the ONE
  // section of the library that could not be put away (reported next to two
  // sections that fold, which is what made it read as broken).
  const [folded, setFolded] = useState(() => {
    try { return localStorage.getItem(FOLD_KEY) === '1' } catch { return false }
  })
  useEffect(() => {
    try { localStorage.setItem(FOLD_KEY, folded ? '1' : '0') } catch { /* private mode */ }
  }, [folded])

  const refresh = useCallback(async () => {
    try {
      const d = await apiFetch('/api/video-datasets', { background: true })
      setDatasets(d.datasets || [])
    } catch {
      setDatasets([])          // never break the image library over this panel
    }
  }, [])
  useEffect(() => { refresh() }, [refresh])

  const remove = async (ds) => {
    if (!window.confirm(`Delete the video dataset “${ds.name}”?\n\nThe encoded clips are deleted. The bank they came from keeps every shot and every decision — you can re-cut at another length without triaging again.`)) return
    try {
      await del(`/api/video-dataset/${ds.id}`)
      toast.success('Video dataset deleted — the bank’s shots are untouched.')
      refresh()
    } catch (e) {
      toast.error(e?.message || 'Could not delete that dataset.')
    }
  }

  // Loading stays silent; EMPTY no longer does. The old rule (hide the section
  // until a bank is promoted) predates the stills road: the section now carries
  // an entry point of its own, and hiding it would hide the only place a user
  // with zero video datasets can start one from their image datasets.
  if (datasets === null) return null

  return (
    <section className="flex flex-col gap-2">
      <h2 className="flex items-center gap-2">
        <button type="button" onClick={() => setFolded((v) => !v)}
          aria-expanded={!folded}
          title={folded ? 'Expand the Video training sets section'
            : 'Collapse the Video training sets section'}
          className="flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-content-subtle transition-colors hover:text-content">
          <span aria-hidden="true"
            className={`text-[0.625rem] transition-transform ${folded ? '' : 'rotate-90'}`}>
            ▶
          </span>
          <Clapperboard aria-hidden="true" className="h-4 w-4" /> Video training sets
          <span className="font-normal normal-case tracking-normal">({datasets.length})</span>
        </button>
        <HelpBadge topic="video-datasets" />
        {/* Stays reachable folded: it is the section's only entry point for a
            user with zero video datasets, and folding must not hide it. */}
        <StillsFromDatasetButton onCreated={refresh} />
      </h2>
      {folded ? null : (
      <ul className="grid gap-2 grid-cols-1 sm:grid-cols-2">
        {datasets.map((d) => (
          <li key={d.id}
            className="flex min-w-0 flex-col gap-1.5 rounded-lg border border-border bg-surface p-3">
            <div className="flex min-w-0 items-center gap-2">
              <button type="button" onClick={() => navigate(`/video-dataset/${d.id}`)}
                className="min-w-0 flex-1 truncate text-left text-sm font-semibold text-content hover:underline">
                {d.name}
              </button>
              <button type="button" onClick={() => remove(d)}
                aria-label={`Delete video dataset ${d.name}`}
                className="px-1.5 text-content-subtle hover:text-rose-300">✕</button>
            </div>
            <p className="text-xs text-content-muted">
              {d.clips} clip{d.clips === 1 ? '' : 's'} · {d.target_label}
              {d.frames ? ` · ${d.frames} frames` : ''}
              {d.clip_seconds ? ` (${d.clip_seconds.toFixed(2)}s)` : ''}
              {d.fps ? ` @ ${d.fps} fps` : ''}
              {d.width && d.height ? ` · ${d.width}×${d.height}` : ' · source size'}
            </p>
            {/* Read-only here, attachable in the workspace: the card's job is to
                say which set is not launchable yet, not to fix it. */}
            {d.requires_references && d.references === 0 && (
              <p className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
                📎 No identity reference attached — the launch is refused without one.
              </p>
            )}
            {!d.training_verified && (
              <p className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
                ⚠ No LoRA trainer is known to exist for {d.target_label} yet.
              </p>
            )}
            {d.licence_note && (
              <p className="rounded border border-rose-500/60 bg-rose-500/10 px-2 py-1 text-[0.6875rem] text-rose-100">
                ⚖ {d.licence_note}
              </p>
            )}
            <p className="truncate font-mono text-[0.625rem] text-content-subtle" title={d.output_dir}>
              {d.output_dir}
            </p>
            <button type="button" onClick={() => navigate(`/video-dataset/${d.id}`)}
              className="self-start rounded-md border border-border bg-surface-raised px-3 py-1 text-xs font-semibold text-content hover:bg-surface">
              Open →
            </button>
          </li>
        ))}
      </ul>
      )}
    </section>
  )
}

/** 🖼 Build an H3 stills set from an image dataset the user already curated.
 *
 * The image side owns everything a stills set needs — kept images, edited
 * captions, a trigger — so this is a picker and a POST, not a pipeline. Lazy on
 * purpose: the dataset list is fetched when the form opens, never on mount. */
function StillsFromDatasetButton({ onCreated }) {
  const toast = useToast()
  const [open, setOpen] = useState(false)
  const [choices, setChoices] = useState(null)
  const [picked, setPicked] = useState('')
  const [busy, setBusy] = useState(false)

  const openForm = async () => {
    setOpen(true)
    try {
      const d = await apiFetch('/api/dataset/list')
      setChoices((d.datasets || []).filter((x) => x.images_total > 0))
    } catch (e) {
      toast.error(e?.message || 'Could not list image datasets.')
      setOpen(false)
    }
  }

  const create = async () => {
    if (!picked) return
    setBusy(true)
    try {
      const r = await postJson('/api/video-datasets/from-dataset',
        { dataset_id: Number(picked) })
      toast.success(`“${r.name}” created — ${r.clips} still(s), ready to train.`)
      setOpen(false)
      setPicked('')
      onCreated?.()
    } catch (e) {
      toast.error(e?.message || 'Could not build the stills set.')
    } finally {
      setBusy(false)
    }
  }

  if (!open) {
    return (
      <button type="button" onClick={openForm}
        className="rounded border border-border bg-surface-raised px-2 py-0.5 text-[0.6875rem] text-content-muted hover:bg-surface">
        🖼 Stills set from an image dataset
      </button>
    )
  }
  return (
    <span className="flex items-center gap-1.5">
      <select value={picked} onChange={(e) => setPicked(e.target.value)}
        className="rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.6875rem] text-content">
        <option value="">{choices === null ? 'Loading…' : 'Pick an image dataset'}</option>
        {(choices || []).map((c) => (
          <option key={c.id} value={c.id}>{c.name} ({c.images_total})</option>
        ))}
      </select>
      <button type="button" disabled={busy || !picked} onClick={create}
        className="rounded border border-border bg-surface-raised px-2 py-0.5 text-[0.6875rem] font-semibold text-content hover:bg-surface disabled:opacity-40">
        {busy ? 'Building…' : 'Create'}
      </button>
      <button type="button" onClick={() => setOpen(false)}
        className="text-[0.6875rem] text-content-subtle hover:underline">cancel</button>
    </span>
  )
}

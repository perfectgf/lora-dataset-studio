import { useCallback, useEffect, useState } from 'react'
import { Clapperboard } from 'lucide-react';
import { apiFetch, del, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import { clipLabel } from './videoClipFragment'
import VideoTrainingBlock from './VideoTrainingBlock'

/** 🎬 Video training sets, in the library, next to the image datasets.
 *
 * Silent when there are none — a permanently empty section on every visit is how
 * a library stops being read. It appears the moment a bank is promoted.
 *
 * TWO FIELDS RIDE ON EVERY CARD and they are not decoration: `training_verified`
 * and `licence_note`. They are shown at the picker as well, but a dataset is
 * something you come BACK to weeks later, and "which of these can I actually
 * train, and which one am I not allowed to publish from where I live" is exactly
 * the question you have then.
 */
// Same lazy localStorage read as the image sections above this panel — the
// fold must survive a reload for the same reason theirs does.
const FOLD_KEY = 'ldsVideoSetsCollapsed'

export default function VideoDatasetsPanel() {
  const toast = useToast()
  const [datasets, setDatasets] = useState(null)
  const [openId, setOpenId] = useState(null)
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
      if (openId === ds.id) setOpenId(null)
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
            {d.requires_references && (
              <ReferenceAttach ds={d} onChanged={refresh} />
            )}
            <div className="flex min-w-0 items-center gap-2">
              <button type="button" onClick={() => setOpenId(openId === d.id ? null : d.id)}
                aria-expanded={openId === d.id}
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
            {/* ONE training block, both destinations — maintainer's call
               (2026-08-30) after two stacked sections each grew their own
               Steps field and i2v checkbox. The clip list alone stays behind
               the fold; it is the one part that fetches a payload per open. */}
            <VideoTrainingBlock ds={d} />
            {openId === d.id ? (
              <VideoDatasetClips datasetId={d.id} />
            ) : (
              <button type="button" onClick={() => setOpenId(d.id)}
                className="self-start rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content-muted hover:bg-surface">
                {d.clips ? `🎞 Clips & captions (${d.clips})…` : '🎞 Clips…'}
              </button>
            )}
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


/** 📎 Identity references for a ref2va dataset — the launch precondition.
 *
 * The trainer reads these as control images; without them it trains
 * unconditioned in silence, so the server refuses a reference-less launch and
 * this control is how the user satisfies it. Replacing is whole-set: refs are
 * one identity, not an album. */
function ReferenceAttach({ ds, onChanged }) {
  const toast = useToast()
  const [busy, setBusy] = useState(false)

  const upload = async (fileList) => {
    const files = Array.from(fileList || [])
    if (!files.length) return
    setBusy(true)
    try {
      const form = new FormData()
      files.forEach((f) => form.append('files', f))
      const r = await apiFetch(`/api/video-dataset/${ds.id}/references`,
        { method: 'POST', body: form })
      toast.success(`${r.references} reference(s) attached — every clip covered.`)
      onChanged?.()
    } catch (e) {
      toast.error(e?.message || 'Could not attach the references.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <label className="flex cursor-pointer items-center gap-1 text-[0.6875rem] text-content-muted">
      <span className={ds.references > 0 ? '' : 'text-amber-300'}>
        📎 References: {ds.references || 0}{ds.references > 0 ? '' : ' (required)'}
      </span>
      <input type="file" multiple accept="image/*" hidden disabled={busy}
        onChange={(e) => { upload(e.target.files); e.target.value = '' }} />
      <span className="underline">{busy ? 'attaching…' : 'attach'}</span>
    </label>
  )
}


/** The clips of one dataset, each replayable and each captionable.
 *
 * ONE <video> at a time here too, for the same reason as the bank's grid: a
 * hundred mounted players is past the browser's silent ceiling. The clip that is
 * expanded is the clip that is mounted.
 *
 * The caption box writes the .txt sidecar next to the .mp4 — the trainer never
 * reads our database, so a caption saved only here would train the previous text
 * while showing the new one. The server does both in one call and tells us
 * whether the disk write landed; a failed sidecar is said out loud.
 */
const STILL_EXTS = ['.png', '.jpg', '.jpeg', '.webp']

function isStillFile(name) {
  const n = String(name || '').toLowerCase()
  return STILL_EXTS.some((ext) => n.endsWith(ext))
}

function VideoDatasetClips({ datasetId }) {
  const toast = useToast()
  const [items, setItems] = useState(null)
  const [playing, setPlaying] = useState(null)
  const [drafts, setDrafts] = useState({})

  useEffect(() => {
    let alive = true
    apiFetch(`/api/video-dataset/${datasetId}`, { background: true })
      .then((d) => { if (alive) setItems(d.items || []) })
      .catch(() => { if (alive) setItems([]) })
    return () => { alive = false }
  }, [datasetId])

  const save = async (item) => {
    const caption = drafts[item.id] ?? item.caption ?? ''
    try {
      const d = await postJson(
        `/api/video-dataset/${datasetId}/clip/${item.id}/caption`, { caption })
      setItems((list) => list.map((i) => (i.id === item.id ? { ...i, caption: d.caption } : i)))
      if (d.sidecar_written) toast.success('Caption saved.')
      // Not a detail: the trainer reads the FILE. A row saved without its
      // sidecar trains the old text with nothing anywhere to reveal it.
      else toast.warning('Caption saved in the app, but its .txt file could not be written — the trainer reads the file.')
    } catch (e) {
      toast.error(e?.message || 'Could not save that caption.')
    }
  }

  if (items == null) return <p className="text-xs text-content-muted">Loading clips…</p>
  if (!items.length) return <p className="text-xs text-content-muted">No clip in this dataset.</p>

  return (
    <ul className="mt-1 space-y-2 border-t border-border pt-2">
      {items.map((item) => (
        <li key={item.id} className="min-w-0 space-y-1">
          <div className="flex min-w-0 flex-wrap items-center gap-1.5">
            <button type="button" onClick={() => setPlaying(playing === item.id ? null : item.id)}
              className="rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.625rem] font-semibold text-content hover:bg-surface">
              {playing === item.id ? '⏹ Close' : isStillFile(item.filename) ? '🖼 View' : '▶ Play'}
            </button>
            <span className="min-w-0 truncate font-mono text-[0.625rem] text-content-subtle"
              title={`${item.src_relpath} — ${clipLabel(item.start_s, item.end_s)}`}>
              {item.filename}
            </span>
          </div>
          {playing === item.id && (
            isStillFile(item.filename) ? (
              /* A stills set holds images; wrapping one in a <video> renders a
                 dead player (found on a phone the day stills shipped). The
                 server already serves the right mimetype — the tag has to
                 match the file. */
              <img src={`/api/video-dataset/${datasetId}/clip/${item.id}/media`}
                alt={item.caption || item.filename}
                className="w-full rounded bg-black" />
            ) : (
              <video controls autoPlay preload="metadata"
                src={`/api/video-dataset/${datasetId}/clip/${item.id}/media`}
                className="w-full rounded bg-black">
                <track kind="captions" />
              </video>
            )
          )}
          <textarea rows={2} value={drafts[item.id] ?? item.caption ?? ''}
            onChange={(e) => setDrafts((m) => ({ ...m, [item.id]: e.target.value }))}
            onBlur={() => save(item)}
            aria-label={`Caption for ${item.filename}`}
            placeholder="Describe the clip — this is written to the .txt next to it."
            className="w-full rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content" />
        </li>
      ))}
    </ul>
  )
}

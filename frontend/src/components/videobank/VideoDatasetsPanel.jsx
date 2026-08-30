import { useCallback, useEffect, useState } from 'react'
import { Clapperboard } from 'lucide-react';
import { apiFetch, del, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import { clipLabel } from './videoClipFragment'
import { ensureLicenceAck } from './licenceAck'
import VideoDatasetCloudPanel from './VideoDatasetCloudPanel'

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
export default function VideoDatasetsPanel() {
  const toast = useToast()
  const [datasets, setDatasets] = useState(null)
  const [openId, setOpenId] = useState(null)

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

  if (!datasets?.length) return null

  return (
    <section className="flex flex-col gap-2">
      <h2 className="flex items-center gap-2">
        <span className="font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-content-subtle">
          <Clapperboard aria-hidden="true" className="h-4 w-4" /> Video training sets
          <span className="font-normal normal-case tracking-normal"> ({datasets.length})</span>
        </span>
        <HelpBadge topic="video-datasets" />
      </h2>
      <ul className="grid gap-2 grid-cols-1 sm:grid-cols-2">
        {datasets.map((d) => (
          <li key={d.id}
            className="flex min-w-0 flex-col gap-1.5 rounded-lg border border-border bg-surface p-3">
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
            <VideoTrainingSection ds={d} />
            {openId === d.id && <VideoDatasetCloudPanel dataset={d} />}
            {openId === d.id && <VideoDatasetClips datasetId={d.id} />}
          </li>
        ))}
      </ul>
    </section>
  )
}

/** How far each video target has actually been taken — which is not the same
 * question as whether it is wired, and not one answer but two.
 *
 * 'local' is a finished run on this machine; 'cloud' is a finished run on a
 * rented pod. MiniMax H3 is 'cloud' and stays 'cloud': its base is 42 GB and it
 * trains with the weights resident, which is what the pod is for — claiming
 * "trained end to end" on the LOCAL card would borrow a proof from the wrong
 * machine. A target absent from this map is wired from the installed
 * ai-toolkit's own code and preset — correct as far as reading goes, never yet
 * trained end to end anywhere. The card says which, because "it started" and
 * "it works" are different claims and only the user can decide whether to spend
 * a night on the second. */
const PROVEN_ON = { wan22_14b: 'local', minimax_h3: 'cloud' }

/** ▶ Train this dataset — the local run, its progress, and its refusals.
 *
 * The button is deliberately quiet until it has something to say. What it must
 * never do is start silently: MiniMax H3 pulls about 43 GB of weights on its
 * first run, so the server refuses with the repository and the size and this
 * asks, once, before that becomes a night of downloading behind a progress bar
 * that reads "Starting up…".
 *
 * Polling only runs while this dataset's own run is live (or just launched):
 * `active` is answered by the server from the training fence, which names the
 * TABLE as well as the id — a face training of the colliding id must not drive
 * this bar.
 */
function VideoTrainingSection({ ds }) {
  const toast = useToast()
  const [progress, setProgress] = useState(null)
  const [busy, setBusy] = useState(false)
  // The local run used to hard-code 2000 with no field on screen - the one lane
  // where the user could not see, let alone change, what they were about to
  // spend a night on. Same dial as the cloud panel now, same prefill source.
  const [steps, setSteps] = useState(ds?.suggested_steps || 2000)
  const [doI2v, setDoI2v] = useState(false)

  const poll = useCallback(async () => {
    try {
      setProgress(await apiFetch(`/api/video-dataset/${ds.id}/train/progress`,
        { background: true }))
    } catch { /* the card stays useful without its progress line */ }
  }, [ds.id])
  useEffect(() => { poll() }, [poll])

  const active = !!progress?.active
  useEffect(() => {
    if (!active) return undefined
    const t = setInterval(poll, 3000)
    return () => clearInterval(t)
  }, [active, poll])

  const start = async (acceptDownload = false) => {
    // The licence question comes BEFORE anything is spent — not after the
    // download confirm, whose 43 GB would already be an investment in a run
    // the licence answer might forbid.
    if (!ensureLicenceAck(ds, {
      storage: window.localStorage, confirmFn: window.confirm,
    })) return undefined
    setBusy(true)
    try {
      const r = await postJson(`/api/video-dataset/${ds.id}/train`,
        { steps, do_i2v: doI2v, accept_download: acceptDownload })
      toast.success(`Training started — ${r.clips} clips, ${r.steps} steps.`)
      // Things the run will not fail on but that change what to expect from it.
      ;(r.warnings || []).forEach((w) => toast.warning(w))
      poll()
    } catch (e) {
      const body = e?.body
      if (body?.needs_download) {
        // `free_gigabytes` is null when the drive could not be measured. Saying
        // nothing is the only honest rendering — "0 GB free" and "plenty of
        // room" are opposite answers and we have neither.
        const room = typeof body.free_gigabytes === 'number'
          ? ` You have ${body.free_gigabytes.toFixed(1)} GB free there.`
          : ''
        if (window.confirm(`${body.error}\n\nDownload about ${body.gigabytes} GB from ${body.repo}?${room}`)) {
          setBusy(false)
          return start(true)
        }
      } else {
        toast.error(e?.message || 'Could not start training.')
      }
    } finally {
      setBusy(false)
    }
    return undefined
  }

  const stop = async () => {
    try {
      const r = await postJson(`/api/video-dataset/${ds.id}/train/stop`, {})
      // `ok: false` means the fence names another run. Saying "stopped" there
      // would tell the user a GPU was released while ai-toolkit still owns it.
      if (r.ok) toast.success('Training stopped.')
      else toast.warning('That run is not this dataset’s — nothing was stopped.')
      poll()
    } catch (e) {
      toast.error(e?.message || 'Could not stop training.')
    }
  }

  if (!ds.training_verified) return null

  const dl = progress?.download
  return (
    <div className="flex flex-col gap-1 border-t border-border pt-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {active ? (
          <button type="button" onClick={stop}
            className="rounded border border-rose-500/60 bg-rose-500/10 px-2 py-1 text-[0.6875rem] font-semibold text-rose-100 hover:bg-rose-500/20">
            ⏹ Stop training
          </button>
        ) : (
          <>
            <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted">
              Steps
              <input type="number" min={100} step={100} value={steps}
                onChange={(e) => setSteps(Number(e.target.value) || 1000)}
                className="w-20 rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.6875rem] text-content" />
            </label>
            {Boolean(ds?.suggested_steps) && (
              <span className="text-[0.625rem] text-content-subtle">
                suggested for {ds.clips} clips
              </span>
            )}
            {ds.target_profile === 'minimax_h3' && (
              <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted">
                <input type="checkbox" checked={doI2v}
                  onChange={(e) => setDoI2v(e.target.checked)} />
                i2v (first-frame)
              </label>
            )}
            <button type="button" onClick={() => start(false)} disabled={busy}
              className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:bg-surface disabled:opacity-50">
              {busy ? 'Starting…' : '▶ Train this dataset'}
            </button>
          </>
        )}
        <HelpBadge topic="video-train-local" />
      </div>
      {active && (
        <p className="text-[0.6875rem] text-content-muted">
          {dl
            ? `Downloading weights — ${dl.percent ?? 0}%`
            : progress.step != null
              ? `Step ${progress.step}${progress.total ? ` / ${progress.total}` : ''}${progress.loss != null ? ` · loss ${progress.loss}` : ''}${progress.eta ? ` · ${progress.eta} left` : ''}`
              : 'Starting up…'}
        </p>
      )}
      {!active && PROVEN_ON[ds.target_profile] !== 'local' && (
        <p className="text-[0.6875rem] text-content-subtle">
          {PROVEN_ON[ds.target_profile] === 'cloud'
            ? `${ds.target_label} has been trained end to end on a rented pod, but not yet on a local GPU.`
            : `${ds.target_label} is wired from ai-toolkit’s own settings but has not been trained end to end yet.`}
        </p>
      )}
      {/* On the card, not only in the toast after launching: a warning that
          arrives once the run is up is a warning about a decision already made. */}
      {!active && progress?.resolution_note && (
        <p className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
          ⚠ {progress.resolution_note}
        </p>
      )}
      {!!progress?.checkpoints?.length && (
        <p className="text-[0.6875rem] text-content-muted">
          {progress.checkpoints.length} saved checkpoint
          {progress.checkpoints.length === 1 ? '' : 's'} in {progress.run_name}
        </p>
      )}
    </div>
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
              {playing === item.id ? '⏹ Close' : '▶ Play'}
            </button>
            <span className="min-w-0 truncate font-mono text-[0.625rem] text-content-subtle"
              title={`${item.src_relpath} — ${clipLabel(item.start_s, item.end_s)}`}>
              {item.filename}
            </span>
          </div>
          {playing === item.id && (
            <video controls autoPlay preload="metadata"
              src={`/api/video-dataset/${datasetId}/clip/${item.id}/media`}
              className="w-full rounded bg-black">
              <track kind="captions" />
            </video>
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

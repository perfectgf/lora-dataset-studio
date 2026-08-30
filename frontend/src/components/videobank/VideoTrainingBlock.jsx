import { useCallback, useEffect, useState } from 'react'
import { apiFetch, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { HelpBadge } from '../../help/HelpMode'
import {
  videoDatasetCloudUrl, videoDatasetCloudProgressUrl,
  videoDatasetCloudCheckpointsUrl, videoDatasetCheckpointUrl,
  videoDatasetCloudRetryUrl, videoDatasetCloudContinueUrl,
} from './videoBankApi'
import {
  isActive, launchBlockedReason, runSummary, canRetry, canContinue, stepLabel,
} from './videoCloudStatus'
import { ensureLicenceAck } from './licenceAck'

/** Targets that have been trained end to end at least once — locally or on a
 * rented pod, it does not matter which: what the note below cares about is
 * whether a real run has ever finished. A target absent from this set is wired
 * from the installed ai-toolkit's own code and preset — correct as far as
 * reading goes, never yet proven by a run — and the card says so, because "it
 * is wired" and "it works" are different claims and only the user can decide
 * whether to spend a night (or a pod bill) on the second. */
const PROVEN_TARGETS = new Set(['wan22_14b', 'minimax_h3', 'minimax_h3_ref2va'])

/** 🎬 The training block of one video dataset: one set of dials, two
 * destinations, and everything the runs report back.
 *
 * This used to be two stacked sections — a local one and a cloud one — each
 * with its own Steps field and its own i2v checkbox. Two fields for one number
 * read as two different features, and a value typed in one lane silently did
 * not apply to the other. The settings describe the RUN, not the machine, so
 * they are asked for once and the destination is just the button you press
 * (maintainer's call, 2026-08-30).
 *
 * WHY THE COST IS ON SCREEN BEFORE THE CLICK (cloud)
 * A pod is billed from the moment it boots, so the block says the GPU and the
 * hourly price as soon as the run has one, and it refuses a launch it can
 * already tell will fail (no clips, a run already on a pod) rather than
 * rendering the server's 409.
 *
 * WHY THE LOCAL BUTTON MUST NEVER START SILENTLY
 * MiniMax H3 pulls about 43 GB of weights on its first local run, so the server
 * refuses with the repository and the size, and this asks, once, before that
 * becomes a night of downloading behind a bar that reads "Starting up…".
 *
 * WHY A CHECKPOINT IS A STEP AND NOT A FILE
 * A Wan 2.2 LoRA is TWO files at one step — the high-noise and low-noise
 * experts — and either one alone is a LoRA no loader can complete. So the unit
 * on screen is the step, its files are downloaded together, and a MiniMax H3
 * step (one file) renders through the same shape without a special case.
 *
 * Polling is strictly on demand: the local line polls only while this
 * dataset's own run is live (`active` is answered from the training fence,
 * which names the TABLE as well as the id — a face training of the colliding
 * id must not drive this bar), the cloud line only while a pod is on the
 * clock, and GPU offers are fetched on click, never on mount — a library page
 * with a dozen datasets must not fan out a vast.ai search per card.
 */
export default function VideoTrainingBlock({ ds }) {
  const toast = useToast()
  // ONE dial set for both destinations. Prefilled with the server's
  // dataset-sized suggestion (steps scale with the clip count — measured, not
  // vibes; see suggested_steps in video_training.py). Still just a prefill:
  // what the user types is what trains, wherever it trains.
  const [steps, setSteps] = useState(ds?.suggested_steps || 2000)
  const [doI2v, setDoI2v] = useState(false)

  // Local lane.
  const [progress, setProgress] = useState(null)
  const [busyLocal, setBusyLocal] = useState(false)

  // Cloud lane.
  const [run, setRun] = useState(null)
  const [groups, setGroups] = useState([])
  const [busyCloud, setBusyCloud] = useState(false)
  const [tiers, setTiers] = useState(null)
  const [gpuName, setGpuName] = useState('')
  const [tiersBusy, setTiersBusy] = useState(false)

  const pollLocal = useCallback(async () => {
    try {
      setProgress(await apiFetch(`/api/video-dataset/${ds.id}/train/progress`,
        { background: true }))
    } catch { /* the card stays useful without its progress line */ }
  }, [ds.id])
  useEffect(() => { pollLocal() }, [pollLocal])

  const localActive = !!progress?.active
  useEffect(() => {
    if (!localActive) return undefined
    const t = setInterval(pollLocal, 3000)
    return () => clearInterval(t)
  }, [localActive, pollLocal])

  const refreshCloud = useCallback(async () => {
    try {
      setRun(await apiFetch(videoDatasetCloudProgressUrl(ds.id), { background: true }))
    } catch { /* a poll that fails is not worth a toast */ }
    try {
      const d = await apiFetch(videoDatasetCloudCheckpointsUrl(ds.id), { background: true })
      setGroups(d.groups || [])
    } catch { setGroups([]) }
  }, [ds.id])
  useEffect(() => { refreshCloud() }, [refreshCloud])
  useEffect(() => {
    if (!isActive(run?.status)) return undefined
    const t = setInterval(refreshCloud, 5000)
    return () => clearInterval(t)
  }, [run?.status, refreshCloud])

  const fetchTiers = async () => {
    setTiersBusy(true)
    try {
      const d = await apiFetch(`${videoDatasetCloudUrl(ds.id)}/offers?steps=${steps}`)
      setTiers(d.tiers || [])
    } catch (e) {
      toast.error(e?.message || 'Could not list GPU offers.')
    } finally {
      setTiersBusy(false)
    }
  }

  const startLocal = async (acceptDownload = false) => {
    // The licence question comes BEFORE anything is spent — not after the
    // download confirm, whose 43 GB would already be an investment in a run
    // the licence answer might forbid.
    if (!ensureLicenceAck(ds, {
      storage: window.localStorage, confirmFn: window.confirm,
    })) return undefined
    setBusyLocal(true)
    try {
      const r = await postJson(`/api/video-dataset/${ds.id}/train`,
        { steps, do_i2v: doI2v, accept_download: acceptDownload })
      toast.success(`Training started — ${r.clips} clips, ${r.steps} steps.`)
      // Things the run will not fail on but that change what to expect from it.
      ;(r.warnings || []).forEach((w) => toast.warning(w))
      pollLocal()
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
          setBusyLocal(false)
          return startLocal(true)
        }
      } else {
        toast.error(e?.message || 'Could not start training.')
      }
    } finally {
      setBusyLocal(false)
    }
    return undefined
  }

  const stopLocal = async () => {
    try {
      const r = await postJson(`/api/video-dataset/${ds.id}/train/stop`, {})
      // `ok: false` means the fence names another run. Saying "stopped" there
      // would tell the user a GPU was released while ai-toolkit still owns it.
      if (r.ok) toast.success('Training stopped.')
      else toast.warning('That run is not this dataset’s — nothing was stopped.')
      pollLocal()
    } catch (e) {
      toast.error(e?.message || 'Could not stop training.')
    }
  }

  const postCloud = async (url, body, okMessage) => {
    // Every caller of this helper rents a pod (launch, retry, continue), so the
    // licence gate lives here once rather than on three buttons. Retries after
    // a first acknowledged launch pass silently — the yes belongs to the
    // profile, and it was already given.
    if (!ensureLicenceAck(ds, {
      storage: window.localStorage, confirmFn: window.confirm,
    })) return
    setBusyCloud(true)
    try {
      await postJson(url, body)
      toast.success(okMessage)
      refreshCloud()
    } catch (e) {
      toast.error(e?.message || 'The cloud run could not be started.')
    } finally {
      setBusyCloud(false)
    }
  }

  if (!ds.training_verified) return null

  const dl = progress?.download
  const blocked = launchBlockedReason(ds, run)
  const latestGroup = groups[0] || null

  return (
    <section className="flex flex-col gap-1.5 border-t border-border pt-1.5">
      {localActive ? (
        <div className="flex flex-wrap items-center gap-1.5">
          <button type="button" onClick={stopLocal}
            className="rounded border border-rose-500/60 bg-rose-500/10 px-2 py-1 text-[0.6875rem] font-semibold text-rose-100 hover:bg-rose-500/20">
            ⏹ Stop training
          </button>
          <HelpBadge topic="video-train-local" />
        </div>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-1.5">
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
          </div>
          <div className="flex flex-wrap items-center gap-1.5">
            <button type="button" onClick={() => startLocal(false)}
              disabled={busyLocal || !ds.clips}
              className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:bg-surface disabled:opacity-50">
              {busyLocal ? 'Starting…' : '▶ Train on this PC'}
            </button>
            <HelpBadge topic="video-train-local" />
            <button type="button" disabled={busyCloud || Boolean(blocked)}
              onClick={() => postCloud(videoDatasetCloudUrl(ds.id),
                {
                  steps,
                  ...(doI2v ? { do_i2v: true } : {}),
                  ...(gpuName ? { gpu_name: gpuName } : {}),
                },
                'Renting a pod — the panel follows it from here.')}
              className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:bg-surface disabled:opacity-40">
              ☁ Train in the cloud
            </button>
            <HelpBadge topic="video-cloud-training" />
            {tiers === null ? (
              <button type="button" disabled={tiersBusy}
                onClick={fetchTiers}
                className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content-muted hover:bg-surface disabled:opacity-40">
                {tiersBusy ? 'Fetching offers…' : '🔍 Choose a GPU'}
              </button>
            ) : (
              <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted">
                GPU
                <select value={gpuName} onChange={(e) => setGpuName(e.target.value)}
                  className="rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.6875rem] text-content">
                  <option value="">Cheapest suitable</option>
                  {tiers.map((t) => (
                    <option key={t.gpu_name} value={t.gpu_name}>
                      {t.gpu_name} — ${t.dph_total}/h
                      {t.est_minutes != null ? ` · ~${t.est_minutes} min · ~$${t.est_cost}` : ''}
                      {t.exceeds_cap ? ' ⚠ over runtime cap' : ''}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {canRetry(run) && (
              <button type="button" disabled={busyCloud}
                onClick={() => postCloud(videoDatasetCloudRetryUrl(ds.id), { run_id: run.run_id },
                  'Relaunched on a fresh pod with the same settings.')}
                className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content hover:bg-surface disabled:opacity-40">
                ↻ Retry
              </button>
            )}
            {canContinue(run, latestGroup) && (
              <button type="button" disabled={busyCloud}
                onClick={() => postCloud(videoDatasetCloudContinueUrl(ds.id),
                  { run_id: latestGroup.run_id, extra_steps: steps },
                  `Continuing from the last harvested step, +${steps} steps.`)}
                className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content hover:bg-surface disabled:opacity-40">
                ▶ Train further
              </button>
            )}
          </div>
        </>
      )}

      {!localActive && blocked && (
        <p className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
          {blocked}
        </p>
      )}

      {localActive && (
        <p className="text-[0.6875rem] text-content-muted">
          {dl
            ? `Downloading weights — ${dl.percent ?? 0}%`
            : progress.step != null
              ? `Step ${progress.step}${progress.total ? ` / ${progress.total}` : ''}${progress.loss != null ? ` · loss ${progress.loss}` : ''}${progress.eta ? ` · ${progress.eta} left` : ''}`
              : 'Starting up…'}
        </p>
      )}

      {run?.run_id && (
        <p className="text-[0.6875rem] text-content-muted">
          ☁ {runSummary(run)}
          {run.phase_detail ? ` — ${run.phase_detail}` : ''}
        </p>
      )}
      {run?.error && (
        <p className="rounded border border-rose-500/60 bg-rose-500/10 px-2 py-1 text-[0.6875rem] text-rose-100">
          {run.error}
        </p>
      )}

      {!localActive && !PROVEN_TARGETS.has(ds.target_profile) && (
        <p className="text-[0.6875rem] text-content-subtle">
          {ds.target_label} is wired from ai-toolkit’s own settings but has not
          been trained end to end yet.
        </p>
      )}
      {/* On the card, not only in the toast after launching: a warning that
          arrives once the run is up is a warning about a decision already made. */}
      {!localActive && progress?.resolution_note && (
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

      {groups.map((g) => (
        <div key={g.run_id} className="space-y-1">
          {/* Deliberately NOT a second copy of the run line above: that one says
              where the newest run is and what it costs, this one only labels
              which run these files came from — and, when there is one, the run
              they grew out of. */}
          <p className="font-mono text-[0.625rem] text-content-subtle">
            Checkpoints — run #{g.run_id}
            {g.parent_run_id ? ` · continued from #${g.parent_run_id}` : ''}
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {g.steps.map((s) => (
              <li key={`${g.run_id}-${s.step}-${s.final ? 'f' : 'i'}`}
                className="flex items-center gap-1 rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.625rem] text-content">
                <span>{stepLabel(s)}</span>
                {/* One link per FILE, because both experts of a Wan pair have
                    to land side by side for the LoRA to load at all. */}
                {s.files.map((f, i) => (
                  <a key={f} href={videoDatasetCheckpointUrl(ds.id, g.run_id, f)}
                    download title={f}
                    aria-label={`Download ${f}`}
                    className="rounded border border-border px-1 py-0.5 text-content-subtle hover:bg-surface hover:text-content">
                    ⬇{s.files.length > 1 ? ` ${i + 1}` : ''}
                  </a>
                ))}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  )
}

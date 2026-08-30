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

/** ☁ Train this video dataset on a rented GPU, and bring the weights back.
 *
 * Until this panel existed a promoted video dataset was a dead end inside the
 * app: the launcher and the whole cloud lane were reachable only from a Python
 * shell. Everything here is one of the four things a user does with a training —
 * start it, watch it, take the result, run it further — and nothing else.
 *
 * WHY THE COST IS ON SCREEN BEFORE THE CLICK
 * A pod is billed from the moment it boots, so the panel says the GPU and the
 * hourly price as soon as the run has one, and it refuses a launch it can
 * already tell will fail (no clips, no known trainer for the target, a run
 * already on a pod) rather than rendering the server's 409.
 *
 * WHY A CHECKPOINT IS A STEP AND NOT A FILE
 * A Wan 2.2 LoRA is TWO files at one step — the high-noise and low-noise
 * experts — and either one alone is a LoRA no loader can complete. So the unit
 * on screen is the step, its files are downloaded together, and a MiniMax H3
 * step (one file, since ai-toolkit's H3 model writes a single save) renders
 * through the same shape without a special case.
 */
export default function VideoDatasetCloudPanel({ dataset }) {
  const toast = useToast()
  const [run, setRun] = useState(null)
  const [groups, setGroups] = useState([])
  const [busy, setBusy] = useState(false)
  // Prefilled with the server's dataset-sized suggestion (steps scale with the
  // clip count - measured, not vibes; see suggested_steps in video_training.py).
  // Still just a prefill: what the user types is what trains.
  const [steps, setSteps] = useState(dataset?.suggested_steps || 1000)
  // GPU tiers are fetched on demand, never on mount: a library page with a
  // dozen datasets must not fan out a vast.ai search per card.
  const [doI2v, setDoI2v] = useState(false)
  const [tiers, setTiers] = useState(null)
  const [gpuName, setGpuName] = useState('')
  const [tiersBusy, setTiersBusy] = useState(false)

  const fetchTiers = async () => {
    setTiersBusy(true)
    try {
      const d = await apiFetch(`${videoDatasetCloudUrl(id)}/offers?steps=${steps}`)
      setTiers(d.tiers || [])
    } catch (e) {
      toast.error(e?.message || 'Could not list GPU offers.')
    } finally {
      setTiersBusy(false)
    }
  }

  const id = dataset?.id

  const refresh = useCallback(async () => {
    if (!id) return
    try {
      setRun(await apiFetch(videoDatasetCloudProgressUrl(id), { background: true }))
    } catch { /* a poll that fails is not worth a toast */ }
    try {
      const d = await apiFetch(videoDatasetCloudCheckpointsUrl(id), { background: true })
      setGroups(d.groups || [])
    } catch { setGroups([]) }
  }, [id])

  useEffect(() => { refresh() }, [refresh])
  useEffect(() => {
    // Poll only while something is actually on the clock. A permanent 5 s poll
    // on a library page with a dozen datasets is a request storm for a screen
    // where nothing changes.
    if (!isActive(run?.status)) return undefined
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [run?.status, refresh])

  const post = async (url, body, okMessage) => {
    // Every caller of this helper rents a pod (launch, retry, continue), so the
    // licence gate lives here once rather than on three buttons. Retries after
    // a first acknowledged launch pass silently — the yes belongs to the
    // profile, and it was already given.
    if (!ensureLicenceAck(dataset, {
      storage: window.localStorage, confirmFn: window.confirm,
    })) return
    setBusy(true)
    try {
      await postJson(url, body)
      toast.success(okMessage)
      refresh()
    } catch (e) {
      toast.error(e?.message || 'The cloud run could not be started.')
    } finally {
      setBusy(false)
    }
  }

  const blocked = launchBlockedReason(dataset, run)
  const latestGroup = groups[0] || null

  return (
    <section className="mt-1 space-y-2 border-t border-border pt-2">
      <h3 className="flex items-center gap-2 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-content-subtle">
        <span aria-hidden>☁</span> Cloud training
        <HelpBadge topic="video-cloud-training" />
      </h3>

      <div className="flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted">
          Steps
          <input type="number" min={100} step={100} value={steps}
            onChange={(e) => setSteps(Number(e.target.value) || 1000)}
            className="w-20 rounded border border-border bg-surface-raised px-1.5 py-0.5 text-[0.6875rem] text-content" />
        </label>
        {Boolean(dataset?.suggested_steps) && (
          <span className="text-[0.625rem] text-content-subtle">
            suggested for {dataset.clips} clips
          </span>
        )}
        {dataset?.target_profile === 'minimax_h3' && (
          <label className="flex items-center gap-1 text-[0.6875rem] text-content-muted">
            <input type="checkbox" checked={doI2v}
              onChange={(e) => setDoI2v(e.target.checked)} />
            i2v (first-frame)
          </label>
        )}
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
        <button type="button" disabled={busy || Boolean(blocked)}
          onClick={() => post(videoDatasetCloudUrl(id),
            {
              steps,
              ...(doI2v ? { do_i2v: true } : {}),
              ...(gpuName ? { gpu_name: gpuName } : {}),
            },
            'Renting a pod — the panel follows it from here.')}
          className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] font-semibold text-content hover:bg-surface disabled:opacity-40">
          ☁ Train in the cloud
        </button>
        {canRetry(run) && (
          <button type="button" disabled={busy}
            onClick={() => post(videoDatasetCloudRetryUrl(id), { run_id: run.run_id },
              'Relaunched on a fresh pod with the same settings.')}
            className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content hover:bg-surface disabled:opacity-40">
            ↻ Retry
          </button>
        )}
        {canContinue(run, latestGroup) && (
          <button type="button" disabled={busy}
            onClick={() => post(videoDatasetCloudContinueUrl(id),
              { run_id: latestGroup.run_id, extra_steps: steps },
              `Continuing from the last harvested step, +${steps} steps.`)}
            className="rounded border border-border bg-surface-raised px-2 py-1 text-[0.6875rem] text-content hover:bg-surface disabled:opacity-40">
            ▶ Train further
          </button>
        )}
      </div>

      {blocked && (
        <p className="rounded border border-amber-500/50 bg-amber-500/10 px-2 py-1 text-[0.6875rem] text-amber-100">
          {blocked}
        </p>
      )}

      {run?.run_id && (
        <p className="text-[0.6875rem] text-content-muted">
          {runSummary(run)}
          {run.phase_detail ? ` — ${run.phase_detail}` : ''}
        </p>
      )}
      {run?.error && (
        <p className="rounded border border-rose-500/60 bg-rose-500/10 px-2 py-1 text-[0.6875rem] text-rose-100">
          {run.error}
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
                  <a key={f} href={videoDatasetCheckpointUrl(id, g.run_id, f)}
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

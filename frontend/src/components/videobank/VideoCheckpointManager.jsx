import { useCallback, useEffect, useState } from 'react'
import { Trash2 } from 'lucide-react'
import { apiFetch, del, postJson } from '../../api/fetchClient'
import { useToast } from '../common/Toast'
import { postWithConfirmations } from '../../utils/trainingRefusals'
import { ensureLicenceAck } from './licenceAck'
import {
  videoDatasetCheckpointsUrl, videoDatasetCheckpointDeployUrl,
  videoDatasetCheckpointUndeployUrl, videoDatasetCheckpointDeleteUrl,
  videoDatasetCloudContinueUrl, videoDatasetCloudRunUrl, videoDatasetLineageUrl,
} from './videoBankApi'
import VideoLineageGraph from './VideoLineageGraph'
import VideoSampleLightbox from './VideoSampleLightbox'
import { EMPTY_GRAPH_NOTE, MUTED_CLS, PREVIEWS_NOTE, ROW_CLS, graphSummary, nodeGroup } from './videoLineage'
import {
  EMPTY_NOTE, checkpointGroups, continueBody, deleteReport, deployReport,
  describeStepDelete, describeUndeploy, detailsRows, fmtSize, groupSub, groupTitle,
  runDeleteConfirmation, stepActionModel, stepKey, undeployReport,
} from './videoCheckpoints'

// The row styles are the lane's shared ones (videoLineage.js): the list and
// the graph popover draw a verb the same way.
const ROW = ROW_CLS
const MUTED = MUTED_CLS

/** 📦 The Checkpoints & LoRAs section of a VIDEO dataset — presentational.
 *
 * Every decision it renders arrives from `stepActionModel`; every sentence it
 * confirms with comes from videoCheckpoints.js. The component owns nothing but
 * the DOM, which is what lets `node --test` render each state from a payload
 * (video-checkpoints-render.test.mjs) without a server behind it.
 *
 * The unit is the STEP: a Wan 2.2 save is two files at one step, and one row
 * per step with one ⬇ per file is the only shape that never offers half a LoRA. */
export function VideoCheckpointList({
  datasetId, payload, busy = null, details = null, continueTarget = null,
  extraSteps = 1000, onExtraSteps, onConfirmContinue, onCancelContinue,
  onDeploy, onUndeploy, onDelete, onDeleteRun, onContinue, onDetails,
}) {
  const groups = checkpointGroups(payload)
  if (!groups.length) {
    return <p className="m-0 text-xs text-content-muted">{EMPTY_NOTE}</p>
  }
  const ctx = {
    canDeploy: payload?.can_deploy !== false,
    deployFolder: payload?.deploy_folder || 'h3/lds',
    deleteMode: payload?.delete_mode,
  }
  return (
    <div className="flex flex-col gap-3">
      {groups.map((g) => (
        <div key={g.key} className="flex flex-col gap-1.5" data-lane={g.lane}>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
            <p className="m-0 text-xs font-semibold text-content">{groupTitle(g)}</p>
            <p className="m-0 font-mono text-[0.625rem] text-content-subtle">{groupSub(g)}</p>
            {/* The run-level 🗑 the training block used to carry, kept as a
                verb of its own: it clears a whole run — files AND history line
                — where the per-step one trashes one save. */}
            {g.lane === 'cloud' && !g.active && (
              <button type="button" onClick={() => onDeleteRun?.(g)}
                disabled={busy === `${g.key}:run`}
                aria-label={`Delete run ${g.run_id} and its checkpoints`}
                title="Delete this run and its LoRA files"
                className="ml-auto rounded border border-border px-1 py-0.5 text-content-subtle hover:border-rose-500/60 hover:text-rose-200 disabled:opacity-60">
                🗑
              </button>
            )}
          </div>
          {details && details.run_id === g.run_id && g.lane === 'cloud' && (
            <dl className="m-0 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 rounded border border-border bg-app/40 px-2 py-1.5 text-[0.6875rem]"
              aria-label={`Run ${g.run_id} details`}>
              {details.rows.map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="text-content-subtle">{k}</dt>
                  <dd className="m-0 break-words text-content">{v}</dd>
                </div>
              ))}
            </dl>
          )}
          <ul className="m-0 flex list-none flex-col gap-1.5 p-0">
            {g.steps.map((s) => {
              const a = stepActionModel(datasetId, g, s, ctx)
              const rowBusy = typeof busy === 'string' && busy.startsWith(`${a.key}:`)
              return (
                <li key={a.key} data-step-key={a.key}
                  className="flex flex-col gap-1 rounded border border-border bg-surface-raised px-2 py-1.5">
                  <div className="flex flex-wrap items-center gap-1.5 text-[0.6875rem]">
                    <span className="font-medium text-content">{a.label}</span>
                    {a.deployed && (
                      <span className="rounded bg-emerald-500/15 px-1 py-px text-[0.5625rem] font-semibold uppercase text-emerald-200">
                        Deployed
                      </span>
                    )}
                    {/* One ⬇ per FILE: both experts of a pair have to land side
                        by side for the LoRA to load at all. */}
                    {a.files.map((f) => (
                      <a key={f.filename} href={f.url} download title={f.filename}
                        aria-label={`Download ${f.filename}`}
                        className={ROW + ' min-w-0 max-w-full border-emerald-500/40 bg-emerald-600/15 text-emerald-100 no-underline hover:bg-emerald-600/25'}>
                        <span aria-hidden>⬇</span>
                        {/* A single-file step shows its whole filename, which is one
                            unbreakable token: truncated inside the pill (the full
                            name is the title), or a 360-px screen overflows by the
                            width of a dataset name — measured by the probe. */}
                        <span className="truncate">{f.short}</span>
                        {f.size ? <span className="shrink-0 text-emerald-200/70">{fmtSize(f.size)}</span> : null}
                      </a>
                    ))}
                  </div>
                  <div className="flex flex-wrap items-center gap-1">
                    {a.continue.ok ? (
                      <button type="button" disabled={rowBusy} onClick={() => onContinue?.(g, s)}
                        title="Rent a fresh pod and train this LoRA further from exactly this step"
                        className={ROW + ' border-indigo-400/40 bg-indigo-500/15 text-indigo-100 hover:bg-indigo-500/25'}>
                        <span aria-hidden>▶</span> Continue from here
                      </button>
                    ) : (
                      <span className={MUTED}><span aria-hidden>▶</span> {a.continue.reason}</span>
                    )}
                    {a.deployed ? (a.undeploy?.ok ? (
                      <button type="button" disabled={rowBusy} onClick={() => onUndeploy?.(g, s)}
                        title="Remove this LoRA from ComfyUI's loras folder. Reversible: the training save is kept, so you can deploy it again"
                        className={ROW + ' border-emerald-500/40 bg-emerald-600/5 text-emerald-200/90 hover:bg-emerald-600/20'}>
                        <span aria-hidden>⏏</span> {busy === `${a.key}:undeploy` ? 'Undeploying…' : 'Undeploy'}
                      </button>
                    ) : (
                      <span className={MUTED}><span aria-hidden>⏏</span> {a.undeploy?.reason}</span>
                    )) : (a.deploy?.ok ? (
                      <button type="button" disabled={rowBusy} onClick={() => onDeploy?.(g, s)}
                        title={`Deploy this step into ComfyUI's ${a.deploy.folder} folder so the Video Test Studio can test it`}
                        className={ROW + ' border-primary/40 bg-primary/20 text-white hover:bg-primary/30'}>
                        <span aria-hidden>📦</span> {busy === `${a.key}:deploy` ? 'Deploying…' : `Deploy → ${a.deploy.folder}`}
                      </button>
                    ) : (
                      <span className={MUTED}><span aria-hidden>📦</span> {a.deploy?.reason}</span>
                    ))}
                    {a.details && (
                      <button type="button" disabled={rowBusy} onClick={() => onDetails?.(g)}
                        title="What this run was launched with: GPU, price, steps, target, timing"
                        className={ROW + ' border-border bg-app/60 text-content hover:border-indigo-400/50'}>
                        <span aria-hidden>ⓘ</span> Details
                      </button>
                    )}
                    {/* 🗑 in retreat — a quiet text row, not a fourth coloured
                        button one clicks by reflex; its label names what goes. */}
                    {a.del.ok ? (
                      <button type="button" disabled={rowBusy} onClick={() => onDelete?.(g, s)}
                        title={a.del.title}
                        className="ml-auto flex items-center gap-1 px-1 py-0.5 text-[0.625rem] text-content-subtle hover:text-rose-200 disabled:opacity-60">
                        <Trash2 aria-hidden="true" className="h-3 w-3" />
                        {busy === `${a.key}:delete` ? 'Deleting…' : a.del.label}
                      </button>
                    ) : (
                      <span className={MUTED + ' ml-auto'}><span aria-hidden>🗑</span> {a.del.reason}</span>
                    )}
                  </div>
                  {continueTarget === a.key && (
                    <form className="flex flex-wrap items-center gap-1.5 border-t border-border pt-1 text-[0.6875rem]"
                      onSubmit={(e) => { e.preventDefault(); onConfirmContinue?.(g, s) }}>
                      <label className="flex items-center gap-1 text-content-muted">
                        +
                        {/* step={1}, not a round number: with min={1} the browser's
                            step base is 1, so step={100} made 500, 1000 and 2000
                            INVALID and the form never submitted — silently in
                            headless, a tooltip in a real browser (live check, 02/09). */}
                        <input type="number" min={1} step={1} value={extraSteps}
                          onChange={(e) => onExtraSteps?.(e.target.value)}
                          aria-label="Extra steps"
                          className="w-20 rounded border border-border bg-app px-1 py-0.5 text-content" />
                        steps from {a.label.toLowerCase()}
                      </label>
                      <button type="submit" disabled={rowBusy}
                        className={ROW + ' border-indigo-400/40 bg-indigo-500/20 text-indigo-100 hover:bg-indigo-500/30'}>
                        {busy === `${a.key}:continue` ? 'Renting a pod…' : '▶ Train further'}
                      </button>
                      <button type="button" onClick={() => onCancelContinue?.()}
                        className="text-content-subtle hover:text-content">Cancel</button>
                    </form>
                  )}
                </li>
              )
            })}
          </ul>
        </div>
      ))}
    </div>
  )
}

/** The section's owner: one read, the verbs, and the toasts. Re-reads on
 * `refreshKey` — the training block reports its save count through it, so a
 * run that just harvested shows up here without a second poll of its own. */
export default function VideoCheckpointManager({ ds, refreshKey = 0, onSavesChange }) {
  const toast = useToast()
  const [payload, setPayload] = useState(null)
  const [tree, setTree] = useState(null)
  const [sampleTarget, setSampleTarget] = useState(null)
  const [err, setErr] = useState(null)
  const [busy, setBusy] = useState(null)
  const [details, setDetails] = useState(null)
  const [continueTarget, setContinueTarget] = useState(null)
  const [extraSteps, setExtraSteps] = useState(ds?.suggested_steps || 1000)

  const load = useCallback(async () => {
    try {
      setPayload(await apiFetch(videoDatasetCheckpointsUrl(ds.id), { background: true }))
      setErr(null)
    } catch (e) {
      setErr(e?.message || 'Could not list the checkpoints.')
    }
    // The ◉ Graph reads its own tree — same runs, laid out as a genealogy. A
    // tree that fails to load leaves the list standing: the graph is the
    // second view of the same saves, never the only one.
    try {
      setTree(await apiFetch(videoDatasetLineageUrl(ds.id), { background: true }))
    } catch { setTree(null) }
  }, [ds.id])
  useEffect(() => { load() }, [load, refreshKey])

  // After a verb changed what is on disk: this list re-reads, and the training
  // block is told so its "Train further" does not offer a run that is gone.
  const changed = () => { load(); onSavesChange?.() }

  const act = async (key, fn) => {
    setBusy(key)
    try {
      await fn()
    } catch (e) {
      toast.error(e?.message || 'That did not work.')
    } finally {
      setBusy(null)
    }
  }

  const deploy = (g, s) => act(`${stepKey(g, s)}:deploy`, async () => {
    const d = await postJson(videoDatasetCheckpointDeployUrl(ds.id),
      { run_id: g.run_id, step: s.step, final: !!s.final })
    toast.success(deployReport(d))
    changed()
  })

  const undeploy = (g, s) => {
    if (!window.confirm(describeUndeploy(s, payload?.delete_mode))) return
    act(`${stepKey(g, s)}:undeploy`, async () => {
      for (const f of s.files || []) {
        if (f.deployed_as && f.undeployable) {
          await postJson(videoDatasetCheckpointUndeployUrl(ds.id), { deployed_as: f.deployed_as })
        }
      }
      toast.success(undeployReport(s))
      changed()
    })
  }

  const remove = (g, s) => {
    if (!window.confirm(describeStepDelete(g, s, payload?.delete_mode))) return
    act(`${stepKey(g, s)}:delete`, async () => {
      const d = await postJson(videoDatasetCheckpointDeleteUrl(ds.id),
        { run_id: g.run_id, step: s.step, final: !!s.final })
      toast[d.files_kept?.length ? 'warning' : 'success'](deleteReport(d))
      changed()
    })
  }

  const removeRun = (g) => {
    if (!window.confirm(runDeleteConfirmation(g))) return
    act(`${g.key}:run`, async () => {
      const d = await del(videoDatasetCloudRunUrl(ds.id, g.run_id))
      toast.success(`Run #${d.deleted} deleted — ${d.files} file(s) removed.`)
      if (details?.run_id === g.run_id) setDetails(null)
      changed()
    })
  }

  const showDetails = (g) => {
    if (details?.run_id === g.run_id) { setDetails(null); return }
    act(`${g.key}:details`, async () => {
      const d = await apiFetch(videoDatasetCloudRunUrl(ds.id, g.run_id))
      setDetails({ run_id: g.run_id, rows: detailsRows(d) })
    })
  }

  // ▶ rents a pod, so it owes what every pod-renting POST of the video lane
  // owes: the licence question first, then the confirmations loop that turns
  // the server's `PARALLEL_RUN:` refusal into a question rather than an error.
  const confirmContinue = (g, s) => {
    if (!ensureLicenceAck(ds, { storage: window.localStorage, confirmFn: window.confirm })) return
    act(`${stepKey(g, s)}:continue`, async () => {
      const url = videoDatasetCloudContinueUrl(ds.id)
      const d = await postWithConfirmations((b) => postJson(url, b),
        continueBody(g, s, extraSteps), 'Launch anyway (force)')
      if (d === null) return                       // declined: nothing rented
      setContinueTarget(null)
      toast.success(`Continuing from ${stepKey(g, s).split(':').pop() === 'final' ? 'the final save' : `step ${s.step}`}, +${continueBody(g, s, extraSteps).extra_steps} steps — the Training section follows the pod from here.`)
      changed()
    })
  }

  if (err && !payload) {
    return (
      <p className="m-0 flex flex-wrap items-center gap-2 text-xs text-amber-200">
        {err}
        <button type="button" onClick={load} className="rounded border border-border px-2 py-0.5 text-content">Retry</button>
      </p>
    )
  }
  if (!payload) return <p className="m-0 text-xs text-content-subtle">Reading the saves…</p>
  // ▶ from a graph pill opens the SAME inline form as the list row, and brings
  // that row into view: one place to type the extra steps, one POST.
  const continueFrom = (g, s) => {
    const key = stepKey(g, s)
    setContinueTarget(key)
    requestAnimationFrame(() => {
      document.querySelector(`li[data-step-key="${key}"]`)?.scrollIntoView({ block: 'nearest' })
    })
  }
  const hasGraph = (tree?.nodes?.length || 0) > 0
  return (
    <div className="flex flex-col gap-3">
      <details open className="rounded-lg border border-border bg-surface-raised p-2"
        data-probe-reading>
        <summary className="cursor-pointer text-xs font-semibold text-content">
          ◉ Run graph{hasGraph ? ` — ${graphSummary(tree)}` : ''}
        </summary>
        <p className="m-0 mt-1 mb-1.5 text-[0.6875rem] text-content-muted">{hasGraph ? PREVIEWS_NOTE : EMPTY_GRAPH_NOTE}</p>
        {hasGraph && (
          <VideoLineageGraph datasetId={ds.id} tree={tree} busy={busy}
            ctx={{ canDeploy: payload?.can_deploy !== false,
              deployFolder: payload?.deploy_folder || 'h3/lds', deleteMode: payload?.delete_mode }}
            onDeploy={deploy} onUndeploy={undeploy} onDelete={remove}
            onContinue={continueFrom} onDetails={(node) => showDetails(nodeGroup(node))}
            onPlaySample={(node, pill) => setSampleTarget({ node, pill })} />
        )}
      </details>
      <VideoCheckpointList datasetId={ds.id} payload={payload} busy={busy} details={details}
        continueTarget={continueTarget} extraSteps={extraSteps} onExtraSteps={setExtraSteps}
        onConfirmContinue={confirmContinue} onCancelContinue={() => setContinueTarget(null)}
        onDeploy={deploy} onUndeploy={undeploy} onDelete={remove} onDeleteRun={removeRun}
        onContinue={continueFrom} onDetails={showDetails} />
      {sampleTarget && (
        <VideoSampleLightbox datasetId={ds.id} target={sampleTarget}
          onClose={() => setSampleTarget(null)} />
      )}
    </div>
  )
}

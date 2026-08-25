import { useEffect, useMemo, useRef, useState } from 'react'
import {
  SKIP_LABEL, acceptLabel, defaultPicked, folderLabel, notReachedLine,
  nothingFoundLine, preflightCostLine, preflightHeadline, preflightRows,
  savingLine, skipNote, togglePicked,
} from './personPreflight.js'
import { HelpBadge } from '../../help/HelpMode'

/** 👤 The preflight of 👤 Group by person — shown at LAUNCH time, standalone or
 *  on the way into 🚀 Launch all.
 *
 *  The critique this answers, verbatim: "the first thing a user does is Launch
 *  all, so they never go through the folder scan". The sampling therefore runs
 *  by itself here, in front of the pass, and the user answers ONE question with
 *  the good folders already ticked.
 *
 *  It never groups anything on its own. Pre-ticked is not decided: every box is
 *  visible and untickable, and "Analyze everything anyway" is offered at every
 *  moment — including while the sampling is still running, where it stops the
 *  probe first (a bank runs one job at a time, so proceeding on top of a live
 *  probe would just be refused).
 *
 *  All the wording lives in personPreflight.js so `node --test` can prove it.
 */
export default function PersonPreflightDialog({
  plan: initialPlan, probing, activity, reload, onProceed, onStopProbe, onCancel,
}) {
  const [plan, setPlan] = useState(initialPlan)
  // 'probing' — the sample job is running · 'choose' — the answer is on screen
  // · 'stopping' — the user opted out mid-probe and we are waiting for the job
  // to let go of the bank before the real pass can start.
  const [phase, setPhase] = useState(probing ? 'probing' : 'choose')
  const [picked, setPicked] = useState(() => defaultPicked(preflightRows(initialPlan)))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const sawJob = useRef(false)
  const settling = useRef(false)

  const rows = useMemo(() => preflightRows(plan), [plan])
  const headline = preflightHeadline(rows)
  const nothing = nothingFoundLine(rows)
  const notReached = notReachedLine(plan)

  const kind = activity?.kind
  const jobLive = !!activity && !activity.finished && kind === 'folder-preflight'
  const jobDone = kind === 'folder-preflight' && !!activity?.finished
  if (jobLive) sawJob.current = true

  /* The probe landed → pull the fresh verdicts and pre-tick the good ones.
     `settling` guards against the double render a poll produces. */
  useEffect(() => {
    if (phase === 'choose' || settling.current) return
    // Settled either because the snapshot says so, or because a job we watched
    // run has been replaced/purged — never on a stale snapshot we never saw run.
    const settled = jobDone || (sawJob.current && !jobLive)
    if (!settled) return
    settling.current = true
    const failed = activity?.error
    Promise.resolve(reload ? reload() : plan)
      .then((fresh) => { if (fresh) setPlan(fresh) })
      .catch(() => {})
      .finally(() => {
        settling.current = false
        if (phase === 'stopping') { onProceed({ accept: [] }); return }
        if (failed) setError(`The folder check did not finish — ${failed}. `
          + 'You can still run the full analysis.')
        setPhase('choose')
      })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobDone, jobLive, phase])

  // Fresh verdicts arrived → the pre-ticked set is recomputed from them, not
  // carried over from the empty list the dialog opened with.
  useEffect(() => {
    if (phase === 'choose') setPicked(defaultPicked(preflightRows(plan)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [plan])

  const dismiss = () => { if (!busy && phase !== 'stopping') onCancel() }
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') dismiss() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [busy, phase])   // eslint-disable-line react-hooks/exhaustive-deps

  const accept = async () => {
    if (busy) return
    setBusy(true)
    try { await onProceed({ accept: picked }) } finally { setBusy(false) }
  }

  /* "Analyze everything anyway". Mid-probe it has to STOP the probe first: one
     bank runs one job, so launching the pass under a live sampling job would be
     refused — and a way out that gets refused is not a way out. */
  const skip = async () => {
    if (busy) return
    if (phase === 'probing') {
      setBusy(true)
      try { await onStopProbe() } finally { setBusy(false) }
      setPhase('stopping')
      return
    }
    setBusy(true)
    try { await onProceed({ accept: [] }) } finally { setBusy(false) }
  }

  const pickedSaving = savingLine(rows, picked)

  return (
    <div role="dialog" aria-modal="true" aria-label="Check folders before the person pass"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onMouseDown={(e) => { if (e.target === e.currentTarget) dismiss() }}>
      <div className="w-full max-w-lg max-h-[90vh] overflow-y-auto rounded-xl border border-border bg-surface-overlay p-4 shadow-2xl space-y-3 sm:p-5">
        <div>
          <h2 className="flex flex-wrap items-center gap-1.5 text-base font-bold text-content">
            👤 Before the person pass
            <HelpBadge topic="bank-person-preflight" />
          </h2>
          <p className="mt-1 text-sm text-content-muted">
            Scraped folders are usually one person each. Rather than embed every
            image to rediscover that, the bank samples a few images per folder
            first — and asks.
          </p>
        </div>

        {phase !== 'choose' ? (
          <div className="rounded-md border border-border bg-surface-raised p-3 text-sm space-y-1">
            <p className="font-semibold text-content">
              {phase === 'stopping' ? 'Stopping the check…' : 'Checking your folders…'}
            </p>
            <p className="text-content-muted">{preflightCostLine(plan)}</p>
            {activity && !activity.finished && (
              <p className="text-xs text-content-subtle">
                {activity.detail || 'sampling'}
                {activity.total > 0 ? ` — ${activity.done || 0} / ${activity.total}` : ''}
              </p>
            )}
          </div>
        ) : (
          <>
            {headline ? (
              <p className="rounded-md border border-emerald-400/40 bg-emerald-500/10 px-3 py-2 text-sm font-semibold text-emerald-200">
                {headline}
              </p>
            ) : (
              <p className="rounded-md border border-border bg-surface px-3 py-2 text-sm text-content-muted">
                {nothing}
              </p>
            )}

            {rows.length > 0 && (
              <ul className="space-y-1.5">
                {rows.map((r) => {
                  const on = picked.includes(r.subfolder)
                  const tone = r.tone === 'ok' ? 'text-emerald-300'
                    : r.tone === 'warn' ? 'text-amber-300' : 'text-content-subtle'
                  return (
                    <li key={r.subfolder || '(root)'}>
                      <label className="flex items-start gap-2 rounded-md border border-border bg-surface p-2 text-sm">
                        <input type="checkbox" className="mt-0.5" checked={on}
                          onChange={() => setPicked(togglePicked(picked, r.subfolder))} />
                        <span className="min-w-0 flex-1">
                          <span className="block break-words font-medium text-content">
                            {folderLabel(r.subfolder)}
                            {r.images > 0 && (
                              <span className="ml-1.5 text-xs font-normal text-content-subtle">
                                {r.images} images
                              </span>
                            )}
                          </span>
                          <span className={`block break-words text-xs ${tone}`}>{r.line}</span>
                        </span>
                      </label>
                    </li>
                  )
                })}
              </ul>
            )}

            {/* Every honest limit, in the order the user needs them. */}
            <div className="space-y-1 text-xs text-content-subtle">
              {pickedSaving && <p className="text-emerald-300">{pickedSaving}</p>}
              {notReached && <p>{notReached}</p>}
              <p>
                A sample of {plan?.sample_size || 15} images cannot prove a folder
                is clean — it says what those images looked like. Ticking one
                groups it as a person you can undo at any time with
                “↩ Not one person after all”.
              </p>
            </div>
          </>
        )}

        {error && (
          <div role="alert"
            className="rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2">
            <span className="block whitespace-pre-wrap break-words text-xs leading-relaxed text-red-200">
              {error}
            </span>
          </div>
        )}

        {/* 400 px: the buttons stack instead of being squeezed to slivers. */}
        <div className="flex flex-col gap-2 sm:flex-row sm:justify-end">
          <button type="button" onClick={dismiss} disabled={busy || phase === 'stopping'}
            className="order-3 rounded-md border border-border px-3 py-2 text-sm text-content-muted hover:bg-surface-raised hover:text-content disabled:opacity-50 sm:order-1">
            Cancel
          </button>
          {/* Hidden only when it would be the SAME action as the primary (no
              folder ticked): two identical buttons is not two ways out. */}
          {(phase !== 'choose' || picked.length > 0) && (
            <button type="button" onClick={skip} disabled={busy} title={skipNote(plan)}
              className="order-2 rounded-md border border-border px-3 py-2 text-sm text-content hover:bg-surface-raised disabled:opacity-50">
              {SKIP_LABEL}
            </button>
          )}
          {phase === 'choose' && (
            <button type="button" onClick={accept} disabled={busy}
              className="order-1 rounded-md bg-gradient-primary px-4 py-2 text-sm font-semibold text-gray-950 disabled:opacity-50 sm:order-3">
              {busy ? 'Starting…' : acceptLabel(picked)}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}
